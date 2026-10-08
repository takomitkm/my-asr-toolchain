r"""
CrispASR 批量转写驱动(Windows · 无显卡 CPU 服务器副本)

本文件是 crispasr-Qwen.py 的副本,为【没有显卡、只有 CPU 的机器】改过:
  · 数据面与模型路径一律走"环境变量优先,否则落在本包目录"(见 CONFIG · 外部资源)
  · --gpu-backend cpu、-t 0(= 自动取逻辑线程数的 0.75 倍,24 vCPU 的机器上就是 18;
    系数来自本机 -t 扫点,见 CONFIG · CrispASR 的线程数一节)
  · 引擎仍是 qwen3,模型仍是 q4_k;语种从 -l zh 改成 -l auto,前置判别器
    whisper-tiny + 本地 ggml-tiny.bin(不联网)
  · VAD = ggml-silero-v6.2.0.bin —— crispasr 自己的默认(crispasr_vad_cli.cpp:19-20)。
    已知洞:它对唱歌素材判 0 段、整条静默丢弃,所以判无语音的文件一律记进
    <输出目录>\no_speech.txt(时间\t原因\t相对路径),细节见 CONFIG · 模型 那节
  · 判别码不写日志(保持 --no-prints):它一个文件一条、不在 VAD 块上,但按她的要求
    纯文本输出,只看转写内容

与旧版的行为差异:
  · 一次 crispasr.exe 调用喂多个 -f,模型 / VAD 只加载一次(旧版每文件重载 1.7B 权重)
  · 批量调用失败或有文件未产出时,自动对这些文件逐个重跑以精确定位坏文件
  · 子进程通过 Windows Job Object 绑定本进程:关窗口 / 强杀 python,crispasr.exe 一并终止
  · Ctrl+C 一次 = 当前批次跑完后优雅退出,两次 = 立刻终止 crispasr 子进程;
    与建立 <输出目录>\STOP 文件等价
  · Ctrl+N 在关闭/恢复 ntfy 推送之间切换:不想收通知时不必为此重启脚本
  · 单实例锁,防止两次误启动抢同一批文件与同一块显存
  · 失败文件按相对路径隔离,不再因同名互相覆盖
  · 规则文件逐条容错,单条非法正则不再导致整份规则失效
  · 批内每 30 秒轮询一次已写出的 .txt,写完一个就立刻落盘并清理源文件;
    以前要等整批(最长两三小时)跑完才统一结算,中途被杀等于白跑
  · 规则文件按 "原词=替换词" 每行一条解析,单条非法正则只跳过该行
  · 语种 -l auto + 判别器 whisper-tiny。auto 与 zh 不是"谁更准",而是各自有已证的坏法
    (zh 在整段非中文的文件上复读崩盘;错码会把内容摘要化丢掉),逐条依据见 CONFIG · 语种
  · 待转写文件先硬链接到 <音频根>\t 下的 ASCII 名再喂给 crispasr.exe:它是 ANSI
    程序,argv 经系统 ACP(cp936) 转换,文件名里 GBK 表示不了的字符(❤ ⚡ 及不可见
    的变体选择符 U+FE0F 等)会变成 '?',路径随之失效
  · 后台启动 / 优雅停止收进本文件(--start / --stop),不再依赖数据盘上的两个 .bat,
    POSIX 下走 start_new_session,迁移时只改 CONFIG 里的路径
  · 只依赖标准库 + send2trash —— 任何 3.10+ 的 Python 都能跑,先
    python -m pip install -r requirements.txt,再 python crispasr-Qwen-cpu.py;
    原作者的生产环境用的是免安装 embeddable Python(随数据盘附带、不用管理员权限),
    --start 派生后台进程复制的是 sys.executable,所以谁启动就用谁
"""

import ctypes
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from send2trash import send2trash

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_dir(env_name, default_rel):
    val = (os.environ.get(env_name) or "").strip()
    val = os.path.expanduser(os.path.expandvars(val))
    if not os.path.isabs(val):
        val = os.path.join(PKG_DIR, val or default_rel)
    return val


# ==================== CONFIG · 外部资源(所有路径集中在这一块)====================
#
# 换机器 / 换模型 / 换数据盘只改这一段,下面所有段落都引用这里的变量,不再出现第二处
# 字面路径。本段【只搬位置、不改任何行为】:引擎选择、阈值、语种配置的说明留在下面
# 各自的小节里。
#
# 路径的取值口径 = 环境变量优先,没给就落在本包目录下(相对名按本文件所在目录解析),
# 所以整包放哪儿都能跑。CRISPASR_BIN_DIR / CRISPASR_MODEL_DIR 两个 env 名和
# assets.json 里 roots 的 env 字段是同一个,fetch_assets.py 把组件落到哪儿,这里就从
# 哪儿读;数据面两个目录用 ASR_TEXT_DIR / ASR_AUDIO_DIR(清单不往里放文件,是输入和记账)。
#
# 迁到没有显卡的机器:CRISPASR_BIN_DIR 指到 crispasr-windows-x86_64-cpu(-legacy) 的
# 解压目录、CRISPASR_GPU_BACKEND 改成 "cpu"、CRISPASR_THREADS 留 0 走自动。
# CPU 包和解压目录里要有【两个】文件才能跑:crispasr.exe 和 openblas.dll(动态链的,
# 必须和 exe 同目录 —— Windows 先搜 exe 所在目录,所以不用往 PATH 里塞)。这个包里没有 ggml-*.dll —— 静态链进 exe 了,和 CUDA 目录
# 那套(ggml.dll / ggml-base.dll / ggml-cpu.dll / ggml-cuda.dll)结构不同。
# 模型文件一个都不用重下:同一份 .gguf 两边通用。

# ---------- 数据面目录 ----------
ASR_ROOT   = Path(resolve_dir("ASR_TEXT_DIR", "txt"))
ASRSOURCE  = Path(resolve_dir("ASR_AUDIO_DIR", "audio"))

PCM_INPUT  = ASRSOURCE / "p"        # 音频输入根目录
FAILED_DIR = ASRSOURCE / "f"        # 失败隔离区
STAGE_DIR  = ASRSOURCE / "t"        # 同盘硬链接暂存区,给 crispasr 提供 ASCII 安全路径

OUT_DIR    = ASR_ROOT               # 转写文本输出目录
LOG_DIR    = ASR_ROOT / "log"
RULES_FILE = ASR_ROOT / "rules.txt"
TMPLIST    = ASR_ROOT / "tmplist.txt"
LOCK_FILE  = ASR_ROOT / ".lock"
STOP_FILE  = ASR_ROOT / "STOP"      # 存在则在下一批边界优雅退出
# VAD 判"整段无语音"的文件 rc=0、不落 .txt,转写树上不会留下任何痕迹,所以在这里记账。
NO_SPEECH_LOG = ASR_ROOT / "no_speech.txt"

EXTENSIONS = {".mp3", ".m4a", ".mp4", ".wav", ".oga", ".ogg", ".opus", ".flac", ".aac"}

# ---------- crispasr 程序 ----------
CRISPASR_BIN_DIR = resolve_dir("CRISPASR_BIN_DIR", "crispasr")
CRISPASR_EXE     = os.path.join(CRISPASR_BIN_DIR, "crispasr.exe")

# ---------- Python ----------
#
# 标准库 + send2trash 就够了,任何 3.10+ 的解释器都能跑(依赖见 requirements.txt)。
# 原作者的生产机用的是官方 embeddable Python 3.13(免安装版,VC 运行库、OpenSSL 都在
# 它自己目录里,Lib\site-packages 预先放好 send2trash),所以那台上不用装 Python、也不
# 需要管理员权限 —— 这只是背景,不是本包的前置条件。--start 派生后台进程用的是
# sys.executable(谁启动本脚本就复制谁),前台 / 后台启动都跟着解释器走。

# ---------- 模型 ----------
MODEL_DIR        = resolve_dir("CRISPASR_MODEL_DIR", "model")
CRISPASR_MODEL   = os.path.join(MODEL_DIR, "qwen3-asr-1.7b-q4_k.gguf")
# CPU 机器上就按这个 1.49 GB 的 q4_k 走(它其实是 q4_k 语言模型 + Q8_0 音频塔,
# 上游那张量化表里"Q8 tower"那一档 = 1490 MB,cos_mean 0.9998、长音频不崩)。
# 全 q8 那份 2.51 GB 在 CPU 上要多搬约 1 GB 权重/token,只会更慢,没有收益 —— 实测过
# (2026-10-02):同一份 90 s、同 -t 12,q8_0 84.3 s 对 q4_k 67.6 s(慢 24.7%),而且两份
# 文本不逐字相同(546 对 544 字),所以"换大量化换精度"在这台机器上是既慢又不可控,别动。
# 语种前置判别器与 VAD 的路径也在这里(两者的配置说明见下面 CONFIG · 语种 / CONFIG · VAD)。
CRISPASR_LID_MODEL = os.path.join(MODEL_DIR, "ggml-tiny.bin")
# VAD 用 silero v6.2.0 —— crispasr 自己的默认(examples/cli/crispasr_vad_cli.cpp:19-20
# 的下载 URL 与文件名就是 ggml-silero-v6.2.0.bin)。本机两个 silero 文件大小相同但内容不同
# (v5.1.2 md5 c8f289… vs v6.2.0 md5 ee99234…),这份脚本只认 v6.2.0。
# 口语素材上三档 VAD 打平(2026-10-02,同一份 89.84 s 中文只换 -vm):v5 切 5 段/语音合计
# 75.78 s,v6 切 7 段/75.88 s,墙钟 55.9 对 56.1 s,文本 549 对 546 字,差异只在断句与标点;
# FireRedVAD 要多花约 31% 墙钟(72.5 / 74.3 s 两次)而内容不变(去标点 545 对 542 字)。
# 上游 PERFORMANCE.md:1170-1176 那张"firered 覆盖少 8 个百分点"的表在中文口语上没复现,别引。
#
# 【已知洞:silero v6 对唱歌素材判 0 段,整条静默丢弃】(2026-10-03,一条 60 s 日推歌曲副歌):
#     firered-vad.gguf   97 字真歌词(5998 帧 / mean_prob 0.79 / speech>0.3 = 4955 帧)
#     silero-v5.1.2      59 字,仅 5 段 / 语音合计 2.60 s(覆盖 4.3%)
#     silero-v6.2.0      0 段 ⇒ 无 .txt、rc 仍是 0、日志只有 "no speech detected"
#     不加 --vad         101 字(最全,但走固定 30 s 分块,见 CONFIG · VAD 那条截断风险)
# rc=0 且不落 .txt 意味着驱动看不出失败,所以这类文件现在一律记账到 <输出目录>\no_speech.txt
# (见 record_no_speech),吞了多少看那个文件。待转库里带唱歌的上传者不止一个
# (光 p/b.3537109134608937 就 30+ 条)。
# 换成 firered 的选项【已排除,这份不改】:代价 +31% 墙钟,而且本包复原出来的 model\ 里
# 【没有】firered-vad.gguf(2,357,952 B),要换得先自己拷进 MODEL_DIR;
# xhs-asr.py(GPU 那台)用 firered,完整说明在那份的 CONFIG · VAD。
# 顺带:-vt 救不了 silero v6(0.50/0.25/0.02 三档全是 0 段),而且这个参数在 crispasr 的
# silero 路径上语义可疑(口语素材 -vt 0.95 反而从 7 段变 13 段),别拿它调。
# 要看 firered 的逐帧概率就加 --firered-vad-debug(它默认不吐段表)。
CRISPASR_VAD_MODEL = os.path.join(MODEL_DIR, "ggml-silero-v6.2.0.bin")

# ==================== CONFIG · CrispASR ====================

CRISPASR_BACKEND     = "qwen3"
CRISPASR_GPU_BACKEND = "cpu"

# ---------- 线程数(-t)----------
#
# 这台机器是 EPYC 9654 的 24 vCPU 切片、125 GB 内存、无显卡直通。
# 0 = 自动取 os.cpu_count() 的 0.75 倍(24 vCPU ⇒ 18)。这个系数不是推出来的,是拿
# 本机 8 物理核 / 16 逻辑 + 同一个 89.84 s 片段、同一个 q4_k、只改 -t 扫出来的:
#   -t 4   79.1 s   RTF 0.88
#   -t 8   58.6 s   RTF 0.65      ← 8 = 物理核数
#   -t 12  53.1 s   RTF 0.59      ← 12 = 1.5x 物理核,曲线在这里进平台
#   -t 16  53.4 s   RTF 0.59      ← 16 = 全部逻辑线程
#   -t 24  54.5 s   RTF 0.61      ┐ 超订开始略微变差
#   -t 32  54.7 s   RTF 0.61      ┘
# 两点和常见说法相反,都记在这里免得再按老直觉改回去:
#   ① 超线程在这条路上是【赚的】不是白抢 —— 8→12 快 9%。原因:qwen3 的 CPU 解码是
#      逐 token 的窄活(mul_mat 的 GEMV),吃不满一个核的向量单元也吃不满内存带宽,
#      隔壁那个逻辑线程正好填缝。"只给物理核"那条老直觉只对 AED 那种大 GEMM 成立。
#   ② 但平台是真的:超过 1.5x 物理核再堆就是白堆,还开始亏。所以默认取平台起点。
# 虚拟化有时把 SMT 藏起来(报 24 核 24 线程 = 24 个真实核),那种情况 0.75 会算出 18
# 而能用的是 24 —— 要硬性指定就把 0 改成具体数字,想给别的租户留余量就填 12(实测只
# 慢约 10%)。
# 另外:这一批数字是热缓存下的;同一台机器同一组参数,本轮【第一次】跑量到 77.9 s
# (t=8),之后重复同一个点变成 58.6 s —— 差 19 s 全是权重/磁盘首读。所以单次观测不能
# 当基线,共享切片上更要重复两三次再取值。
CRISPASR_THREADS = 0
if CRISPASR_THREADS <= 0:
    CRISPASR_THREADS = max(1, int((os.cpu_count() or 2) * 0.75))

# ---------- 为什么无显卡的机器反而该用 qwen3(AED 在这台机器上是死路)----------
#
# 机制:AED 的解码器恒定在 CPU(src/firered_asr.cpp:442 原话 "Decoder weights ALWAYS go
# to CPU"),但【编码器】是 split-load 上显卡的(:453-465,enc_gpu 为真时只把 enc.* 送
# GPU)。所以拔掉显卡对 AED 是双重打击:编码器那 1.1B 参数的 Conformer 掉回 CPU,而它本来
# 就慢的解码器(每出一个 token 都要在 CPU 上过一遍整份解码器权重)一点没变快。
# 更正一处旧说法:上游 PERFORMANCE.md:337 把 AED 解码器写成"没有 KV cache、每步 O(T²) 重算",
# 这条与出货代码不符 —— 每层每个 beam 都带一份 sa_k/sa_v 增量缓存,读码依据在 gpu 分支
# CrispASR-FireRed/README.md §5.1 ①。所以 AED 慢不能拿"没做 KV"来解释,能确定的是解码整趟
# 都在 CPU 上跑;具体卡在矩阵乘还是每步的 dispatch 没有拆开。
# 上游那张 CPU-only 表(PERFORMANCE.md:693-707,4 线程 AVX2 / 7.6 GB)量到的就是这件事:
#     FireRed ASR2 AED   0.1x(CPU,123 s) → 0.6x(T4)   —— 掉显卡差 6.5x,全表最惨
#     Qwen3 ASR 0.6B     1.7x(CPU, 6.5 s) → 4.7x(T4)   —— 掉显卡差 2.8x
# "AED 本来就跑 CPU 所以拔掉显卡不吃亏"是倒因为果:它 CPU 上跑的是【最慢的那一半】。
# 注意表里那两行是 0.6B,上游没为 1.7B 发过任何 CPU-only 数值(全表 5 处 qwen3 性能行清一色
# 0.6B),那张表在这里只用来看【排序】和【掉显卡的倍数】,具体数值看下面本机实测。
#
# ---------- 本机 CPU 实测(2026-10-02)----------
#
# 拿 crispasr-windows-x86_64-cpu(v0.8.37,exe 18,387,456 B + openblas.dll)+ 本脚本
# 解析出的同一组参数,在【本机 8 物理核 / 16 逻辑、无 GPU 参与】跑 89.84 s 中文单人口语
# 片段:RTF 0.59(t=12 起进入平台)、输出 546 字、exit 0。
# 六个线程档产出的字节数完全相同(546 字),所以 -t 不改变内容。与 GPU 路同一份文本的对拍
# 没做(GPU 侧她不测),要核就同一份 in.wav 再跑一次 --gpu-backend cuda。
#   换算:纯 CPU ≈ 每秒音频 0.6 秒 CPU;本机那块 4060 上 qwen3 记录在案的实测 RTF 是
#   0.19,所以拔掉显卡大约慢 3 倍。服务器 24 vCPU 的单核比本机弱(2.40 GHz 基频 + 共享
#   切片),线程多一半,两边大致抵消 —— 把 0.59 当【量级】用,别当承诺。
#   整库账:待转 7,786 个文件 / 907 小时音频 / 中位 427 s 每文件。按 RTF 0.59 单实例
#   约 535 小时连续计算(≈22 天)。这是【上界】:上面那一趟走的是单文件路径,每次调用
#   都要重装一遍权重和判别器;生产走的是批量多 -f 路径,固定开销按批摊,而待转文件
#   中位 427 s 远长于测试用的 90 s,所以真实 RTF 会低于 0.59 —— 低多少只能上机量。
#   ⇒ 决策含义:CPU 服务器不是"换台机器接着跑"那么轻,量级上是把几天变成两三周。
#   要压缩只有多实例分片(见下面一节);真想要速度,这批还是留在有显卡的机器上跑完划算。
#
# ---------- 同一段中文,AED 对 qwen3 的 CPU 对拍(2026-10-02)----------
#
# 30.0 s 单人口语、同一台机器、同一份 CPU 构建、都 -t 12、都带 silero v6.2.0 + whisper
# 前置判别(AED 那趟另加 --punc-model,因为它自己不出标点):
#     qwen3-q4_k  29.0 s / 26.4 s 两次   →  RTF 0.97 / 0.88
#     AED-q4_k    53.9 s / 43.2 s 两次   →  RTF 1.80 / 1.44
# ⇒ 在这台纯 CPU 机器上 AED 比 qwen3 慢 1.6-1.9 倍(两次重复同向),而【上游那张表里
#   AED 只有 0.1x 是对 4 线程 VPS 说的,别拿 0.1x 当本机数字】。
# 质量:两份文本几乎是同一句话,差异在同音字与断句(AED 多"咱…啊""来经营"几处、
# 少 11 字),没有一方出现复读、摘要或整段丢失 —— 也就是【这个样本上分不出好坏】。
# "中文必须 AED"只在 GPU 那侧有支撑,CPU 上换 AED 既慢又没有可证的质量收益,
# 所以这份副本只留 qwen3。n=1、30 s,C 级。
#
# ---------- 一次调用喂多个文件能省多少(2026-10-02 实测)----------
#
# 两段 90 s:分开跑两次 144.0 s,合一次 141.8 s ⇒ 少一次调用只省 2.2 s(1.5%)。
# 1.5 GB 权重第二次起就在 OS 页缓存里,重载几乎不要钱 —— 所以【批量不是提速手段】,
# 它只是把 28 个文件的失败隔离做整齐(生产脚本已经这么用了)。
# 更好的拟合模型:每次调用固定约 13 s(加载 + LID + VAD),音频本身约 0.45 s/s
#   (30 s 素材 26.4 s、90 s 素材 53.1 s 两点解出;逐文件内容差异大,另一个 90 s 文件
#   量到 84.4 s ⇒ 边际 0.79)。按这个模型整库 907 h 音频 + 278 次调用 ≈ 405-630 h。
#
# 内存这笔账别拿来推断速度:q4_k 权重 1.49 GB + KV 约 0.22 GB(28 层 × head_dim 128 ×
# n_kv 8 × ctx 4096,F16)每实例约 1.7-2 GB,125 GB 内存随便开多实例。占用大 ≠ 慢 ——
# qwen3 存下 28 层的 K/V 所以每 token 只算一步;AED 也存(16 层、每片解码上限 150 token),
# "AED 不留 KV、每步重算整段"是上游文档的口误,读码依据见 gpu 分支 CrispASR-FireRed/README.md
# §5.1 ①。两边占用差的是层的规模,不是有没有缓存。
# 反证一条(2026-10-02 实测):同一份 90 s、同样 -t 12,换 q8_0 要 84.3 s 而 q4_k 是
# 67.6 s —— 内存用得多 25%,反而慢 24.7%,而且输出并不逐字相同(q4 546 字 / q8 544 字)。
# 所以"内存大就随便用"在这个负载上换不来速度:瓶颈是把权重每个 token 搬一遍的带宽,
# 不是装得装不下。大内存唯一实打实的用处是给多实例腾位置(见下一节)。
#
# ---------- 多实例(收益是真的,但只有三成,不是翻倍)----------
#
# 本机(8 物理核 / 16 逻辑)实测,同样两段 90 s 音频(合计 179.8 s),只改"几个进程、
# 每个几线程":
#     1 进程 × -t 12          137.7 s  →  1.31x 实时
#     2 进程 × 各 -t 6        103.2 s  →  1.74x 实时   (+33%)
#     2 进程 × 各 -t 12       105.1 s  →  1.71x 实时   (超订到 24/16,没有额外收益)
#     4 进程 × 各 -t 4        131.9 s(另含两段 30 s)→ 1.82x 实时   (比 2 进程只多 3%,在噪声里)
# 结论两条:① 【同样的总线程数,拆成两个进程比一个进程快三成】—— 所以 -t 自动值那个
# 单实例口径不是终点;② 到 1.8x 实时就见底,再多开只是排队,方向上撞的是内存带宽而不是
# 核数(单线程解码是流式读权重,q4_k 每 token 要搬约 0.9 GB)。
# 噪声也要记一笔:同一个 90 s 文件、同一个 -t 12,三次分别量到 53.1 / 59.5 / 67.6 s
# (±13%),所以上面 2 进程 vs 4 进程那 3% 不能当结论,33% 可以。
# 做法:多实例必须各配一份 ASR_ROOT / ASRSOURCE / STAGE_DIR / LOCK_FILE(否则两份互抢
# 同一批文件和同一个单实例锁),并且【按顶层目录分片】各扫各的子树 —— 只改路径不分片
# 会转两遍同一批文件。分片代码本副本还没写,要上多实例先说一声。
# 服务器 24 vCPU 上的落法:先按单实例 -t 18 跑一天拿到本机自己的吞吐,再决定要不要
# 拆两份各 -t 9~12(预期 +30% 左右,不会更多)。
#
# ---------- 数据盘 ----------
#
# SAS HDD 顺序读约 150-300 MB/s,而 16 kHz 单声道 16 bit 的 WAV 是 32 KB/s(115 MB/小时),
# 单实例解码每秒只吃约 1 MB —— 磁盘不是瓶颈。多实例同时随机读同一块盘才可能出问题,
# 那时先看 crispasr 的墙钟是否变长,再考虑把待转文件挪到别的盘。
#
# ---------- 语种(-l)与语种判别(--lid-backend / --lid-model)----------
#
# 结论先说:默认 -l auto + whisper-tiny 前置判别(本地文件,不联网)。要退回强制中文就把
# CRISPASR_LANGUAGE 改成 "zh",下面两个 CRISPASR_LID_* 就不参与拼命令了。两条路不是
# "谁更准"的区别 —— 在混语文件上判别并不比强制 zh 准 —— 而是各自有已证的坏法:zh 在整段
# 非中文的文件上会崩,判别在混语文件上会丢内容。所以下面每一条都要读完再决定改不改。
# 全部是生产机 0.8.37 + qwen3-asr-1.7b-q4_k 实测;基准树是作者的本地 lid 基准目录
# (不在本包里) —— 45 s 组见 labels.tsv / results.tsv,120 s 组见 v3_results.tsv 与
# out/v3_*.txt(逐字文本都在,可自己核)。
#
# (1) 强制语种对 qwen3 只是一条 assistant 前缀,不是硬约束。源码 examples/cli/
#     crispasr_backend_qwen3.cpp:180-184 把 -l XX 拼成 "language <英文名><asr_text>"
#     预填到 assistant 轮。实测同一句日语歌:
#       -l zh → "どうするなら、君はそこなんだろうか。"
#       -l ja → 逐字相同
#     所以"语种填错就一定转坏"这句对 qwen3 不成立,对 whisper 才成立。
#
# (2) 但错码会把 qwen3 推进翻译模式,而后果【不是"翻译了意思还在",是摘要化+整段
#     静默丢弃】。BBC 那段中文受访者发言
#       「前几天,我跟我妈妈还好好的,结果他给你重重的捅了一刀,把你送进立贞素质教育学校里面。」
#     在 en 码下没有任何对应,被一句 "The schools promise to correct problem behavior."
#     顶掉;同段另外两处具体陈述也只有 -l zh 留下来了。
#
# (3) 信息保留度按内容类型分岔(8 段 120 s,三种配置:强制 zh / --lid-backend off 即
#     Qwen 自判 / auto+whisper-tiny):
#       纯中文口语 ×3  三种配置只差同音字和标点,内容单元 0 丢 —— 判别在这里既无害也无功
#       中英混(BBC)   -l zh 11/11 内容单元 > Qwen 自判 8/11 > 判成 en 的 7/11
#                      (另一个窗口判成 en 却什么都没丢,所以"en 必坏"也不成立)
#       日语歌 ×2      whisper-tiny 判 ja 最全 > Qwen 自判(整段译英但逐句对得上)
#                      > -l zh 全场最差:其中一首输出逐字母 romaji 加复读「想得遠」上百次,
#                      整段作废
#     ⇒ 一个文件只能贴一个标签,混语文件谁来判都保不住两种语言;LID 真正的价值只在
#       "整段都不是中文"的文件上,而 -l zh 在那种文件上是灾难 ⇒ 真要处理非中文素材,
#       比"全局开判别"更划算的是按目录分组给语种(非中文目录才 auto),但这条没实测。
#
# (4) 判别器成绩(12 样本 / 每样本 45 s):whisper-tiny 是唯一每次都给出码的,中文 5/6、
#     日语 2/4;silero-95 判不出 6/13,且判不出时的回退是坏的 —— 它拿 --lid-model 那个
#     silero 文件去走 whisper 加载器,报 "failed to load …silero-lid-lang95-f32.gguf",
#     另外会吐 zh-TW / si(僧伽罗语)/ da(丹麦语);ecapa-107 中文 5/6、日语 0/4(全给
#     欧洲小语种码)—— crispasr 官方 README 里写着 "ecapa — recommended",但它是拿朗读/
#     TTS 语料训的,唱歌直接出分布,那句 recommended 在本语料上不成立。firered-lid
#     (887M 参数 Conformer,100+ 语种含 20+ 中文方言,模型卡自报 FLEURS-82 97.18%)
#     按指示未测。
#
# (5) 判别的耗时别当理由:45 s 组量到 +1.8 s/文件,120 s 组不复现(逐文件差 −0.5 到
#     −1.7 s,判别反而略快),落在抖动里。所以"LID 有害"只剩内容层面的理由。
#
# (6) 判别只看【前 15 秒】,而且是原始音频不是语音段:CLI 把整段交给
#     crispasr_detect_language(),它截 kLidMaxSamples = 16000 * 15
#     (src/crispasr_lid.cpp:284;会挑语音段的 crispasr_lid_speech_prefix 只有 server
#     路径用)。开头是片头音乐的文件,语种等于按音乐定 —— 这是本语料最常见的形态,
#     也是判别最主要的失手点。
#
# (7) 没有置信度闸门:CRISPASR_SILERO_LID_MIN_LOGIT 只作用于 silero。whisper 路径实测
#     判到 p=0.395(中文播客判成 en)也照样往下走,没法"判得不确信就当没说"。
#
# (8) 判别结果在生产日志里看不见:`crispasr[lid]: detected 'xx' (p=…)` 这行受
#     opts.verbose = !params.no_prints 控制,而命令行带 --no-prints,所以是静默生效的。
#
# 别指望 --lid-backend probe 能让 qwen3 自己判:--help 写的是"问主模型自己",但
# qwen3 没有 self-probe,实测直接报 "this backend has no self-probe — falling back
# to whisper-tiny LID",然后没给 --lid-model 就去联网下 ggml-tiny.bin,卡满超过
# 2 分钟后 "LID failed and no -l was set — defaulting to 'en'" —— 兜底这个 en 才是
# 把音频推进翻译模式的元凶。真正"让模型自己判"只有 --lid-backend off,而它同样翻。
#
# 常用 ISO 639-1:zh 中文 / en 英语 / ja 日语 / ko 韩语 / yue 粤语 / es / fr / de /
# it / pt / ru / ar / hi / th / vi。完整 99 种见 crispasr.exe --help。
#
# 本副本按她的要求取 "auto"。两条已经核验过的连带事实(留档,不改变行为):
#   · qwen3 没有申报 CAP_LANGUAGE_DETECT(examples/cli/crispasr_backend_qwen3.cpp:48-49),
#     所以 -l auto 时框架的 whisper-tiny 前置【真跑】,并且判到的码会被写回
#     params.language(examples/cli/crispasr_run.cpp:5045-5052)—— 在 qwen3 上 auto
#     不是"让模型自己判",是"让 tiny 替它判完再喂回去"。
#   · 判别失败时兜底是硬编码的 'en'(crispasr_run.cpp:1015-1017 "defaulting to 'en'"),
#     而 en 恰恰是把 qwen3 推进翻译模式的那个码(见上面第 (2) 条)。
# 判别码【一个文件一条】,不在 VAD 块上:前置跑在分块之前(crispasr_run.cpp:5045 在前、
# :5073 的 crispasr_compute_audio_slices 在后),所以一批 28 个文件最多 28 条。
# 按她的要求保持 --no-prints ⇒ 判别码和上面那条 "LID failed" 警告都不写日志,
# 唯一的防线是 main_loop 启动前对 CRISPASR_LID_MODEL 的存在性硬检查(缺文件直接拒启)。
CRISPASR_LANGUAGE = "auto"

# ---------- LID 后端与模型(仅 CRISPASR_LANGUAGE = "auto" 时才用到)----------
#
# 取 whisper | silero | ecapa | firered | probe | off。留空 = 用 crispasr 的默认
# whisper,但 CRISPASR_LID_MODEL 仍要指到本地文件,否则它每个文件联网下载
# ggml-tiny.bin 超时(约 30 s)后按 crispasr_run.cpp:1017 强制 en 兜底 ——
# 所以 main_loop 启动时把"auto 但模型缺失"判为致命错误。
CRISPASR_LID_BACKEND = "whisper"
# 模型路径 CRISPASR_LID_MODEL 在文件顶部 CONFIG · 外部资源 那一段(ggml-tiny.bin)。

# ---------- VAD 说明(-vm 的路径在文件顶部 CONFIG · 外部资源)----------
#
# 置空字符串则不传 -vm,由 crispasr 自行到 ~/.cache/crispasr/ 找,
# 找不到就联网下载(下载失败时 VAD 静默失效,退回固定分块)。
# 填绝对路径即可绕开下载,指向本机已有的模型文件。
# 也接受特殊值 "webrtc":纯算法 VAD,无需任何权重文件。
# 也接受关键字 auto | silero | firered | marblenet | whisper-vad,但给关键字等于
# 走【联网下载】那条路,本机拉不动 = 每批白等一次超时,所以一律填绝对路径。
# 路径 CRISPASR_VAD_MODEL 在文件顶部 CONFIG · 外部资源 那一段(ggml-silero-v6.2.0.bin,
# 全文只有那一处赋值)。

# ---------- 分块上限(--chunk-seconds)----------
#
# qwen3 属于基于 LLM 的后端,即使开着 --vad 也仍按此值分块
# (避免 KV cache 随音频时长增长而爆显存)。默认 30。
VAD_MAX_SEGMENT_SEC = 30

# 单次调用喂几个文件。模型只加载一次,显存占用不随此值增长(推理是顺序的)。
# 有了下面的批内轮询结算,调大它不再意味着中断时要重跑整批,只影响日志粒度。
BATCH_SIZE = 28

# 批内轮询间隔:每这么多秒去看一眼 crispasr 已经写出的 .txt,写完一个就立刻
# 落盘并清理源文件,而不是等整批跑完(一批可达 1~3 小时)才统一结算。
BATCH_POLL_SEC = 30

# ==================== CONFIG · 其他 ====================

LOG_LEVEL      = logging.INFO
NTFY_TOPIC_URL = os.environ.get("NTFY_TOPIC_URL", "")

REPEAT_FILTER_MAX_LINE = 4000   # 超过此长度的行跳过带反向引用的正则,规避回溯爆炸

# ==================== INIT ====================

for _p in [ASR_ROOT, OUT_DIR, LOG_DIR, PCM_INPUT, FAILED_DIR, STAGE_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

os.environ["CRISPASR_KV_ON_CPU"] = "0"
os.environ["CRISPASR_GGUF_MMAP"] = "0"
os.environ["PATH"] = CRISPASR_BIN_DIR + os.pathsep + os.environ.get("PATH", "")

# ==================== LOGGER ====================
#
# 后台启动(--start)时 stdout / stderr 是文件不是控制台。实测这样一来 Python 改用系统
# 代码页(本机 cp936)编码:同一批日志里 crispasr_*.log 是 UTF-8、console_*.log 是 GBK,
# 而且 crispasr 或文件名里一旦出现 GBK 表示不了的字符,logging 就会抛 UnicodeEncodeError。
# 统一改成 UTF-8;真控制台(isatty)不动,那条路径 Python 走 WriteConsoleW 本来就正确。
for _s in (sys.stdout, sys.stderr):
    try:
        if _s is not None and not _s.isatty():
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"crispasr_{log_timestamp}.log"
logger = logging.getLogger("crispasr")
logger.setLevel(LOG_LEVEL)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh = logging.FileHandler(log_file, encoding="utf-8")
_fh.setFormatter(_fmt)
_ch = logging.StreamHandler(sys.stderr)
_ch.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_ch)
logger.propagate = False

# ==================== 单实例锁 ====================

_lock_fh = None


def acquire_lock() -> bool:
    global _lock_fh
    try:
        _lock_fh = open(LOCK_FILE, "a+b")
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(_lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(_lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except Exception:
        if _lock_fh:
            _lock_fh.close()
            _lock_fh = None
        return False


def release_lock():
    global _lock_fh
    if not _lock_fh:
        return
    try:
        if os.name == "nt":
            import msvcrt
            _lock_fh.seek(0)
            msvcrt.locking(_lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
    except Exception:
        pass
    try:
        _lock_fh.close()
    except Exception:
        pass
    _lock_fh = None


# ==================== Job Object:子进程随父进程消亡 ====================
#
# 关闭控制台窗口时 Windows 走 CTRL_CLOSE_EVENT,宽限期后强杀 python,
# finally / atexit 都不保证执行 —— 那条路径上只有内核级的 Job 能兜住,
# 否则 crispasr.exe 会成为孤儿继续占着显存。

_JOB_HANDLE = None
_JOB_TRIED = False

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JobObjectExtendedLimitInformation = 9


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong)]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32)]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _get_job():
    """惰性创建全局 Job;句柄必须常驻,一旦被回收 job 内进程立即被杀。"""
    global _JOB_HANDLE, _JOB_TRIED
    if _JOB_TRIED:
        return _JOB_HANDLE
    _JOB_TRIED = True
    if os.name != "nt":
        return None
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = ctypes.c_void_p
        job = k32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW")
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(
            ctypes.c_void_p(job), _JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info),
        ):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject")
        _JOB_HANDLE = job
    except Exception as e:
        logger.warning(f"Job Object 创建失败,子进程将只依赖常规退出清理:{e}")
    return _JOB_HANDLE


def bind_child(proc: subprocess.Popen):
    job = _get_job()
    if not job:
        return
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = getattr(proc, "_handle", None)
        if handle is None:
            k32.OpenProcess.restype = ctypes.c_void_p
            handle = k32.OpenProcess(0x0100 | 0x0001, False, proc.pid)
        k32.AssignProcessToJobObject(ctypes.c_void_p(job),
                                     ctypes.c_void_p(int(handle)))
    except Exception as e:
        logger.warning(f"子进程绑定 Job 失败:{e}")


# ==================== Ctrl+C ====================
#
# 一次 Ctrl+C:等当前批次跑完再退,已产出的结果照常落盘、源文件照常清理。
# 两次 Ctrl+C:立刻杀掉子进程并退出,本批次作废(源文件仍在 p\,下轮重跑)。
# 与 STOP 文件等价,主循环在每个批次边界检查同一个条件。
# 处理器内不调用 logger:可能与主线程争同一把日志锁。

_ctrl_c_count = 0
_stop_requested = False
_current_proc = None

# 子进程自成一个进程组:控制台的 Ctrl+C 不再直达 crispasr,
# 否则第一次按键就会把整批打掉,拿不到任何结果。
_POPEN_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0


def _on_ctrl_c(signum, frame):
    global _ctrl_c_count, _stop_requested
    _ctrl_c_count += 1
    if _ctrl_c_count == 1:
        _stop_requested = True
        sys.stderr.write("[Ctrl+C] 当前批次结束后停止;再按一次立即终止\n")
    else:
        _stop_requested = True
        sys.stderr.write("[Ctrl+C] 立即终止子进程\n")
        try:
            if _current_proc is not None and _current_proc.poll() is None:
                _current_proc.kill()
        except Exception:
            pass


def install_ctrl_c_handler():
    signal.signal(signal.SIGINT, _on_ctrl_c)


def stop_requested() -> bool:
    return _stop_requested or STOP_FILE.exists()


# ==================== NTFY(urllib,无第三方依赖)====================

# Ctrl+N 开关。单独一个线程轮询按键:主线程跑批时阻塞在 communicate(),
# 若只在批次边界查键,这次按键要等下一批才生效,而本批结束的那条通知已发出。
# 无控制台(分离启动)时 kbhit 恒为 0,线程空转,不影响主流程。

_notify_muted = False


def _key_watcher():
    global _notify_muted
    try:
        import msvcrt
    except ImportError:
        return
    while True:
        try:
            ch = msvcrt.getwch() if msvcrt.kbhit() else None
        except Exception:
            return
        if ch is None:
            time.sleep(0.2)
            continue
        if ch == "\x0e":
            _notify_muted = not _notify_muted
            sys.stderr.write("[Ctrl+N] 通知已关闭\n" if _notify_muted
                             else "[Ctrl+N] 通知已恢复\n")


def start_key_watcher():
    threading.Thread(target=_key_watcher, daemon=True).start()


def notify(msg: str):
    if not NTFY_TOPIC_URL or _notify_muted:
        return
    try:
        req = urllib.request.Request(
            NTFY_TOPIC_URL,
            data=msg.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except Exception as e:
        logger.warning(f"ntfy 推送失败:{e}")


# ==================== RULES ====================

def load_rules(path: Path):
    """逐条容错:单条非法正则不应导致整份规则失效。"""
    rules_list = []
    if not path.exists():
        logger.info(f"规则文件不存在:{path}")
        return rules_list
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except Exception as e:
        logger.error(f"读取规则文件失败:{e}")
        return rules_list

    bad = 0
    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        pat, repl = line.split("=", 1)
        try:
            rules_list.append((re.compile(pat.strip()), repl.strip()))
        except re.error as e:
            bad += 1
            logger.warning(f"规则第 {lineno} 行正则非法,已跳过:{e}")
    logger.info(f"加载规则 {len(rules_list)} 条" + (f",跳过 {bad} 条" if bad else ""))
    return rules_list


def apply_rules(text: str, rules_list: list) -> str:
    for pattern, repl in rules_list:
        text = pattern.sub(repl, text)
    return text


# ==================== 幻觉过滤 ====================

_REPEAT_SHORT = re.compile(r'(.{1,20}?)\1{5,}')
_REPEAT_LONG  = re.compile(r'(.{10,50}?)\1{3,}')


def remove_repetitions(text: str) -> str:
    """按行处理并限长:带反向引用的正则在超长单行上会回溯爆炸。"""
    processed = []
    for line in text.splitlines():
        if len(line) <= REPEAT_FILTER_MAX_LINE:
            line = _REPEAT_SHORT.sub(r'\1', line)
            line = _REPEAT_LONG.sub(r'\1', line)
        processed.append(line)

    dedup = []
    for line in processed:
        if dedup and line.strip() and line == dedup[-1]:
            continue
        dedup.append(line)
    return "\n".join(dedup).strip()


def postprocess(raw: str, rules_list: list) -> str:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return ""
    return apply_rules(remove_repetitions("\n".join(lines) + "\n"), rules_list)


# ==================== 文件扫描与分组 ====================

def build_file_groups() -> dict:
    """按 PCM_INPUT 下第一级子目录分组,组内按 mtime 升序。"""
    groups = {}
    for f in PCM_INPUT.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in EXTENSIONS:
            continue
        parts = f.relative_to(PCM_INPUT).parts
        groups.setdefault("p" if len(parts) == 1 else parts[0], []).append(f)

    for key in groups:
        groups[key].sort(key=lambda p: p.stat().st_mtime)
    groups = {k: groups[k] for k in sorted(groups)}

    total = sum(len(v) for v in groups.values())
    try:
        with open(TMPLIST, "w", encoding="utf-8") as tf:
            tf.write(f"# 生成时间:{datetime.now()}  总计:{total} 个文件\n\n")
            for key, files in groups.items():
                tf.write(f"# [{key}] → {key}.txt  共 {len(files)} 个\n")
                for f in files:
                    tf.write(f"  {f}\n")
                tf.write("\n")
    except Exception as e:
        logger.warning(f"写入队列文件失败:{e}")

    logger.info(f"本次队列:{len(groups)} 个分组,共 {total} 个文件")
    for key, files in groups.items():
        logger.info(f"  [{key}] {len(files)} 个文件 → {key}.txt")
    return groups


def quarantine(source_file: Path):
    """按相对路径隔离失败文件,保留目录结构以免不同子目录的同名文件互相覆盖。"""
    try:
        rel = source_file.relative_to(PCM_INPUT)
    except ValueError:
        rel = Path(source_file.name)
    dest = FAILED_DIR / rel
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(source_file), str(dest))
        logger.info(f"已隔离至:{dest}")
    except Exception as e:
        logger.error(f"隔离失败 {source_file.name}:{e}")


def trash(source_file: Path):
    try:
        send2trash(str(source_file))
    except Exception as e:
        logger.error(f"删除失败 {source_file.name}:{e}")


# ==================== ASCII 暂存 ====================

_STAGE_PREFIX = "asr"

_NO_SPEECH_RE = re.compile(r"no speech detected in '([^']*)'")


def sweep_stage():
    """清掉上一轮被强杀时残留的硬链接,只认本脚本生成的命名。"""
    n = 0
    for p in STAGE_DIR.iterdir():
        if not (p.is_file() and p.name.startswith(_STAGE_PREFIX)
                and p.suffix.lower() in EXTENSIONS):
            continue
        try:
            p.unlink()
            n += 1
        except OSError as e:
            logger.warning(f"清理暂存区失败 {p.name}:{e}")
    if n:
        logger.info(f"清理暂存区残留硬链接 {n} 个")


def stage_paths(files: list) -> dict:
    """同卷硬链接零拷贝;失败退回原路径,最坏等于未修复前的行为。"""
    staged = {}
    for i, f in enumerate(files):
        link = STAGE_DIR / f"{_STAGE_PREFIX}{i}{f.suffix.lower()}"
        try:
            link.unlink(missing_ok=True)
            os.link(str(f), str(link))
            staged[f] = link
        except OSError as e:
            logger.warning(f"硬链接失败,改用原路径 {f.name}:{e}")
            staged[f] = f
    return staged


@contextmanager
def staged_inputs(files: list):
    staged = stage_paths(files)
    try:
        yield {f: str(p) for f, p in staged.items()}
    finally:
        for p in staged.values():
            if p.parent != STAGE_DIR:
                continue
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                logger.warning(f"移除暂存硬链接失败 {p.name}:{e}")


# ==================== 转写 ====================

def _base_cmd():
    cmd = [
        CRISPASR_EXE,
        "--backend",       CRISPASR_BACKEND,
        "-m",              CRISPASR_MODEL,
        "-l",              CRISPASR_LANGUAGE,
        "--vad",
        "--chunk-seconds", str(VAD_MAX_SEGMENT_SEC),
        "--gpu-backend",   CRISPASR_GPU_BACKEND,
        "--no-prints",
        "-t",              str(CRISPASR_THREADS),
        "--no-timestamps",
        "-ml",             "0",
        "-otxt",
    ]
    if CRISPASR_VAD_MODEL:
        cmd += ["-vm", CRISPASR_VAD_MODEL]
    if CRISPASR_LANGUAGE == "auto":
        cmd += ["--lid-backend", CRISPASR_LID_BACKEND or "whisper",
                "--lid-model", CRISPASR_LID_MODEL]
    return cmd


def _run(cmd, on_tick=None):
    """执行 crispasr。不设超时:单个音频跑一两小时属正常工况。

    on_tick 给出时,标准错误改由子线程排空,主线程每 BATCH_POLL_SEC 秒回调一次,
    让调用方能在批次还在跑的时候就取走已产出的结果。
    """
    global _current_proc
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            encoding="utf-8", errors="replace", creationflags=_POPEN_FLAGS,
        )
    except FileNotFoundError:
        return -1, f"可执行文件未找到:{CRISPASR_EXE}"
    except Exception as e:
        return -1, str(e)

    _current_proc = proc
    bind_child(proc)
    box = {}
    try:
        if on_tick is None:
            box["err"] = proc.communicate()[1]
        else:
            th = threading.Thread(
                target=lambda: box.__setitem__("err", proc.communicate()[1]),
                daemon=True)
            th.start()
            while th.is_alive():
                th.join(BATCH_POLL_SEC)
                if not th.is_alive():
                    break
                try:
                    on_tick()
                except Exception as e:
                    logger.error(f"批内结算出错:{e}")
            th.join()
    finally:
        _current_proc = None
    err = "\n".join(l for l in (box.get("err") or "").splitlines() if l.strip())
    return proc.returncode, err


def transcribe_batch(files: list, on_done=None):
    """
    一次调用转写多个文件。

    -f 与 -of 均可重复,且 -of 给出时数量必须与 -f 相等(whisper.cpp 沿袭的契约)。
    输出写进临时目录,不污染音频目录。

    on_done(file_path, raw_text) 给出时,crispasr 每写完一个文件就在那个轮询点
    回调一次。判据是"下标更大的 .txt 已经出现"⇒ 下标更小的那个必然已经写完并
    关闭,以此避开读到半截的文件;最新的那一个留到批次结束再取。中间缺号(空结
    果可能不落文件)不会挡住前面已完成的结算。

    返回 (results, error):
        results —— {Path: 原始文本};未产出 .txt 的文件不出现在其中,
                   但 VAD 判为"整段无语音"的会以 "" 计入(合法空转写)
        error   —— 进程级错误信息,正常退出为 None
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        bases = {f: os.path.join(tmpdir, f"o{i}") for i, f in enumerate(files)}
        results = {}
        cursor = [0]
        staged_of = {}

        def last_written():
            last = -1
            for i, f in enumerate(files):
                if os.path.exists(bases[f] + ".txt"):
                    last = i
            return last

        def flush(upto):
            while cursor[0] < upto:
                i = cursor[0]
                cursor[0] += 1
                f = files[i]
                out_txt = bases[f] + ".txt"
                if not os.path.exists(out_txt):
                    continue
                try:
                    with open(out_txt, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read().strip()
                except Exception as e:
                    logger.error(f"读取输出失败 {f.name}:{e}")
                    continue
                results[f] = text
                if on_done:
                    on_done(f, text)

        def on_tick():
            k = last_written()
            if k > 0:
                flush(k)

        cmd = _base_cmd()
        with staged_inputs(files) as paths:
            staged_of.update({os.path.normcase(p): f for f, p in paths.items()})
            for f in files:
                cmd += ["-f", paths[f]]
            for f in files:
                cmd += ["-of", bases[f]]

            rc, err = _run(cmd, on_tick=on_tick if on_done else None)

        flush(len(files))

        # VAD 判"整段无语音"的文件合法地不落 .txt(crispasr_run.cpp:1272 只打一行
        # warning,rc 仍是 0)。按空转写结算,否则它会掉进逐文件重跑,白付一次模型加载。
        for name in _NO_SPEECH_RE.findall(err or ""):
            f = staged_of.get(os.path.normcase(name))
            if f is not None and f not in results:
                logger.info(f"VAD 判无语音:{f.name}")
                results[f] = ""
                if on_done:
                    on_done(f, "")

        if rc != 0:
            return results, f"CrispASR 退出码 {rc}:{err or '无标准错误输出'}"
        return results, None


def transcribe_one(file_path: Path):
    """
    单文件转写,用于批量失败后的精确定位。
    返回 (text, error);text 为 "" 表示合法的空转写。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "out")
        with staged_inputs([file_path]) as paths:
            cmd = _base_cmd() + ["-f", paths[file_path], "-of", base]
            rc, err = _run(cmd)
        if rc != 0:
            return None, f"CrispASR 退出码 {rc}:{err or '无标准错误输出'}"

        out_txt = base + ".txt"
        if not os.path.exists(out_txt):
            return "", None
        try:
            with open(out_txt, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read().strip(), None
        except Exception as e:
            return None, f"读取输出文件失败:{e}"


# ==================== 主处理 ====================

class Stats:
    def __init__(self):
        self.ok = 0
        self.failed = []
        self.stopped = False
        self.batch_ok = True     # 批量模式一旦被判定不可用,后续全走单文件


def emit(out_txt: Path, source_file: Path, text: str) -> bool:
    """写入一条转写记录。输出文件被独占打开等情况下不终止整轮。"""
    try:
        with open(out_txt, "a", encoding="utf-8") as f:
            f.write(f"title:{source_file.relative_to(PCM_INPUT)}\n")
            f.write(text)
            f.write("\n\n")
        return True
    except Exception as e:
        logger.error(f"写入 {out_txt.name} 失败,保留源文件:{e}")
        return False


def record_no_speech(source_file: Path, reason: str) -> None:
    """吞掉的文件必须留下痕迹:VAD 判无语音时 crispasr 返回码 0 且不落 .txt,
    下游只看转写树的话,这条内容就等于无声消失。记账失败不能拖垮本轮。"""
    try:
        with open(NO_SPEECH_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{reason}\t"
                     f"{source_file.relative_to(PCM_INPUT)}\n")
    except Exception as e:
        logger.warning(f"无语音记账失败 {source_file.name}:{e}")


def settle(out_txt: Path, source_file: Path, raw: str, rules_list: list,
           stats: Stats):
    """把一份原始转写落盘并处置源文件。"""
    text = postprocess(raw, rules_list)

    # 空转写(纯静音 / 纯音乐)是合法结果:直接删除,不写输出、不计失败
    if not text:
        record_no_speech(source_file, "vad" if not raw.strip() else "postprocess")
        logger.info(f"空输出,删除:{source_file.name}")
        trash(source_file)
        return

    if not emit(out_txt, source_file, text):
        stats.failed.append((source_file.name, "写输出失败"))
        return

    logger.info(f"完成:{source_file.name}")
    stats.ok += 1
    trash(source_file)


def process_batch(out_txt: Path, batch: list, rules_list: list, stats: Stats):
    batch = [f for f in batch if f.exists()]
    if not batch:
        return

    pending = batch
    if stats.batch_ok and len(batch) > 1:
        logger.info(f"批量转写 {len(batch)} 个文件")
        started = time.time()
        settled = set()
        settled_lock = threading.Lock()

        def on_done(f, raw):
            with settled_lock:
                if f in settled:
                    return
                settled.add(f)
            settle(out_txt, f, raw, rules_list, stats)

        results, error = transcribe_batch(batch, on_done=on_done)
        logger.info(f"批次耗时 {(time.time() - started) / 60:.1f} min,"
                    f"产出 {len(results)}/{len(batch)}")
        notify(f"asr批次耗时 {(time.time() - started) / 60:.1f} min,")

        for f in batch:
            if f in results:
                with settled_lock:
                    if f in settled:
                        continue
                    settled.add(f)
                settle(out_txt, f, results[f], rules_list, stats)

        pending = [f for f in batch if f not in results and f.exists()]
        if error:
            logger.warning(f"批量调用异常:{error}")
        if pending and not results:
            # 整批零产出:多半是批量调用形式本身不被接受,本轮后续不再尝试
            logger.warning("批量调用零产出,本轮后续改为逐文件模式")
            stats.batch_ok = False
        elif pending:
            logger.info(f"{len(pending)} 个文件未产出,逐个重跑以定位")

    for f in pending:
        if stop_requested():
            logger.info("已请求停止,本批未产出的文件留在原位下轮再跑")
            return
        if not f.exists():
            continue
        logger.info(f"单独处理:{f.name}")
        started = time.time()
        raw, error = transcribe_one(f)
        if error:
            logger.error(f"FAILED: [{f.name}] {error}")
            stats.failed.append((f.name, error))
            quarantine(f)
            continue
        logger.info(f"单文件耗时 {(time.time() - started) / 60:.1f} min")
        settle(out_txt, f, raw, rules_list, stats)


def process_group(group_key: str, file_list: list, rules_list: list,
                  stats: Stats):
    out_txt = OUT_DIR / f"{group_key}.txt"
    logger.info(f"分组 [{group_key}]:{len(file_list)} 个文件 → {out_txt.name}")

    for i in range(0, len(file_list), BATCH_SIZE):
        if stop_requested():
            logger.info("检测到停止请求(Ctrl+C 或 STOP 文件),停止处理")
            stats.stopped = True
            return
        process_batch(out_txt, file_list[i:i + BATCH_SIZE], rules_list, stats)


# ==================== MAIN ====================

def main_loop() -> int:
    if not os.path.isfile(CRISPASR_EXE):
        logger.critical(f"可执行文件不存在:{CRISPASR_EXE}")
        return 2
    if not os.path.isfile(CRISPASR_MODEL):
        logger.critical(f"模型文件不存在:{CRISPASR_MODEL}")
        return 2
    if (CRISPASR_VAD_MODEL and CRISPASR_VAD_MODEL != "webrtc"
            and not os.path.isfile(CRISPASR_VAD_MODEL)):
        # 不致命:crispasr 会退回固定分块继续跑,但转写会在语句中途被切断
        logger.warning(f"VAD 模型不存在:{CRISPASR_VAD_MODEL}(VAD 将失效)")
    if CRISPASR_LANGUAGE == "auto":
        # 见 CONFIG · 语种:缺 LID 模型时 crispasr 不报错,而是每个文件联网超时
        # 约 30 秒后按 en 硬转。13k 个文件 = 上百小时白等 + 整批语种错配,
        # 所以这里判为致命,而不是让它静默降级。
        if not (CRISPASR_LID_MODEL and os.path.isfile(CRISPASR_LID_MODEL)):
            logger.critical(
                f'-l auto 需要 LID 模型,但文件不存在:{CRISPASR_LID_MODEL or "(未配置)"}\n'
                f"  要么填对 CRISPASR_LID_MODEL,要么把 CRISPASR_LANGUAGE 改成具体语种(如 \"zh\")")
            return 2
    if STOP_FILE.exists():
        try:
            STOP_FILE.unlink()
        except Exception:
            pass
    sweep_stage()

    rules = load_rules(RULES_FILE)
    groups = build_file_groups()
    if not groups:
        logger.info("队列为空,无文件需要处理")
        notify("asr successful 😀")
        return 0

    stats = Stats()
    for group_key, file_list in groups.items():
        process_group(group_key, file_list, rules, stats)
        if stats.stopped:
            break

    if stats.failed:
        logger.error(f"失败 {len(stats.failed)} 个:")
        for fname, err in stats.failed:
            logger.error(f"  {fname} | {err}")
    else:
        logger.info("所有文件处理成功")

    notify("asr stopped 😀" if stats.stopped else "asr successful 😀")
    return 1 if stats.failed else 0


# ==================== 后台启动 / 优雅停止 ====================
#
# 这一段替代生产机上原来的 run-crispasr.bat 和 stop-crispasr.bat。那两个 .bat
# 写死了 python.exe 的绝对路径、PowerShell 调用和反斜杠,换机器或迁 Linux 都得重
# 写一遍;收进本文件后启动方式和配置在同一处,Windows / POSIX 各走各的进程分离参数。
#
#   python crispasr-Qwen-cpu.py            前台跑(关窗口即断,
#                                      但 Job Object 会连带杀掉 crispasr.exe,不留孤儿占显存)
#   python crispasr-Qwen-cpu.py --start    后台跑,控制台输出重定向
#                                      到 <输出目录>\log\console_*.log
#   python crispasr-Qwen-cpu.py --stop     建 STOP 标志,实例在
#                                      下一批边界优雅退出
#
# --start 先抢一次锁再放掉,只为把"已经有实例在跑"报在当场,而不是让后台子进程
# 静默起一个然后自己退掉。真正防并发仍靠子进程里的那次 acquire_lock。

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


def spawn_background() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_h = open(LOG_DIR / f"console_{ts}.out", "a", encoding="utf-8", errors="replace")
    err_h = open(LOG_DIR / f"console_{ts}.log", "a", encoding="utf-8", errors="replace")
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": out_h,
        "stderr": err_h,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__)], **kwargs)
    print(f"已后台启动:pid={proc.pid}")
    print(f"控制台输出:{err_h.name}")
    print(f"停止:{sys.executable} {os.path.abspath(__file__)} --stop")
    return 0


def cmd_start() -> int:
    if not acquire_lock():
        print(f"已有实例在运行(锁:{LOCK_FILE}),本次不启动")
        return 4
    release_lock()
    return spawn_background()


def cmd_stop() -> int:
    STOP_FILE.write_bytes(b"")
    print(f"已建立停止标志:{STOP_FILE}")
    print("在跑的实例会转完当前文件后退出并释放锁;下次启动会自动删掉该标志。")
    return 0


# ==================== ENTRY ====================

if __name__ == "__main__":
    _arg = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    _USAGE = ("用法: python crispasr-Qwen-cpu.py [--start | --stop]\n"
              "  (依赖: 标准库 + send2trash,先 python -m pip install -r requirements.txt)\n"
              "  (无参数)  前台转写整个队列\n"
              "  --start   后台转写(控制台输出 -> <输出目录>\\log\\console_*.log)\n"
              "  --stop    让在跑的实例在下一批边界退出\n")
    if _arg in ("--help", "-h", "/?"):
        print(_USAGE)
        sys.exit(0)
    if _arg in ("--stop", "/stop"):
        sys.exit(cmd_stop())
    if _arg in ("--start", "--bg", "/start"):
        sys.exit(cmd_start())
    if len(sys.argv) > 1:
        # 不带参数就会直接开跑整条队列(并且结算后会删源文件),拼错的参数必须拦下
        sys.stderr.write(_USAGE)
        sys.exit(2)
    code = 3
    if not acquire_lock():
        logger.critical("已有实例在运行(锁:.lock),本次退出")
        sys.exit(4)
    install_ctrl_c_handler()
    start_key_watcher()
    try:
        code = main_loop()
    except KeyboardInterrupt:
        logger.info("用户中断")
        code = 130
    except Exception as e:
        logger.critical(f"未捕获异常:{e}", exc_info=True)
    finally:
        release_lock()
        logging.shutdown()
    sys.exit(code)
