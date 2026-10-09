r"""
CrispASR 批量转写驱动(Windows) —— FireRed AED 主引擎 + qwen3 兜底,由 crispasr-Qwen.py 复制而来
(10-08 起改用本名,旧名 xhs-asr.py;名字里的 qwen 指兜底那台引擎,主引擎仍是 FireRed AED)

本文件与 crispasr-Qwen.py 的差异在引擎组合、VAD、标点、语种处理和"换引擎"这件事由谁
来做这几处;队列、硬链接暂存、批内结算、单实例锁、Job Object、通知全部原样沿用。

  · 主引擎 FireRedASR2-AED(小红书 FireRedTeam,Conformer 编码器 + 注意力解码器,
    1.1 B 参数,Apache-2.0)。它没有 LLM 解码器,结构上不可能把一段话概括成一句,所以
    中文长录音的信息保留度比 qwen3 稳;代价是语种覆盖窄、而且慢(见 CONFIG · 为什么 AED
    这条路慢:本机实测 RTF 0.93 对 qwen3 的 0.19)。
  · 语种是它唯一的短板,所以本副本【两台引擎都装】,按语种分工:
      主 aed    —— 默认所有文件走它
      兜底 qwen3 —— 只接 whisper-tiny 前置判别(CRISPASR_LID_BACKEND="whisper")判到
                    AED 覆盖范围外的文件
    为什么必须换引擎而不是改参数:AED 自带 LID 头,覆盖 中文(+约20种汉语方言)/ 英语 /
    粤语,没有日语这类位置,而且适配器源码明写"offers no per-request language override,
    so `-l <lang>` cannot steer it"(crispasr_backend_firered_asr.cpp:49-57)—— 给它
    `-l ja` 不是"换个语种转",是让它拿汉字凭空编造(issue #199)。它那份 gguf 的 8,667 个
    token 里假名数为 0(直接数出来的:含ぁ-ン 的 0 个、含汉字的 7,566 个)。
    机制上能做到的是 qwen3:它的 `-l` 被拼成一条 assistant 前缀,真能换语种。
    反例仍在(lidbench,同一首日语歌的 120 s 切片 samples2/t07_ayts.wav):qwen3 强制
    `-l zh` → 1,147 字逐字母 romaji + 复读「想得遠」48 次,整段作废(out/v3_zh_t07_ayts.txt);
    `-l auto` 挂 whisper-tiny 前置判别判到 ja → 547 字可用假名(out/v3_lid_t07_ayts.txt);
    什么都不给让模型自判 → 英文歌词 325 字(out/v3_qwen_t07_ayts.txt)。
    所以兜底那一趟也照旧 `-l auto` + 前置判别,不由驱动把判到的码传下去(那等于把筛子的
    错误固化);而 AED 这一路的 `-l` 一律 "auto",不写死 zh。
  · 筛子怎么工作(实现在 _lid_map / transcribe_batch / process_batch):
      判到 AED 范围外 → 那份 AED 文本【不落盘、不结算】,源文件留在原位,只记进 flagged;
      一批跑完把 flagged 攒成一批换 qwen3 重跑(一次模型加载吃掉整批);逐文件重跑阶段
      认得 flagged,直接上兜底引擎,不再白喂 AED 一次。
      因为批内结算是实时的(每 30 秒轮询、写完一个就删源文件),标准错误必须【逐行实时】
      读,_run 因此多了 live 参数(以前是 communicate() 等进程结束才一次性拿到)。
      语种结论与文件的对应靠"批量循环是单线程顺序跑的"(crispasr_run.cpp:4974-4981),
      并额外处理两类错位:读音频就失败的文件(它不产生判别行,但 error 行带路径)和
      走到判别却没出码的文件(占一个槽位)。对不上号时整批【不采信】筛子。
  · 筛子的失手方式要知道:whisper-tiny 只看【前 15 秒原始音频】、whisper 路径【没有置信度
    闸门】,lidbench 上它的召回是 中文 5/6、日语 2/4。所以【假阴性会漏】—— 判成 zh 的外语
    文件仍会被 AED 编造一段话并正常落盘,这一层没有兜底;假阳性的代价只是白跑一遍 qwen3
    (它反而快 5 倍)。前置判别每文件都要重装一次 whisper-tiny(crispasr_run.cpp:1021 的
    crispasr_lid_free_cache() 是 #35 显存修补),耗时按实测落在抖动里(见 CONFIG · 语种 (5))。
  · VAD 用 FireRedVAD(2,357,952 B)—— 小红书自家模型,FireRedASR2S 把唱歌算进 voice,
    所以它是三档里唯一不会把唱歌素材整条吞掉的那个;代价是口语素材上多约 31% 墙钟,
    而内容三家打平。两个引擎共用同一个 VAD,依据与实测见 CONFIG · VAD。
  · 加 FireRedPunc(q8_0,104 MB)做书面标点还原,但【只对 AED 这条路加】:它是 CLI 层的
    后处理(crispasr_run.cpp:322 apply_punc_model),而 qwen3 有能力位
    CAP_PUNCTUATION_TOGGLE(crispasr_backend_qwen3.cpp:56),本来就自己出标点,再叠一层
    就是二次打标 —— 源码注释写得很直白"already emit punctuation should not get a second
    pass"(crispasr_punctuation_policy.h)。反过来 AED 不给它 --punc-model 就会被
    crispasr_should_auto_enable_punctuation() 自动置成 "auto",那是【联网下载】的路径,
    本机拉不到会每批白等超时,所以必须显式给本地路径。
  · 边界处理【不加】--lcs-dedup on / --chunk-overlap:查了源码,这条路径上它们是空转,
    强行打开反而有害。gate 在 crispasr_chunk_context_gate.h:82 —— vad_slicing 为真时
    should_use_chunk_context() 直接返回 false(issue #114:VAD 段之间本就是静音,没有
    边界信号可恢复,加上下文会把下一段语音拽进当前编码窗口,parakeet-tdt-0.6b-ja 上实测
    汉字塌成平假名并整段丢掉短片段);同一张 kBlocked 表(同文件 :33-55 的注释)里
    【qwen3 也被明令禁止】overlap-save,理由是它自己的 transcribe() 在 ~30 s 边界附近
    还会再内部分块,外层再裹 ±chunk_overlap 的声学上下文会把单次输入推过那个边界,
    后果是"按词级时间戳裁剪时丢掉后续的块 —— qwen3 在 90 s 素材上第 1 块就中途截断"
    (crispasr_chunk_context_gate.h:36-39)。注意别记串:同一段注释里"每个接缝都把文本
    重复解一遍、并把贪心解码推进复读循环"是 moss-transcribe 与 canary-qwen 的失效形态
    (#218),不是 qwen3 的。
    由此对她旧脚本 crispasr-Qwen.py 的结论:--vad 与 kBlocked 两道闸门【都已经挡着】
    overlap-save(她的命令里没有 --chunk-overlap 也没有 --lcs-dedup),所以旧脚本这条
    路不需要为这个坑改任何参数;把 VAD 阈值调高或干脆去掉 --vad 反而是倒退 —— 去掉
    --vad 就同时丢了 crispasr_rechunk_slices() 的按能量极小点切分。
    真正在治边界的是 --chunk-seconds:VAD 段超长时 crispasr_rechunk_slices()
    (src/crispasr_vad.cpp:403-449)按【能量极小点】切,不在词中间下刀。
  · 加 --strict-pipeline(#311):-vm / --punc-model 这类辅助环节加载失败时直接非零退出,
    不再"warning 一声然后无标点静默跑完整棵树"。VAD 判全文件无语音不算失败,但这类文件
    一律记进 <输出目录>\no_speech.txt(时间\t原因\t相对路径),吞了多少看那个文件。
  · 【没接】CTC 强制对齐器(canary-ctc-aligner,已下到模型目录但本脚本不引用)。
    理由:它只产出词级时间戳,而这条管线带 --no-timestamps、只出 .txt,下游 asr-triage
    按段落/行切,不消费时间戳。要用得同时改 -ojf 输出格式和 triage 的解析口径,属联动
    改动,不在本副本范围内。要接就加 `-am <路径> --force-aligner`(qwen3 是 LLM 解码器,
    原生词级时间戳档位是 ts-word:-,得靠 -falign 才拿到词级)。
  · 单实例锁与旧脚本共用 <输出目录>\.lock —— 故意的:两个驱动各载一份权重会抢
    同一块 8 GB 显存,同一时刻只能跑一个。本脚本内部两趟引擎是【先后】跑(AED 批次进程
    结束后才起 qwen3 进程),不存在两份权重同时驻留。
  · 兜底让 qwen3 的 2.51 GB(q8_0)也变成"必需文件",所以启动前把两个引擎的模型、标点
    模型、判别模型逐个 os.path.isfile 校验,缺哪个报哪个(全部字面路径集中在文件顶部
    CONFIG · 外部资源 那一段)。

用法不变:python crisper-xhs-qwen-asr.py 前台跑 / --start 后台跑 / --stop 优雅停止。

以下驱动行为原样继承 crispasr-Qwen.py(对比最早那版逐文件调用 crispasr 的脚本):
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
  · 语种一律 `-l auto`(CRISPASR_LANGUAGE),两个引擎都是 auto,不写死 zh。按目录名给
    语种的整套路由【整块删除】了(不是清空)—— 换引擎现在只由前置判别的结论决定。
    前置判别器开着当筛子(CRISPASR_LID_BACKEND = "whisper" + LID_VERBOSE = True,
    后者是筛子的眼睛:判别行受 !no_prints 控制,关掉它筛子就瞎了)。逐条依据见
    CONFIG · 语种
  · 外部资源(可执行文件目录、两台引擎的权重、标点/判别/VAD 模型)的字面路径全部集中在
    文件顶部的 CONFIG · 外部资源 一段,换机器只改那一段;脚本其余地方只引用变量。
  · 没有显卡的机器(CPU-only)能直接拿这份脚本去跑,只改 CONFIG 里这几行:
    CRISPASR_BIN_DIR 指到 crispasr 的 cpu 版解压目录(本机现用目录里其实已经有
    ggml-cpu.dll,所以不换目录也能跑)、CRISPASR_GPU_BACKEND 改 "cpu",再把
    CRISPASR_THREADS 按【逻辑线程数的 0.75 倍】设。那种机器上该把 ENGINE_PRIMARY 换成
    qwen3(上游那张 CPU-only 表里 AED 只有 0.1x,是最慢的一行;qwen3 那行给的是 0.6B 版,
    1.7x)。本机 1.7B 的 CPU 数值 10-02 已经自己量到了:纯 CPU、t=12、89.84 s 片段
    RTF 0.59 —— 账写在 CONFIG · CrispASR 的"没有显卡的机器能不能跑"那一节。
  · 待转写文件先硬链接到 <音频根>\t 下的 ASCII 名再喂给 crispasr.exe:它是 ANSI
    程序,argv 经系统 ACP(cp936) 转换,文件名里 GBK 表示不了的字符(❤ ⚡ 及不可见
    的变体选择符 U+FE0F 等)会变成 '?',路径随之失效
  · 后台启动 / 优雅停止收进本文件(--start / --stop),不再依赖数据盘上的两个 .bat,
    POSIX 下走 start_new_session,迁移时只改 CONFIG 里的路径
  · 只依赖标准库 + send2trash
"""

import ctypes
import itertools
import json
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
# 字面路径。启动时 main_loop 会逐个 os.path.isfile 校验,缺哪个就报哪个。
#
# 取值口径 = 环境变量优先,没给就落在本包目录下(相对名按本文件所在目录解析),所以整包
# 放哪儿都能跑。CRISPASR_BIN_DIR / CRISPASR_MODEL_DIR 两个 env 名和 assets.json 里 roots
# 的 env 字段是同一个,fetch_assets.py 把组件落到哪儿,这里就从哪儿读;数据面两个目录用
# ASR_TEXT_DIR / ASR_AUDIO_DIR(清单不往里放文件,那是输入和记账的地方)。
#
# 迁到没有显卡的机器(CPU-only)只需动两行:CRISPASR_BIN_DIR 指到
# crispasr-windows-x86_64-cpu(-legacy).zip 的解压目录、下面的 CRISPASR_GPU_BACKEND
# 改成 "cpu";模型文件一个都不用重下(依据见 CONFIG · CrispASR 里 CPU-only 那一节)。

# ---------- 可执行文件 ----------
# 两个引擎共用同一个 crispasr.exe(qwen3 与 firered-asr 都在同一个二进制里)。
CRISPASR_BIN_DIR = resolve_dir("CRISPASR_BIN_DIR", "crispasr")
CRISPASR_EXE     = os.path.join(CRISPASR_BIN_DIR, "crispasr.exe")

# ---------- 模型目录与全部权重 ----------
MODEL_DIR = resolve_dir("CRISPASR_MODEL_DIR", "model")

# 主力:FireRedASR2-AED q4_k(962,807,328 B)
CRISPASR_MODEL_AED = os.path.join(MODEL_DIR, "firered-asr2-aed-q4_k.gguf")
# 兜底:qwen3-ASR-1.7B。按她的决定用 q8_0(2,506,723,200 B);同目录另有 q4_k
# (1,490,915,200 B),显存不够就换成它。
CRISPASR_MODEL_QWEN3 = os.path.join(MODEL_DIR, "qwen3-asr-1.7b-q8_0.gguf")
# 书面标点后处理,【只给 AED 用】(qwen3 自己出标点,理由见 CONFIG · 引擎)。
CRISPASR_PUNC_MODEL = os.path.join(MODEL_DIR, "fireredpunc-q8_0.gguf")
# whisper-tiny 前置语种判别器(77,691,713 B)。当筛子用,见 CONFIG · 语种。
CRISPASR_LID_MODEL = os.path.join(MODEL_DIR, "ggml-tiny.bin")
# VAD 权重 = FireRedVAD(小红书自家,voice 含唱歌)。firered-vad.gguf,2,357,952 B。
# 选它是因为 silero v6.2.0 对唱歌素材判 0 段 → 整条静默丢弃(无 .txt、rc 仍为 0),
# v5.1.2 只捡回 59 字,firered 出 97 字真歌词。详细数据与代价见 CONFIG · VAD。
# 另两份脚本(crispasr-Qwen.py / crispasr-Qwen-cpu.py)用 silero v6.2.0。
CRISPASR_VAD_MODEL = os.path.join(MODEL_DIR, "firered-vad.gguf")
# 语种闸门用的 VAD = silero v6.2.0(885,098 B),就是 crispasr 不加 -vm 时自己去
# huggingface.co/ggml-org/whisper-vad 下的那一份。【只有闸门读它】,转写本身仍用上面
# 的 FireRedVAD(唱歌那条形不变)。为什么闸门反过来要用 silero:这个模型是"默认"的那一个,
# 拿它判语种和拿 crispasr 自己那套前置判别同源,而真正的分段交给 ONNX 链里的 FireRedVAD,
# 两条腿各用各的、互不影响 —— 详见 CONFIG · 语种闸门。
GATE_VAD_MODEL = os.path.join(MODEL_DIR, "ggml-silero-v6.2.0.bin")

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

# ==================== CONFIG · CrispASR ====================

# ---------- 引擎:主力 AED + 兜底 qwen3 ----------
#
# 分工只有一个判据 —— 语种。FireRedASR2-AED 的中文信息保留度比 qwen3 稳(它没有 LLM
# 解码器,结构上不可能把一段话概括成一句),但它内置的语种判别只覆盖 中文(+约20种汉语
# 方言)/ 英语 / 粤语,范围外【不是差一点,是拿汉字编造】,而且 -l 改不了它。所以:
#
#   主引擎  aed    —— 默认所有文件都走它
#   兜底    qwen3  —— 只接前置筛子(CRISPASR_LID_BACKEND = whisper)判到 AED 范围外的文件
#
# 换引擎这件事以前是"她自己切回 crispasr-Qwen.py 手工跑",现在由本脚本在同一批里自动
# 完成:判到范围外的文件【不落盘】,攒下来立刻用 qwen3 重跑一遍(见 process_batch)。
# 兜底那趟仍然是 `-l auto` + whisper 前置判别 —— 不是图省事,是因为 qwen3 的 -l 是真生效
# 的(被拼成一条 assistant 前缀),而"把驱动判到的码再传一次"会把筛子的错误固化;
# 让 qwen3 自己判一遍, lidbench 里 120 s 日语歌最好那一档(547 字可用假名)就是这么出来的
# (v3_results.tsv 第 25 行)。
ENGINES = {
    "aed": {
        "label":   "FireRedASR2-AED",
        "backend": "firered-asr",
        "model":   CRISPASR_MODEL_AED,
        # 标点后处理:AED 的解码器不出标点,靠 CLI 后处理层加(apply_punc_model,
        # crispasr_run.cpp:322)。留空会让 crispasr 自动取 "auto" 并【联网下载】
        # FireRedPunc(crispasr_run.cpp:3940),本机拉不动就是每批白等一次超时,
        # 所以必须给本地路径。
        "punc":    CRISPASR_PUNC_MODEL,
        # AED 声明 CAP_BEAM_SEARCH(qwen3 声明的是 CAP_TEMPERATURE),-bs 只对它有意义。
        "beam":    True,
    },
    "qwen3": {
        "label":   "Qwen3-ASR-1.7B",
        "backend": "qwen3",
        "model":   CRISPASR_MODEL_QWEN3,
        # 【必须留空】:qwen3 有能力位 CAP_PUNCTUATION_TOGGLE(crispasr_backend_qwen3.cpp:56),
        # 本来就自己出标点,再叠一层就是二次打标 —— 源码注释写得很直白
        # "already emit punctuation should not get a second pass"
        # (crispasr_punctuation_policy.h:10-12)。
        "punc":    "",
        "beam":    False,
    },
}

# 前置判别判到 AED 范围外时是否真的换引擎。关掉它 = 只在日志里报一句、内容仍按 AED 落盘。
USE_LID_FALLBACK = True

ENGINE_PRIMARY  = "aed"
ENGINE_FALLBACK = "qwen3"

# AED 内置语种判别的覆盖范围。超出这个范围它不是"差一点",是凭空编造(依据见下面
# 「换引擎备查」那段)。留着当边界提示,不参与任何判断。
AED_COVERS = ("zh", "en", "yue", "+约20种汉语方言")

# ---------- 换引擎备查(兜底那趟的全部依据,别删这段)----------
#
# qwen3 那条路上 `-l` 是【真】生效的:被拼成一条 assistant 前缀
# "language <英文名><asr_text>"(crispasr_backend_qwen3.cpp:180-184),属软约束。
# 传错码的代价本机实测过:同一首 120 s 日语歌切片,强制 `-l zh` 是全场最差一档 ——
# 1,147 字逐字母 romaji + 复读「想得遠」48 次、整段作废(out/v3_zh_t07_ayts.txt);
# 让前置 LID 判到 ja 才有 547 字可用假名(out/v3_lid_t07_ayts.txt)。基准树
# 是作者的本地 lid 基准目录(不在本包里;v3_results.tsv 第 9 行=zh、第 25 行=lid,逐字可数)。
# AED 这边机制相反:词表 8,667 个 token 里假名为 0(本机那份 gguf 直接数出来的 ——
# firered.odim=8667、tokenizer.ggml.tokens 长度 8667、含ぁ-ン 的 0 个、含汉字的
# 7,566 个),所以给它非中英语种不是"换个码"的事,是让它用汉字编一段话
# (crispasr_backend_firered_asr.cpp:49-70,issue #199:纯日语音频给 `-l ja` 照样
# 输出中文幻觉)。这就是 AED 只适合中/英/粤、别的语种必须换引擎的全部理由。
#
# 筛子为什么"看得见"的两个能力位事实(本轮新核,决定整个兜底机制成不成立):
# 前置判别器是否真的跑,取决于后端有没有声明 CAP_LANGUAGE_DETECT ——
# crispasr_run.cpp:948 有 `has_native_lid = (backend.capabilities() & CAP_LANGUAGE_DETECT)`,
# 而 :992 那段外部判别的条件是 `want_auto_lang && !has_native_lid && !lid_disabled`。
#   · firered-asr 的 capabilities()(crispasr_backend_firered_asr.cpp:21-36)返回
#     CAP_UNBOUNDED_INPUT | CAP_TIMESTAMPS_CTC | CAP_AUTO_DOWNLOAD | CAP_BEAM_SEARCH |
#     CAP_TOKEN_CONFIDENCE | CAP_FLASH_ATTN | CAP_DIARIZE —— 【没有】 CAP_LANGUAGE_DETECT,
#     它那句"auto-detects the spoken language"说的是模型内部行为,不占能力位。
#   · qwen3 同样没有,而且是明写的(crispasr_backend_qwen3.cpp:48-49
#     "CAP_LANGUAGE_DETECT intentionally NOT declared")。
# 所以 whisper 前置判别在【两个】引擎上都会真的跑:在 AED 上判到的码只用于挑文件(它改不了
# 输出),在 qwen3 上判到的码会真的进 -l。这正好是"筛子 + 兜底"各取所需的那两种用法。

CRISPASR_GPU_BACKEND = "cuda"

# ---------- 为什么 AED 这条路"慢得离谱还不吃显卡"(本机实测 + 源码)----------
#
# 现象(<输出目录>\log\crispasr_20261002_011648.log,2026-10-02 06:12 采样):
#   nvidia-smi dmon 连采 18 秒 sm/mem 全 0%,显存占着 1,230 MiB,整卡 2-6 W、
#   210 MHz(P8 省电档);同期 crispasr.exe 每 4 秒墙钟涨 27 秒 CPU 时间 = 稳吃
#   6-7 个核。批一 28 个文件跑 245.4 min,批二前 5 个文件(音频合计 54.3 min)
#   跑 50.5 min → RTF ≈ 0.93,基本是"1 秒音频 1 秒墙钟"。旧脚本的 qwen3 是
#   RTF ≈ 0.19,所以换引擎=每文件慢约 5 倍,这是买"不摘要"的价。
#
# 原因不在配置,在这个后端的实现:CrispASR 把 AED 的【解码器权重恒定放在 CPU】
# (src/firered_asr.cpp:441-444 原话 "Decoder weights ALWAYS go to CPU: the AED decode
# loop uses native Q4_K SIMD vecmats on the CPU backend (60ms/step; per-token GPU
# launches were 20ms each)" —— 注意这两个数的单位不同:60 ms 是【一整步】在 CPU 上算完,
# 20 ms 是【一次 kernel launch】;解码一步要过 16 层、每层好几个矩阵乘,将 launch 次数
# 直接乘爆,所以显存里放着权重反而更慢。源码注释没把这笔账写全,以上是按结构推的,
# 想证实只能自己测),只有 enc.* 张量 split-load 到显卡(firered_asr.cpp:447-451 注释、
# :453-465 代码 load_weights_split,官方在 Metal 2.3x / Vulkan-MoltenVK 2.1x / CUDA P100
# 2.2x 上验过转写逐字一致)。所以显存里有货、卡却整天空闲:每个切片(出厂上限 20 s,见下面
# CONFIG)编码器闪一下零点几秒,接着 CPU 解码几十秒,1 秒粒度采样就归零了。上游 PERFORMANCE.md:653
# 自己列的数字是 AED RTx 0.6x(全表最慢),第 712 行原话"FireRed decoder still runs on
# CPU even with GPU" —— 这句是对的。但同一份文档 337 行把解码器自注意力列为 P0、说它"没有
# KV cache、O(T²) 重算",这条与出货代码不符:每层每个 beam 都带一份 sa_k/sa_v 增量缓存
# (firered_asr.cpp:2247-2258 预留容量、:2340-2343 贪心路与 :2563-2568 beam 路每步只 append
# 当前这一步的 K/V);解码器还留了个逐 token 计时开关 CRISPASR_FIRERED_BENCH=1,打的是
# "decode step N/150 (ms elapsed)",想复核到底是"每步重算整段"还是"每步只加一个 token",
# 跑一条长音频看这一列随 N 怎么走就行。按代码读出来的结论是后者:单步成本≈已出 token 数
# (线性),整句的 attention 打分本身才是 O(T²) 那一项 —— 逐行依据见 README.md §5.1 ①。
#
# 对比:qwen3 是【反过来】的实现 —— 真 KV cache(src/qwen3_asr.cpp:249-261 字段、
# :2023-2102 qwen3_asr_kv_init,默认 F16,按 28 层 × head_dim 128 × n_kv 8 × max_ctx
# 4096 算 ≈ 224 MiB,运行时 verbosity>=1 会自己打印 "qwen3_asr: kv cache %d MiB"),
# 后端选择是 :1535 `params.use_gpu ? crispasr_init_gpu_backend() : core_cpu_backend::init()`
# —— 权重整份进显卡,所以它 RTF 0.19、卡在忙、显存也在涨。同一个 crispasr.exe,两个后端
# 把力气花在不同地方。顺带两条对她有用:qwen3 的 KV 可以用 CRISPASR_KV_ON_CPU=1
# (src/core/attention.h:151)落回内存换显存;而 qwen3 本身就能纯 CPU 跑(见下面 CPU-only
# 那张表,同一台机器上它比 AED 快一个数量级 —— 表里那一行是 0.6B,本机 1.7B 的数字
# 10-02 已实测:RTF 0.59,t=12,见下面"改两个常量"那一段)。
#
# 能动的三个档位(三条现在都有校机纯 CPU 实测,见下面 CONFIG 里各自的说明):
#   1) CRISPASR_THREADS:8845HS 是 8 大核 16 线程,现在 -t 6 只用 6 个,还有富余。
#      瓶颈是 CPU 的 mul_mat,调大最直接。CPU 满载会推高温度和风扇,自己权衡。
#   2) VAD_MAX_SEGMENT_SEC(--chunk-seconds):10-08 已实测,定这条的不是速度是丢字 ——
#      AED 每片解码上限硬顶 150 token,30 s 的连续中文会撞顶并静默丢尾,20 s 不撞。
#      解码器【有】KV cache(逐 token 流式读权重),所以成本≈token 数而非平方;
#      实测 30→20 的速度代价只有 1 s 量级。AED 绝对不能设 0(整趟 >50 s 在 CUDA 上挂死,issue #125)。
#   3) BEAM_SIZE(-bs):留空 = 默认 beam=3。10-08 实测 beam=3 比贪心慢 35%(12 线程)~49%(18 线程),
#      加线程救不回来;"beam=3 更准"这一条未实测,只算先验。
#
# ---------- 没有显卡的机器能不能跑(CPU-only)----------
#
# 能,而且这条路本来就是"CPU 为主"的 —— 解码器一直在 CPU 上(见上一节),显卡只承担
# 编码器那 2.1-2.3 倍。改两个常量就够,不需要重装:
#   CRISPASR_GPU_BACKEND = "cpu"   (--gpu-backend cpu;src/core/gpu_backend_pref.h:96-108
#                                    有专门的 cpu 短路分支,直接 core_cpu_backend::init()
#                                    并打印 "using the CPU backend, no GPU device
#                                    initialised";这分支是 T18 修的,之前 "cpu" 会
#                                    fall THROUGH 到显卡)
#   CRISPASR_THREADS     = 逻辑线程数的 0.75 倍  (2026-10-02 本机 CPU 实测扫出来的,
#                                    不是推算:同一份 89.84 s 片段、只改 -t,qwen3 +
#                                    --gpu-backend cpu 量到 4→0.88、8→0.65、12→0.59、
#                                    16→0.59、24→0.61、32→0.61 RTF。本机 8 大核 16 线程,
#                                    所以膝点在 12 = 1.5x 物理核。超线程对 qwen3 的逐
#                                    token GEMV 是有用的(8→12 快 9%),超过 1.5x 物理核
#                                    才开始白堆;别照"只给物理核"的老直觉压到 8)
# 本机这套 CUDA 目录里已经有 ggml-cpu.dll(962 KB)和 ggml-base.dll,所以【不用另外下载】
# 就能在 CUDA 的 exe 上试 --gpu-backend cpu;纯 CPU 包也已经下到
# 生产机的 <模型目录>\crispasr-windows-x86_64-cpu(zip 8,705,179 B,sha256 630ffec1…3f0b,
# 上面那组扫点就是用它跑的)。CPU 包里【没有】ggml-*.dll(静态链进 exe),但必须和
# openblas.dll 同目录。
# 官方也单独发纯 CPU 包:v0.8.37 与 v0.8.40 的资产里都有 crispasr-windows-x86_64-cpu.zip
# (8.3 / 8.5 MB)和 crispasr-windows-x86_64-cpu-legacy.zip(7.8 / 7.9 MB,给没有新指令集
# 的老 CPU);另有 vulkan.zip(约 36 MB)可在没 CUDA 的机器上走显卡。
#
# 速度按上游的账(PERFORMANCE.md:693-707,"CPU-only VPS — 2026-04-24",4 线程 AVX2、
# 7.6 GB 内存、无显卡),表里相关的是两行:
#   FireRed ASR2 AED  0.1x(CPU)/ 123 s  →  0.6x(T4)   —— 全表最慢的一行
#   Qwen3 ASR   0.6B  1.7x(CPU)/   6.5 s  →  4.7x(T4)
#
# 【这一行的型号是 0.6B,不是本机在用的 1.7B】,倍数不能套。1.7B 用真数:
# 本机 8 物理核/16 逻辑、crispasr-windows-x86_64-cpu v0.8.37、q4_k、silero v6.2.0 +
# whisper 前置判别、无 GPU 参与:
#     qwen3-1.7B   RTF 0.59(89.84 s 中文,-t 12;30 s 素材 26.4 s)
#     AED-q4_k     同段慢 1.6-1.9 倍(53.9 s / 43.2 s 两次)
# 所以 CPU 机器优先 qwen3,AED 那 0.1x 明确是【4 线程 VPS 的口径】,在本机纯 CPU 上 AED
# 反而是更慢的那个。文本两份几乎同一句话,这个样本上分不出好坏(n=1、30-90 s,C 级),
# 所以 CPU 侧没有"必须换 AED 保中文"的理由。
# 注意 0.59 是【量级】不是承诺:服务器 24 vCPU 单核更弱(2.40 GHz 基频 + 共享切片)、线程
# 多一半,两边大致抵消。要在服务器上定 ENGINE_PRIMARY,拿同一批文件 --gpu-backend cpu 再跑
# 一次就够,不需要重做线程扫点(见上面 -t 那节)。
#
# AED 那 6.5x(0.1x → 0.6x)里属于解码器的部分接近 0 —— 解码器两边都在 CPU(见上一节
# "解码器权重恒定放在 CPU"),大头是【编码器 offload】,再叠加那台 VPS 只有 4 线程。
# 本机 8 核 16 线程 + 4060 实测 AED RTF 0.93,而这份 RTF 已经是"解码全在 CPU"跑出来的
# —— 所以核数够多的纯 CPU 服务器,完全可能比 T4 上的 AED 快。
#
# 纯 CPU 机器该怎么配(不是"照搬本地这套"):
#   ENGINE_PRIMARY = "qwen3"、ENGINE_FALLBACK = "aed" 或直接只留 qwen3。依据是实测数值,不是
#   机制故事:这台笔记本纯 CPU 上 AED 比 qwen3 慢 1.6-1.9 倍(上面那组,n=1、C 级);10-08 在
#   24 vCPU 的机器上同机复测,差距缩到同一量级 —— AED 贪心整链 wall RTF 0.398(80.2 s 素材),
#   qwen3 整链 0.321(63.7 s 素材),素材不同,只能看量级不能当倍数。
#   机制那一侧要更正一句:PERFORMANCE.md:337 说 AED 解码器"没有 KV cache",这条与出货代码
#   不符(见 README.md §5.1 ①),所以"因为 AED 没做 KV、每步重算整段注意力才慢"这个解释已经
#   作废。两边都是增量 K/V;能确定的是 AED 的解码器权重恒定在 CPU(firered_asr.cpp:442),
#   每出一个 token 都要在 CPU 上过一遍 matvec。这件事上游自己逐节点量过:每步 90 多个小
#   matvec(8 个投影 × 16 层),卡的是 dispatch 而不是注意力复杂度,并为此加了默认开的常驻
#   图缓存开关 CRISPASR_FIRERED_MATVEC_CACHE(HISTORY.md:3449-3458、firered_asr.cpp:385-391)。
#   读码就能复算,不需要跑 GPU;三条依据的逐行出处在本目录 README.md §5.1 ①。
#   "qwen3 吃的显存比 AED 多近 3 倍,所以搬到 CPU 上一定更慢"是倒因为果:占用与速度是
#   两回事。它占得多是因为 LLM 那 28 层的 K/V 全存下来复用;AED 同样存 K/V,只是 16 层 ×
#   每片上限 150 token,规模小一个量级。搬到纯 CPU 机器上这笔账变成内存:
#   q8_0 权重 2.51 GB + KV 约 0.22 GB,比 AED 的 0.96 GB 多约 1.8 GB —— 服务器内存
#   通常不是瓶颈,别用它来反推速度。
#
# 纯 CPU 机器真正的优势不在单文件速度,在【核多可以开多个实例】:上游那张表是单进程 4 线程
# 的口径。多实例要按目录分片,并把 LOCK_FILE(<输出目录>\.lock)、PCM_INPUT、OUT_DIR、
# STAGE_DIR 各配一份,否则两个实例会互抢同一批文件与同一个锁。
#
# 收益【不是线性】,这条 2026-10-02 在本机(8 物理核/16 逻辑)量过了,同样两段 90 s:
#     1 进程 × -t 12   137.7 s → 1.31x 实时
#     2 进程 × 各 -t 6 103.2 s → 1.74x 实时(+33%)
#     2 进程 × 各 -t 12 105.1 s → 1.71x(超订无额外收益)
#     4 进程 × 各 -t 4 131.9 s → 1.82x(比 2 进程只多 3%,在 ±13% 的噪声里)
# 也就是"总线程数不变,拆成两个进程快三成",但到 1.8x 实时就见底 —— 撞的是内存带宽不是
# 核数(逐 token 流式读权重,q4_k 每步约搬 0.9 GB)。内存按每实例 2-3 GB 预留,大内存的
# 意义只是给多实例腾位置。
#
# 别的实事实:S: 是 USB 外接盘,服务器上要么把音频挪进本地盘,要么接受顺序读把 CPU 饿着;
# 脚本里的路径全在 CONFIG 顶部,换机就是改那几行;硬链接暂存(ASCII 名)在本地 NTFS 上仍然
# 需要 —— 它治的是 ACP/cp936 把非 GBK 文件名变成 '?' 的问题,与显卡无关;RDP 断开不等于注销,
# 纯 CPU 进程不受会话锁影响。
# 想验证纯 CPU 的速度不必换机器:把 CRISPASR_GPU_BACKEND 改成 "cpu" 跑一小批对比同一批文件
# 的墙钟。但那会和正在跑的实例抢 CPU,生产停了再测。
CRISPASR_THREADS     = 6

# ---------- 语种(-l)与语种判别(--lid-backend / --lid-model)----------
#
# 结论先说:默认 `-l auto`,不写死 zh。原因不是"判别比 zh 准",而是这条路上的 `-l`
# 【根本不生效】:FireRedASR2-AED 训练时就带 LID 头,解码器里没有语言开关,适配器源码
# 原话 "It has no token for any other language and offers no per-request language
# override, so `-l <lang>` cannot steer it"(crispasr_backend_firered_asr.cpp:49-57)。
# 所以 zh / auto 在 AED 上输出同一份东西,写死 zh 唯一的作用是:① 让日志里看不到任何
# 语种线索,② 一旦哪天换了引擎(qwen3 的 -l 是真生效的)就立刻变成硬约束。留 "auto"
# = "我不替模型决定语种"。
# 覆盖范围只有 中文(+约20方言) / 英语 / 粤语。超出这个范围不是"差一点",是拿汉字
# 编造(issue #199),这一层任何 CLI 参数都救不了 —— 非中英语种必须换引擎,换引擎就
# 用 crispasr-Qwen.py,本副本不做这件事。
#
# 下面 (1)-(8) 是【qwen3 上】的实测(基准树是作者的本地 lid 基准目录,不在本包里;45 s 组见
# labels.tsv / results.tsv,120 s 组见 v3_results.tsv 与 out/v3_*.txt,逐字文本都在,
# 可自己核),留着是因为它决定了"换引擎时 -l 该怎么填",不是 AED 的决策依据。
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
#                      > -l zh 全场最差:其中一首输出逐字母 romaji 加复读「想得遠」48 次
#                      (grep -c 可数),整段作废
#     ⇒ 一个文件只能贴一个标签,混语文件谁来判都保不住两种语言;LID 真正的价值只在
#       "整段都不是中文"的文件上,而 -l zh 在那种文件上是灾难(qwen3 上实测:一首
#       120 s 日语歌被逐字母 romaji 化 + 复读「想得遠」48 次、整段作废)。按目录名给
#       语种的那套路由【已按她的决定整块删除】,现在一律走 -l auto:目录名与语种一致
#       只是个假设,起错名字就拿错码,而错码在 qwen3 上是有实测代价的(见上)。
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
# (8) 判别结果在生产日志里看不见:`crispasr[lid]: detected 'xx' (p=…)` 与 CLI 那行
#     `crispasr: LID -> language = …` 都受 opts.verbose = !params.no_prints 控制,而
#     命令行带 --no-prints,所以是静默生效的。要看得见就开 LID_VERBOSE(见下面
#     "LID 前置判别器"一节),它去掉 --no-prints 并按批次顺序把码对回文件名。
#
# 别指望 --lid-backend probe 能让 qwen3 自己判:--help 写的是"问主模型自己",但
# qwen3 没有 self-probe,实测直接报 "this backend has no self-probe — falling back
# to whisper-tiny LID",然后没给 --lid-model 就去联网下 ggml-tiny.bin,卡满超过
# 2 分钟后 "LID failed and no -l was set — defaulting to 'en'" —— 兜底这个 en 才是
# 把音频推进翻译模式的元凶。真正"让模型自己判"只有 --lid-backend off,而它同样翻。
#
# 常用 ISO 639-1:zh 中文 / en 英语 / ja 日语 / ko 韩语 / yue 粤语 / es / fr / de /
# it / pt / ru / ar / hi / th / vi。完整 99 种见 crispasr.exe --help。
# 改成具体码 = 替模型决定语种,只有换回 qwen3 时这个改动才有实际意义(见上面结论先说)。
CRISPASR_LANGUAGE = "auto"

# ---------- LID 前置判别器(当筛子用,仅 CRISPASR_LANGUAGE = "auto" 时才用到)----------
#
# LID = language identification,语种判别:先只听一小段音频判断这是哪种语言,
# 再把判到的语种码喂给转写模型。
#
# 取 off | whisper | silero | ecapa | firered | probe。
#
# 现在取 whisper,角色是【筛子】:判到的码不改 AED 的输出(AED 根本没有 per-request
# 语言开关,crispasr_backend_firered_asr.cpp:49-57 原话 "offers no per-request language
# override, so `-l <lang>` cannot steer it"),只用来决定"这个文件该不该交给 qwen3 兜底"。
# 判到 AED 范围外 → 那份 AED 文本不落盘、源文件留在原位,整批改用 qwen3 重跑
# (实现见 _lid_map / transcribe_batch / process_batch)。
#
# 代价账(都不大):
#   · 每个文件多跑一次 whisper-tiny。AED 的 962 MB 权重此时已经驻留,是额外分配;
#     跑完 crispasr 会调 crispasr_lid_free_cache() 释放(crispasr_run.cpp:1021 注释,
#     那是 #35 显存 OOM 的修补)—— 也就是说这个缓存在批内【每个文件都要重建一次】,
#     不是我原先写的"每批一次"。耗时按 lidbench 实测:45 s 组 +1.8 s/文件,120 s 组
#     不复现(逐文件差 −0.5 到 −1.7 s),落在抖动里(见上面第 (5) 条)。
#   · 判据本身别当尺子用:只看前 15 秒【原始音频】(第 (6) 条:片头音乐会把语种定错)、
#     whisper 路径没有置信度闸门(第 (7) 条:实测 p=0.395 也照样往下走)。
#     lidbench 里 whisper-tiny 的召回是 中文 5/6、日语 2/4 —— 所以【假阴性一定会漏】:
#     判成 zh 的日语文件仍会被 AED 用汉字编造一段话并正常落盘,这层没有任何兜底。
#     要收这一层只能靠事后按内容特征筛,不在本脚本范围内。
#   · 假阳性的代价是白跑一遍 qwen3(0.19 RTF,比 AED 快 5 倍),几乎不要钱。
#
# 什么时候改回 off:不想要兜底、只想让 AED 安静跑完时。off 的含义【不是】"关掉语种
# 判别" —— AED 的判别烧在模型里,任何 CLI 参数都关不掉;off 关掉的只是上面这层筛子。
# 源码里的分支:crispasr_run.cpp:947 want_auto_lang = (detect_language || -l auto)、
# :948 has_native_lid、:949 lid_disabled = (lid_backend 是 off|none)、:975 self-probe
# (AED 与 qwen3 都没有)、:992 外部判别器、:1012 判别失败强制 en。
# 填了非 off 的值就必须让 CRISPASR_LID_MODEL(在 CONFIG 顶部的资源块里)指到本地文件,
# 留空或不存在的后果是每个文件联网下载 ggml-tiny.bin、超时约 30 s 后按
# crispasr_run.cpp:1017 强制 en,所以 main_loop 启动时把"auto + 前置模型缺失"判为致命错误。
CRISPASR_LID_BACKEND = "whisper"

# LID_VERBOSE = True 时去掉 --no-prints,把 crispasr 的标准错误里那行
# `crispasr: LID -> language = 'xx' (whisper, p=0.xxx)` 捞出来。那行受 !no_prints 控制
# (见上面第 (8) 条),所以【开着兜底就必须开着它】—— 关掉的后果不是"日志少几行",
# 而是筛子一行判别都抓不到、_lid_map 返回空、所有文件都按"无结论"走 AED 落盘,兜底静默
# 失效(main_loop 会为此打一条警告)。开着它同时把 USE_LID_FALLBACK=False,就变成
# "只报不改":判别结论进日志,内容仍按 AED 落盘。
# 代价:日志变成逐文件详细输出,一批 28 个文件多几百行。
LID_VERBOSE = True


def _lid_is_off() -> bool:
    return (CRISPASR_LID_BACKEND or "off").strip().lower() in ("off", "none", "")

# ---------- VAD 模型路径(-vm)----------
#
# -vm 同时接受关键字和路径(auto | silero | firered | marblenet | webrtc |
# whisper-vad)。给关键字 = 去 ~/.cache/crispasr 找、找不到就【联网下载】,
# 本机拉不动 = 每批白等一次超时后静默退回固定分块,所以这里一律填绝对路径。
# crispasr_vad_cli.cpp:31-41 按文件名里同时含 "firered" 和 "vad" 来识别引擎,
# 所以这份路径走 FireRedVAD 分支;silero 那份文件名走默认分支。
#
# 为什么是 FireRedVAD:唱歌素材会被 silero 整条吞掉。同一条 60 s 日推歌曲
# (あたらよ「パレード」BV1Fj411z7hM,45 s 起副歌)、同一份 CPU 构建、同一组参数,只换 -vm:
#     firered-vad.gguf   97 字,真歌词(--firered-vad-debug:5998 帧 / mean_prob 0.79
#                        / speech(>0.3) 4955 帧 ⇒ 覆盖约 82.6%)
#     silero-v5.1.2      59 字,5 段,语音合计 2.60 s ⇒ 覆盖 4.3%
#     silero-v6.2.0      【0 段 / 不落 .txt / rc=0 / "no speech detected"】= 静默丢弃
#     不加 --vad(固定分块) 101 字,内容最全 —— 但见下面"为什么还留着 --vad"
# silero 判 0 段时 rc 仍是 0,驱动看不出失败,文件就这么无声消失,所以这类文件现在
# 一律记账到 <输出目录>\no_speech.txt(见 record_no_speech)。待转库里带唱歌/BGM 的占比不低
# (光 p/b.3537109134608937 一个上传者就有 30 多条歌曲)。
# 这是小红书的自家模型:GitHub org FireRedTeam = 小红书 Super Intelligence 基础算法实验室,
# FireRedASR2S arXiv:2603.10420 §4.1 明确 voice = speech ∪ singing(UGC 场景把唱歌算语音),
# crispasr 只把它当二值 VAD 用(src/firered_vad.h 的片段结构只有 start_sec/end_sec)。
#
# 中文【口语】三家打平,选 firered 不换来内容:同一份 89.8 s 素材 firered 545 字、
# silero-v6 542 字、v5 545 字,逐片段比只差同音字与断句。代价是 firered 多花约 31% 墙钟
# (72.5/74.3 对 56.0/56.1,两次同向,超出 ±13% 抖动),机制只到 C 级(疑似切分形状不同
# 导致解码次数不同)。上游 PERFORMANCE.md:1170-1176 那张"firered 覆盖少 8 个百分点"的表
# 在本机中文素材上没复现,别再引。
#
# 别拿 -vt 去救 silero:唱歌素材上 -vt 0.50/0.25/0.02 三档全 0 段;口语正对照里 -vt 0.95
# 反而从 7 段变 13 段 —— 阈值升高段数变多,和"越高越严"不一致,语义与文档不符,没搞清前别调。
#
# 版本说明:上游不带 -vm 的默认是 silero v6.2.0(examples/cli/crispasr_vad_cli.cpp:19-20),
# crispasr-Qwen.py 与 crispasr-Qwen-cpu.py 就用那份;两份 silero 都是 885,098 B 但 md5 不同
# (c8f28919… / ee99234b…)。whisper-vad-asmr-q4_k.gguf(日语 ASMR 专用判别式 VAD)本机没有。
# --vad 要留:不加它走固定 30 s 分块,qwen3 在块边界上有已证的截断形态(overlap-save 裁掉
# 词级时间戳,见 CONFIG · 引擎里 kBlocked 那条);--vad 还接 crispasr_rechunk_slices() 的
# 按能量谷底再切(src/crispasr_vad.cpp:403-449)。
# 也接受特殊值 "webrtc":纯算法 VAD(GMM),无需任何权重文件。
# 路径定义在文件顶部 CONFIG · 外部资源 那一段。两个引擎共用同一个 VAD。

# ---------- VAD 细节阈值 ----------
#
# 全部留空 = 用 crispasr 自带默认(-vt 0.50 / -vspd 250 ms / -vsd 100 ms /
# -vp 30 ms)。这几档没有本机实测依据,不擅自改;要动就填数字,空串表示不传参。
VAD_MIN_SILENCE_MS = ""     # -vsd:判为段间切断的最小静音长度,调大 = 段更少更长
VAD_MAX_SPEECH_SEC = ""     # -vmsd:超过这个长度的语音段自动再切(默认 FLT_MAX)

# ---------- 分块上限(--chunk-seconds)----------
#
# 两条引擎都靠它兜边界,但原因不同:
#   aed    必须给。FireRedASR2-AED 的编码器是相对位置编码 pe_maxlen=5000(≈200 s)
#          + O(T²) 自注意力,issue #125 实测 >50 s 单趟在 CUDA 上直接挂死
#          (crispasr_backend_firered_asr.cpp:22-36)。VAD 段超过 --chunk-seconds(出厂 20 s)时由
#          crispasr_rechunk_slices() 在能量极小点处切开。真正决定这个值能不能放大的是下面
#          VAD_MAX_SEGMENT_SEC 那条:每个切片的解码预算硬顶 150 token。
#   qwen3  同样要给:LLM 解码器的 KV cache 随音频时长线性增长,不分块会顶显存。
# 设 0 = 不分块(整文件一趟),AED 这条路【不要】这么设。
#
# 【为什么是 20 不是 30】10-08 校机纯 CPU 实测:crispasr 的 AED 每个切片解码上限写死在
# src/firered_asr.cpp:2075 `max_len = min(T_sub, 150)`,没有任何 CLI 参数能改
# (`-n/--max-new-tokens` 是 LLM 后端用的,AED 路径不读)。连续中文旁白的实际密度约
# 5.5 token/s(源码注释里"3-4 token/s"是英文的密度,套不到中文上),150 token ≈ 27 s:
#   80.2 s 样本 s1 —— 30 s 切 3 片,token 150/150/118,两片撞顶,丢了"遍地""定独自持枪
#                   突入在所有民众的镜头中""毙",去标点 445 字;20 s 切 5 片,
#                   token 109/100/109/108/10,零撞顶,463 字。
#   344.9 s 样本 s2 —— 30 s 切 12 片,**7 片撞顶**,去标点 1771 字;20 s 切 19 片,
#                   零撞顶,1850 字(找回 3 个整句)。
# 撞顶时 crispasr 不报错、日志只打 token 数、rc 仍为 0,所以 30 s 这一档是【静默丢内容】,
# 不是慢的问题。速度代价实测接近零:s1 wall 44.0→43.9 s,s2 184.4→181.4 s。
VAD_MAX_SEGMENT_SEC = 20

# ---------- 波束搜索(-bs)----------
#
# 留空 = 各后端自己的默认(AED 在 firered_asr.cpp 适配器里回落到 beam 3,
# crispasr_backend_firered_asr.cpp:47)。
#
# 【beam 到底换来什么】10-08 校机同一条 80 s 中文样本实测:beam=3 比 -bs 1 贪心慢
# 35%(12 线程)~49%(18 线程),而去标点后的正文**只差 1 个字**(463 vs 464,相似度 0.9989)。
# 也就是说在这条链上"beam 更准"是**先验、不是实测**——我没有参考答案,两个数都无从判对错,
# 能确定的只有代价。要固定在贪心就显式填 1(CLI 的 -bs 1 确实进贪心分支,
# firered_asr.cpp:2311);顺带能拿到 firered 只在 beam_size==1 那条路上接的 F1 复读熔断
# (:2274-2283,beam 路没有,硬音频上会空转到 max_len)。本机未测 qwen3 侧,默认不动。
BEAM_SIZE = ""

# 单次调用喂几个文件。模型只加载一次,显存占用不随此值增长(推理是顺序的)。
# 有了下面的批内轮询结算,调大它不再意味着中断时要重跑整批,只影响日志粒度。
BATCH_SIZE = 28

# 批内轮询间隔:每这么多秒去看一眼 crispasr 已经写出的 .txt,写完一个就立刻
# 落盘并清理源文件,而不是等整批跑完(一批可达 1~3 小时)才统一结算。
BATCH_POLL_SEC = 30

# ==================== CONFIG · FireRed ONNX 主链 + 语种闸门 ====================
#
# ---------- 主力那一路换成 ONNX 全链路 ----------
#
# PRIMARY_LEG 决定"小红书那一路"(ENGINE_PRIMARY = aed)由谁来实现:
#   "onnx"     —— 本包 onnx/ 下的 FireRedASR2S 全链路:FireRedVAD(ONNX) →
#                 FireRedASR2-AED(纯 onnxruntime, aed_ort.py) → FireRedPunc(ONNX),
#                 三个会话在本进程里常驻一次,逐文件跑。
#   "crispasr" —— 10-09 之前的形态:整批交给 crispasr.exe 的 firered-asr 后端。
#                 留着它有两个用处:权重/闸门任一缺席时还能有个跑得起来的样子;
#                 同机对拍时要拿它当参照。
#
# 为什么值得换(全部是本仓实测,不是推测):
#   · crispasr 的 AED 后端把【解码器权重恒定放在 CPU】(src/firered_asr.cpp:441-444),
#     本机 nvidia-smi dmon 连采 18 秒 sm/mem 全 0%、整卡 2-6 W —— 显卡整天空闲,
#     而它吃 6-7 个 CPU 核。(10-09 用带时刻的 2 秒采样复验过这条:只走 AED 的单件跑
#     120 个采样点里 util>5% 只有 6 次、中位 0%、显存恒 1,116 MiB —— 见 README.md §6.6 的 g5。)
#     ONNX 这条链把 f32 编码器交给 CUDA EP,是"真能在 N 卡上跑"的那条路
#     (10-09 在 4060 Laptop 上端到端实测:同批三件素材 provider=cpu 555.4 s → cuda 418.9 s,
#      驱动自己读的 get_providers() 里落了 CUDAExecutionProvider,见 README.md §6.6)。
#   · 10-08 校机纯 CPU 同机同件对拍:单件 29.2 s vs crispasr 贪心 29.4 s(打平),
#     批次上 ONNX 少 15%~38%。
#   · 丢字形态不同:crispasr 每个切片解码上限硬顶 150 token(src/firered_asr.cpp:2075,
#     无任何 CLI 参数可改),撞顶不报错、rc 仍为 0 = 【静默丢整句】;这条链的每段预算是
#     min(段时长 × 8 token/s, cache 上限),撞顶会把 truncated 标出来。VAD 段上限设 20 s
#     就是为绕开前者(见上面 VAD_MAX_SEGMENT_SEC 那段实测),换掉后端之后 20 这个数不再是
#     正确性前提,但仍留着 —— 它同时是"标点按段跑"的粒度。
#
# 换引擎的判据仍然【只有语种】,变的是"谁来判":以前判语种这件事发生在 crispasr 批内
# (whisper 前置判别 + 从 stderr 对号),现在发生在调用主链之前,由下面的 GATE 独立完成。
# 于是筛子那套 _lid_map 实时对号在主链为 onnx 时不再参与(它只服务 crispasr 批量调用),
# 兜底那一趟照常走 crispasr 自己的 -l auto(不把判到的码传下去,理由见 CONFIG · CrispASR
# 里"换引擎备查"那段)。
PRIMARY_LEG = (os.environ.get("ASR_PRIMARY_LEG") or "onnx").strip().lower()

# ---------- ONNX 链路的代码与权重 ----------
# 全部相对本包目录,和 crispasr 那两件一样支持环境变量覆盖;models 子树不进库,
# 由 onnx/fetch_assets.py 一键下载 + 建造(逐件 sha256 验收)。
ONNX_DIR        = resolve_dir("FIREDASR_ONNX_DIR", "onnx")
ONNX_MODELS_DIR = resolve_dir("FIREDASR_ONNX_MODELS_DIR", os.path.join("onnx", "models"))
# AED 目录留空 = 在 ONNX_MODELS_DIR 下按 sherpa-onnx-fire-red-asr2* 自动找(与链路同口径)。
# 空值是"自己找"的哨兵,不能过 resolve_dir —— 它会把 "" 拼成本包目录,链路就改成在包目录下
# 找 encoder.f32.onnx,起链当场 FileNotFoundError(10-09 校机沙箱实测)。
_ONNX_ASR_ENV = (os.environ.get("FIREDASR_ONNX_ASR_DIR") or "").strip()
ONNX_ASR_DIR = os.path.abspath(os.path.expandvars(_ONNX_ASR_ENV)) if _ONNX_ASR_ENV else ""
ONNX_VAD_DIR  = resolve_dir("FIREDASR_ONNX_VAD_DIR",
                            os.path.join(ONNX_MODELS_DIR, "fireredvad-onnx"))
ONNX_PUNC_DIR = resolve_dir("FIREDASR_ONNX_PUNC_DIR",
                            os.path.join(ONNX_MODELS_DIR, "fireredpunc-onnx"))
ONNX_PUNC_FILE = os.environ.get("FIREDASR_ONNX_PUNC_FILE", "")   # 空 = 先 punc.f32.onnx

# ---------- 执行提供者:只用 N 卡 ----------
# cpu = 纯 CPU;cuda = CUDA EP(只会挑中 NVIDIA 独显)。这条链【不列】DirectML 与
# OpenVINO —— 那两个能把 AMD 核显挑中,而要求是绝不用核显。
# 装了 onnxruntime-gpu 也不代表用上了显卡:CUDA/cuDNN 运行库不齐时 ORT 只打一行警告
# 就把会话整体退回 CPU,所以 main_loop 会拿 AedOnnx.enc_ep 断言,对不上直接不起批。
ONNX_PROVIDER = (os.environ.get("FIREDASR_ONNX_PROVIDER") or "cpu").strip().lower()
# 图形状:int8 = 官方量化图(对拍背书,但 CUDA EP 跑不了那些整型算子);
# mixed = f32 编码器 + int8 解码器(实测最快,也是 cuda 档唯一有意义的选择);
# f32   = 全逆量化(最占内存)。
ONNX_GRAPH = (os.environ.get("FIREDASR_ONNX_GRAPH") or "mixed").strip().lower()
ONNX_MODE  = "greedy"          # beam 实现了但未调通(慢 9 倍、输出退化),见 onnx/asr_chain.py
ONNX_CACHE_LEN = "auto"        # KV cache 预分配:按 8 token/s 估
# 解码器要不要也上卡。默认不给:f32 解码器是逐 token 自回归,每个 token 都要发一批
# kernel launch,而 crispasr 那边实测过这笔账(源码注释原话"per-token GPU launches
# were 20ms each" 对 CPU 一整步 60ms)。这里留一个开关是为了在本机测出结论而不是引用它
# —— 10-09 在 N 卡(4060 Laptop)上测完了,结论是【上卡更慢】:同一批三件素材(981.6 s 音频)
# 解码器留 CPU wall 418.9 s、也上卡 537.2 s(+28%,三件逐件都慢),显存峰值 4,256→4,788 MiB
# (8 GB 档放得下,所以挡人的是速度不是显存)。逐字与出处见 README.md §6.6。
ONNX_DEC_ON_GPU = False

# ---------- 线程 ----------
# 与 cpu 分支那个分发包同口径:ASR 吃逻辑核的一半,VAD 上限 8,标点 4。
# 换到显卡之后 ASR 的 CPU 线程只影响前端特征与解码器(mixed 档解码器在 CPU),别指望调它
# 能救编码器 —— 编码器在卡上时 CPU 线程数与它无关。
ONNX_ASR_THREADS  = int(os.environ.get("FIREDASR_ONNX_THREADS") or
                        max(2, (os.cpu_count() or 8) // 2))
ONNX_VAD_THREADS  = min(8, os.cpu_count() or 8)
ONNX_PUNC_THREADS = 4

# ---------- 语种闸门 ----------
#
# 两步,都是 crispasr 的短调用,固定 --gpu-backend cpu(独显留给主链,而且这两次调用
# 各只要一两秒,让 crispasr 去建 CUDA 上下文纯属浪费一份显存):
#   1) `--vad -vm <silero> --vad-export-raw <json> --strict-pipeline --require-vad`
#      拿真语音段。必须用 --vad-export-raw:不带 -raw 的那份导的是 kind="chunks"
#      (30 s 网格),不是语音段(10-09 校机实测)。
#   2) 按语音段拼出【前 15 秒语音】写成临时 wav,再 `-m <ggml-tiny.bin> -dl` 判语种。
#      为什么切 15 s:whisper 那条判别路径本来就只截原始音频前 15 s
#      (src/crispasr_lid.cpp:284 kLidMaxSamples = 16000*15),把"原始前 15 s"换成
#      "语音前 15 s"才是这一步的全部价值。
#   正对照(10-09 校机合成件 ctrl_music_intro.wav = 15 s 合成器乐 + 20 s 中文语音,
#   标准答案 zh;探针 b1009_scan/gate5.py):
#      原始前 15 s  → en 0.784(错)
#      语音前 15 s  → zh 0.966(对)
#   代价实测(校机 CPU、-t 2,与她的生产批次同机争抢;s1.wav 80.2 s):
#   VAD 导段 0.46 s + 判别 1.07 s ⇒ 约 1.5 s/文件(探针 b1009_scan/g6b.py)。
#   两步【不能合成一次调用】:同一条命令里 --vad-export-raw 与 -dl 同时给时,crispasr
#   0.46 s 就退出且【不出判别行】= 静默假成功,闸门会全批没结论。
#   三条边界:
#     · silero 判 0 段(唱歌素材的已知形态)⇒ 拼不出语音前缀 ⇒ 退回对原始文件直接 -dl,
#       仍然有结论;判到范围外才扣文件。
#     · 闸门【没有】结论(没抓到判别行 / crispasr 非零退出)⇒ 不扣文件,按主链跑,
#       日志里报一行。宁可让 AED 编造一段(那是已知的老风险),也不能因为闸门自己瞎了
#       就把整棵树拖去兜底引擎。
#     · 语种名单沿用 _AED_IN_RANGE,没有另立一套。ONNX 这条 AED 和 crispasr 那个是同一个
#       模型,覆盖范围一样,而且它【同样不吃语言标记】—— 判到的码只用来挑文件。
GATE_LID_SEC     = 15.0    # 判别用多长语音(与 crispasr 内部那个 15 s 上限同值)
GATE_MIN_SPEECH_SEC = 1.0   # 拼不出这么多秒语音就退回对原始文件直接 -dl
GATE_THREADS     = 2       # 闸门两次 crispasr 调用的 -t
GATE_TIMEOUT_SEC = 300     # 单步超时:闸门本该一两秒,超这个数就是环境出了问题
GATE_SELFCHECK_FILES = 3   # 起批前拿前几个待处理文件实测闸门,一次出结论就放行

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

# 进程级自增序号，见 stage_paths 的注释（嵌套暂存会撞名）。
_STAGE_SEQ = itertools.count()

_NO_SPEECH_RE = re.compile(r"no speech detected in '([^']*)'")

# CLI 的语种判别行:crispasr_run.cpp:1003 `crispasr: LID -> language = 'xx' (whisper, p=0.395)`
_LID_RE = re.compile(r"LID -> language = '([^']+)' \(([^)]*)\)")

# 走到了判别这一步、但没出码的行 —— 它同样【占一个槽位】,不记下来就会让它后面的文件
# 整体错位一位。措辞取自 crispasr_run.cpp:1013 与 :1020(都以 "crispasr: LID failed"
# 开头);检测器自己那些 "crispasr[lid]: … LID failed" 的前缀不同,不会误计两遍。
_LID_NOVERDICT_RE = re.compile(r"^crispasr: LID failed|^crispasr: .*-only — skipping language detection")

# 读音频阶段就失败的文件(process_one_input 里 :812,在判别【之前】就 return 20)
# —— 它不产生判别行,但它打的 error 行带原路径,可以据此把这个下标从对号队列里摘掉。
_UNREADABLE_RE = re.compile(r"crispasr: error: failed to read audio '([^']*)'")

# AED 能覆盖的语种码前缀。超出这个范围 AED 不会"换个语种转",只会拿汉字编造
# (crispasr_backend_firered_asr.cpp:49-70 + issue #199)—— 这就是兜底引擎存在的理由。
# 汉语方言码按 whisper-tiny 实际会吐的 ISO 639-1/4 取值列(它认 yue,方言多半归到 zh,
# 所以这份名单宁可宽一点:多算"范围内"= 少一次 qwen3 重跑,代价小)。
_AED_IN_RANGE = ("zh", "en", "yue", "cmn", "hak", "nan", "wuu", "hsn", "gan", "jin",
                 "mnp", "cxh", "cta", "cdo", "hne", "waw", "mwv")


def _in_aed_range(code: str) -> bool:
    return (code or "").lower().split("-")[0] in _AED_IN_RANGE


def _can_read_lid() -> bool:
    """筛子读得到东西吗:判别行受 !no_prints 控制,前置判别器关着就根本没有这行。"""
    return LID_VERBOSE and not _lid_is_off()


def _sieve_on(engine: str) -> bool:
    return (_can_read_lid() and USE_LID_FALLBACK and engine == ENGINE_PRIMARY)


def _sieve_hit_single(err: str):
    """单文件专用:整份 stderr 里只可能有一个判别行,判到范围外就返回 (码, 依据)。"""
    if not _can_read_lid():
        return None
    m = _LID_RE.search(err or "")
    if not m:
        return None
    code, src = m.group(1), m.group(2).strip()
    return None if _in_aed_range(code) else (code, src)


def _lid_map(lines: list, files: list, index_of_arg: dict):
    """把 stderr 里按出现顺序的语种判别结论对回本批文件。

    返回 ({文件下标: (码, 依据)}, 是否对不上)。

    对号靠一个事实:批量调用是【单线程顺序】跑的 —— crispasr_run.cpp:4974-4981 在
    n_processors==1(默认)时逐下标调 process_one_input,而前置判别就在这个函数里
    (:947-1021)、排在转写之前。所以"第 k 个判别事件 ↔ 第 k 个真走到了判别的文件"。
    两类行会让朴素的下标对应错位,这里各按规矩处理:
      · 读音频就失败的文件不占槽位,但它会打一行带原路径的 error —— 按路径(index_of_arg
        是暂存区 ASCII 名 → 下标 的反查表)把这些下标从队列里摘掉。
      · 走到了判别却没出码的文件照样占一个槽位,记成 None 占位。
    对完之后如果事件数比文件数还多,说明撞上了我没核到的第三种形态,整批【不采信】:
    宁可让 AED 正常落盘(等于这次没兜底),也不能把判别结论安错文件 —— 安错的后果是
    好文件被拉去重跑、真该救的那个留在编造内容上。
    """
    events = []
    skipped = set()
    for line in lines:
        m = _LID_RE.search(line)
        if m:
            events.append((m.group(1), m.group(2).strip()))
            continue
        if _LID_NOVERDICT_RE.search(line):
            events.append(None)
            continue
        m = _UNREADABLE_RE.search(line)
        if m:
            i = index_of_arg.get(os.path.normcase(m.group(1)))
            if i is not None:
                skipped.add(i)

    queue = [i for i in range(len(files)) if i not in skipped]
    if len(events) > len(queue):
        return {}, True
    verdicts = {queue[k]: v for k, v in enumerate(events) if v is not None}
    return verdicts, False


def _lid_summary(verdicts: dict, files: list, engine: str):
    """一批跑完后把语种分布压成一行(范围外的文件在扣下时就单独 warning 过了)。"""
    if not verdicts:
        logger.info(f"[{engine}] 本批没抓到语种判别行 —— 前置判别器未生效?"
                    f"(LID_VERBOSE={LID_VERBOSE})")
        return
    dist = {}
    for code, _src in verdicts.values():
        key = code if _in_aed_range(code) else f"{code}*"
        dist[key] = dist.get(key, 0) + 1
    logger.info(f"[{engine}] 语种判别 {len(verdicts)}/{len(files)} 个:"
                + " ".join(f"{k}×{v}" for k, v in sorted(dist.items(), key=lambda kv: -kv[1]))
                + "(*=不在 AED 可判范围)")


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
    """同卷硬链接零拷贝;失败退回原路径,最坏等于未修复前的行为。

    链接名用全局自增序号，不用本批下标：主链的 ONNX 路径会在"批次暂存"里再套
    一层"单文件暂存"（闸门和 crispasr 回退各自都要 ASCII 路径），两层各从 0 计数的
    话，内层会 unlink 掉外层正拿着的那个 asr0.wav 再改成别的文件，外层命令就指向
    错音频了。自增序号保证任何一次嵌套都不重名。
    """
    staged = {}
    for f in files:
        link = STAGE_DIR / f"{_STAGE_PREFIX}{next(_STAGE_SEQ)}{f.suffix.lower()}"
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

def _base_cmd(engine: str = ENGINE_PRIMARY):
    """拼命令骨架。两个引擎只差 --backend / -m / 要不要 --punc-model 与 -bs,
    其余(VAD、分块、-l auto、前置判别、严格管线)完全一致 —— 差别写在 ENGINES 里。
    """
    eng = ENGINES[engine]
    cmd = [
        CRISPASR_EXE,
        "--backend",       eng["backend"],
        "-m",              eng["model"],
        "-l",              CRISPASR_LANGUAGE,
        "--vad",
        "--chunk-seconds", str(VAD_MAX_SEGMENT_SEC),
        "--gpu-backend",   CRISPASR_GPU_BACKEND,
        "--strict-pipeline",
        "-t",              str(CRISPASR_THREADS),
        "--no-timestamps",
        "-ml",             "0",
        "-otxt",
    ]
    if CRISPASR_VAD_MODEL:
        cmd += ["-vm", CRISPASR_VAD_MODEL]
    # LID_VERBOSE 时不去掉 --no-prints 就抓不到语种判别行(CLI 那行受 !no_prints 控制),
    # 筛子就瞎了;代价是日志变成逐文件详细输出。
    if not LID_VERBOSE:
        cmd += ["--no-prints"]
    if VAD_MIN_SILENCE_MS:
        cmd += ["-vsd", str(VAD_MIN_SILENCE_MS)]
    if VAD_MAX_SPEECH_SEC:
        cmd += ["-vmsd", str(VAD_MAX_SPEECH_SEC)]
    if BEAM_SIZE and eng["beam"]:
        cmd += ["-bs", str(BEAM_SIZE)]
    # 标点后处理只给"自己不出标点"的引擎加(见 ENGINES 里那两条注释),留空 = 不加。
    if eng["punc"]:
        cmd += ["--punc-model", eng["punc"]]
    # 语种判别:只在 -l auto 时才需要决定前置判别器。填 off/none = 不加前置模型,
    # AED 内置的 LID 照常工作(它烧在模型里,关不掉);填 whisper/silero/ecapa/firered
    # 才追加 --lid-model,且必须指到本地文件,否则 crispasr 每个文件联网下载超时
    # (约 30 s)后按 crispasr_run.cpp:1017 强制 en 兜底。
    if CRISPASR_LANGUAGE == "auto":
        cmd += ["--lid-backend", CRISPASR_LID_BACKEND or "off"]
        if not _lid_is_off():
            cmd += ["--lid-model", CRISPASR_LID_MODEL]
    return cmd


def _engine_label(engine: str) -> str:
    eng = ENGINES[engine]
    return f"{engine}({eng['label']}/{os.path.basename(eng['model'])})"


def _run(cmd, on_tick=None, live=None):
    """执行 crispasr。不设超时:单个音频跑一两小时属正常工况。

    on_tick 给出时,标准错误改由子线程排空,主线程每 BATCH_POLL_SEC 秒回调一次,
    让调用方能在批次还在跑的时候就取走已产出的结果。

    live 给出时,标准错误按【行】实时追加进这个共享列表(而不是等进程结束才一次性
    拿到),语种筛子就是靠它在结算前看见判到的码 —— 批内结算会当场删源文件,判别结论
    要是只能等整批跑完才读到,该救的文件早就被 AED 的编造内容顶掉了。
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
    sink = live if live is not None else []

    def drain():
        # stderr 在 C 侧是不带缓冲的 fprintf,所以行到得比 .txt 落盘早;
        # 逐行读还顺带顶住了管道缓冲区(攒满 4 KB 会把子进程写阻塞)。
        try:
            for line in proc.stderr:
                if line.strip():
                    sink.append(line.rstrip("\r\n"))
        except Exception as e:
            logger.warning(f"读取 crispasr 标准错误出错:{e}")
        finally:
            try:
                proc.stderr.close()
            except Exception:
                pass

    th = threading.Thread(target=drain, daemon=True)
    th.start()
    try:
        if on_tick is None:
            th.join()
        else:
            while th.is_alive():
                th.join(BATCH_POLL_SEC)
                if not th.is_alive():
                    break
                try:
                    on_tick()
                except Exception as e:
                    logger.error(f"批内结算出错:{e}")
            th.join()
        proc.wait()
    finally:
        _current_proc = None
    return proc.returncode, "\n".join(sink)


def _crispasr_batch(files: list, on_done=None, engine: str = ENGINE_PRIMARY,
                     flagged=None):
    """
    一次调用转写多个文件。engine 用哪套引擎,配置见 ENGINES。

    -f 与 -of 均可重复,且 -of 给出时数量必须与 -f 相等(whisper.cpp 沿袭的契约)。
    输出写进临时目录,不污染音频目录。

    on_done(file_path, raw_text) 给出时,crispasr 每写完一个文件就在那个轮询点
    回调一次。判据是"下标更大的 .txt 已经出现"⇒ 下标更小的那个必然已经写完并
    关闭,以此避开读到半截的文件;最新的那一个留到批次结束再取。中间缺号(空结
    果可能不落文件)不会挡住前面已完成的结算。

    筛子(只对主引擎开):每次结算前先从【已经收到】的 stderr 行里取这个文件的语种
    结论,判到 AED 范围外的【不落盘、不回调 on_done、源文件留在原位】,只记进 flagged,
    由 process_batch 攒起来换兜底引擎重跑。必须实时读是因为批内结算当场就删源文件,
    等整批跑完再拿到判别结论就来不及了。对号的规矩与错位风险写在 _lid_map 里。
    兜底那一趟不再设筛子(它什么语种都能转,再筛就是自己筛自己),但语种分布照常进日志。

    返回 (results, error):
        results —— {Path: 原始文本};未产出 .txt 的文件、以及被筛子扣下的文件都不在
                   其中(VAD 判为"整段无语音"的仍以 "" 计入 —— 那种文件没有内容可编造,
                   判成任何语种都不必救)
        error   —— 进程级错误信息,正常退出为 None
    """
    can_read = _can_read_lid()
    sieve = _sieve_on(engine)
    if flagged is None:
        flagged = set()

    with tempfile.TemporaryDirectory() as tmpdir:
        bases = {f: os.path.join(tmpdir, f"o{i}") for i, f in enumerate(files)}
        results = {}
        cursor = [0]
        staged_of = {}
        arg_index = {}
        live = []
        distrust = [False]

        def last_written():
            last = -1
            for i, f in enumerate(files):
                if os.path.exists(bases[f] + ".txt"):
                    last = i
            return last

        def flush(upto):
            verdicts = {}
            if can_read:
                # 排空线程还在往 live 里追加,先取快照再对号,免得两次 flush 看到半截
                snapshot = list(live)
                verdicts, mismatch = _lid_map(snapshot, files, arg_index)
                if mismatch and not distrust[0]:
                    distrust[0] = True
                    logger.warning("语种判别事件数多于文件数 —— 撞上了没核过的错位形态,"
                                   "本批【不采信】筛子,全部按主引擎落盘")
                    verdicts = {}
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
                v = verdicts.get(i) if sieve else None
                if v is not None and not _in_aed_range(v[0]):
                    # 扣下:不写 results ⇒ process_batch 会把它算进 pending,再由
                    # flagged 认出来交给兜底引擎。源文件不 trash,所以数据一点没丢。
                    flagged.add(f)
                    logger.warning(f"语种判别 [{f.name}] = {v[0]} ({v[1]}) —— 不在 AED 可判范围,"
                                   f"AED 遇到它只会用汉字编造,已扣下待 {ENGINE_FALLBACK} 兜底")
                    continue
                results[f] = text
                if on_done:
                    on_done(f, text)

        def on_tick():
            k = last_written()
            if k > 0:
                flush(k)

        cmd = _base_cmd(engine)
        with staged_inputs(files) as paths:
            staged_of.update({os.path.normcase(p): f for f, p in paths.items()})
            arg_index.update({os.path.normcase(p): i
                              for i, p in enumerate(paths.values())})
            for f in files:
                cmd += ["-f", paths[f]]
            for f in files:
                cmd += ["-of", bases[f]]

            rc, err = _run(cmd, on_tick=on_tick if on_done else None, live=live)

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

        if can_read:
            _lid_summary(_lid_map(live, files, arg_index)[0], files, engine)

        if rc != 0:
            return results, f"CrispASR 退出码 {rc}:{err or '无标准错误输出'}"
        return results, None


def _crispasr_one(file_path: Path, engine: str = ENGINE_PRIMARY):
    """
    单文件转写,用于批量失败后的精确定位。
    返回 (text, error);text 为 "" 表示合法的空转写。

    单文件模式下筛子同样工作,而且更省事:一次调用只有一个文件,判别行必然是它的,
    不需要 _lid_map 那套对号 —— 判到 AED 范围外就直接换兜底引擎再跑一次(递归一层,
    兜底引擎不再筛)。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "out")
        with staged_inputs([file_path]) as paths:
            cmd = _base_cmd(engine) + ["-f", paths[file_path], "-of", base]
            rc, err = _run(cmd)

        if rc != 0:
            return None, f"CrispASR 退出码 {rc}:{err or '无标准错误输出'}"

        # 判到 AED 范围外 → 这份文本不认(它多半是编造的),换兜底引擎重跑一次。
        # 放在 rc 之后:进程级失败要照常计失败、进隔离区,不能被兜底悄悄盖掉。
        hit = _sieve_hit_single(err) if _sieve_on(engine) else None
        if hit:
            logger.warning(f"语种判别 [{file_path.name}] = {hit[0]} ({hit[1]})"
                           f" —— 不在 AED 可判范围,改用 {ENGINE_FALLBACK} 重跑")
            return _crispasr_one(file_path, ENGINE_FALLBACK)

        out_txt = base + ".txt"
        if not os.path.exists(out_txt):
            return "", None
        try:
            with open(out_txt, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read().strip(), None
        except Exception as e:
            return None, f"读取输出文件失败:{e}"


# ==================== 语种闸门 + FireRed ONNX 主链 ====================
#
# 这一整段只在 PRIMARY_LEG == "onnx" 时工作。crispasr 那两条腿(_crispasr_batch /
# _crispasr_one)一行没改,兜底 qwen3 照旧走它们 —— 换的只是"主力那一路由谁实现"。

_GATE_LANG_RE = re.compile(r"auto-detected language:\s*(\S+)\s*\(p\s*=\s*([\d.]+)\)")


def _gate_crispasr(extra):
    """闸门那两次 crispasr 短调用。

    固定 `--gpu-backend cpu`:独显要留给 ONNX 主链,而且这两步各只要半秒到一秒
    (10-09 校机纯 CPU 实测:VAD 导段 0.46 s、-dl 1.07 s),为它们建一次 CUDA 上下文
    纯属浪费,还会多一份 ggml-cuda 常驻显存。
    `-m` 用 whisper-tiny(77 MB):`--vad-export-raw` 和 `-dl` 都只要一个能加载的后端,
    tiny 是手里最小的那一个,而且判别器本来就是它。

    这里【不传】--no-prints:判别行是 crispasr 的 print,受 !no_prints 控制,加了它
    闸门就一行结论都抓不到(同一个坑在 LID_VERBOSE 那段写过)。
    也不传 --strict-pipeline 之外的任何东西到第 2 步:-dl 与 --vad-export-raw 同时给时,
    实测 0.46 s 就退出且不打印语种 = 静默假成功(10-09 校机 E 档),所以两步必须分开跑。
    """
    cmd = [CRISPASR_EXE, "-m", CRISPASR_LID_MODEL, "-t", str(GATE_THREADS),
           "--gpu-backend", "cpu"] + [str(x) for x in extra]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, encoding="utf-8",
                                errors="replace", creationflags=_POPEN_FLAGS)
    except FileNotFoundError:
        return -1, f"可执行文件未找到:{CRISPASR_EXE}"
    except Exception as e:
        return -1, str(e)
    bind_child(proc)          # 驱动被强杀时闸门子进程不能变孤儿
    try:
        err = proc.communicate(timeout=GATE_TIMEOUT_SEC)[1] or ""
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            err = proc.communicate()[1] or ""
        except Exception:
            err = ""
        return -124, f"闸门调用超时 {GATE_TIMEOUT_SEC:.0f}s:{err}"
    return proc.returncode, err


def _gate_vad_segments(src, tmpdir: str):
    """silero VAD 导真语音段 → ([(起秒, 止秒)], None) 或 (None, 错误说明)。

    必须用 --vad-export-raw:不带 -raw 的那份导出来是 kind="chunks"(30 s 网格),
    不是语音段(10-09 校机实测)。段表里 start/end 是【采样点】、t0_cs/t1_cs 是【百分秒】,
    文件里没写单位,所以优先取带 _cs 的那两个字段除以 100 —— 名字自己说明单位,不靠猜。

    还要防 crispasr 自己的一个二次兜底:silero 一条语音都没找到时,它对长素材不空手
    退出,而是打一行 "VAD returned no speech at all on a Ns clip — falling back to
    full-clip chunks",然后把整条按 30 s 网格当段表写出来(10-09 校机实测:120 s 日语歌
    → silero 0 段 → 段表 4 条整格;20 s 纯器乐 → 0 段、不落网格)。那些格子【不是】语音,
    留着会把"原始前 15 s"包装成"语音前缀",日志写成"原始前 15s 那段没用上" —— 恰好反了。
    所以按 stderr 那句标记(取纯 ASCII 片段,那行里的破折号经编码转换会花)把网格丢掉,
    让下面那条已经写好的 0 段分支处理;两条分支的处置本来就相同(都是判原始文件)。
    """
    json_out = os.path.join(tmpdir, "vad.json")
    rc, err = _gate_crispasr(["-f", src, "--vad", "-vm", GATE_VAD_MODEL,
                              "--vad-export-raw", json_out,
                              "--strict-pipeline", "--require-vad"])
    if not os.path.isfile(json_out):
        return None, f"VAD 没出段表(rc={rc}):{(err or '').strip()[-200:]}"
    if "full-clip chunks" in (err or ""):
        return [], None
    try:
        d = json.loads(Path(json_out).read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"VAD 段表读不进(rc={rc}):{e}"
    v = d.get("crispasr_vad", d)
    segs = []
    for s in (v.get("slices") or []):
        if "t0_cs" in s and "t1_cs" in s:
            segs.append((float(s["t0_cs"]) / 100.0, float(s["t1_cs"]) / 100.0))
        else:
            sr = float(v.get("sample_rate") or 16000)
            segs.append((float(s["start"]) / sr, float(s["end"]) / sr))
    return segs, None


def _gate_load_audio(src):
    """按链路口径读成 16k 单声道 float32;读不了返回 None。

    libsndfile 认 wav/flac/ogg/oga/opus/mp3,不认 m4a / mp4 / aac —— 那几种原来是靠
    crispasr 内部的 ffmpeg 解的。读不了的文件闸门只能退回"直接判原始文件",主链也接不了,
    由调用方转给 crispasr 的【同一个模型】(不换引擎:语种没问题,只是容器解不了)。
    """
    try:
        return chain_module().load_audio(src)
    except Exception as e:
        logger.info(f"闸门:{os.path.basename(str(src))} 这个容器链路读不了"
                    f"({type(e).__name__}),按原始文件判语种")
        return None


def _speech_prefix(wav, sr, segs, want_sec: float):
    """把语音段按时间顺序接起来,接够 want_sec 就截断。够不到返回 (None, None)。

    第二个返回值说这段前缀占在【原始时间轴】的哪儿:{"head": 首个用到的采样秒位,
    "tail": 末个用到的采样秒位, "k": 用了几段},只给日志用。带上它是因为光一句
    "语音前缀 15 s" 看不出闸门到底跳过了多少片头,而"原始前 15 s 那段没用上"这种话
    在整段只有一个长语音段的素材上是假的(10-09 校机实测:段表 = 一条 0.03–79.55 s,
    拼出来的前缀就是原始的前 15 s,一秒都没跳过)。
    """
    parts, got, head, tail, k = [], 0.0, None, 0.0, 0
    for s, e in segs:
        i0 = max(0, int(s * sr))
        i1 = min(int(wav.size), int(e * sr))
        if i1 <= i0:
            continue
        if head is None:
            head = i0 / sr
        parts.append(wav[i0:i1])
        before, got = got, got + (i1 - i0) / sr
        k += 1
        if got >= want_sec:
            tail = (i0 + int((want_sec - before) * sr)) / sr
            break
        tail = i1 / sr
    if not parts:
        return None, None
    out = np.concatenate(parts)
    return out[: int(want_sec * sr)], {"head": head or 0.0, "tail": tail, "k": k}


def gate_language(src):
    """语种闸门:silero VAD → 语音前缀 → whisper-tiny -dl。

    返回 {"code": 码|None, "p": 置信度|None, "why": 依据说明, "readable": 主链能不能读}。

    code=None 的含义是【闸门没结论】,调用方必须按主链跑并把这一行写进日志 ——
    不能拿"看不见"当"范围外",否则一个 crispasr 起不来就能把整批文件拖去兜底引擎。
    判据本身沿用 _AED_IN_RANGE(与 crispasr 筛子同一份名单,没有另立一套)。
    """
    import soundfile as sf
    src = str(src)
    with tempfile.TemporaryDirectory(prefix="asrgate") as tmpdir:
        segs, verr = _gate_vad_segments(src, tmpdir)
        if segs is None:
            logger.warning(f"闸门 VAD 失败:{verr}")
            segs = []
        wav = _gate_load_audio(src)
        if wav is None:
            target, why = src, "容器读不了,直接判原始文件"
            readable = False
        else:
            readable = True
            pref, meta = (_speech_prefix(wav[0], wav[1], segs, GATE_LID_SEC)
                          if segs else (None, None))
            if pref is None or len(pref) / wav[1] < GATE_MIN_SPEECH_SEC:
                target, why = src, (f"silero 判 {len(segs)} 段、拼不出 "
                                   f"{GATE_MIN_SPEECH_SEC:g}s 语音,直接判原始文件")
            else:
                target = os.path.join(tmpdir, "lid.wav")
                sf.write(target, pref, int(wav[1]), subtype="PCM_16")
                why = (f"silero {len(segs)} 段 → 语音前缀 {len(pref) / wav[1]:.1f}s"
                       f"(取自原始 {meta['head']:.1f}–{meta['tail']:.1f}s 的 "
                       f"{meta['k']} 段)")
        rc, err = _gate_crispasr(["-f", target, "-dl"])
        m = _GATE_LANG_RE.search(err or "")
        if not m:
            return {"code": None, "p": None, "readable": readable,
                    "why": f"没抓到判别行(rc={rc},判别器 whisper-tiny,"
                           f"{why}):{(err or '').strip()[-160:]}"}
        return {"code": m.group(1).lower(), "p": float(m.group(2)),
                "readable": readable, "why": f"{m.group(1)} p={m.group(2)} / {why}"}


# ---------- ONNX 链路的加载与调用 ----------

_chain_mod = None
_chain_engine = None
_chain_lock = threading.Lock()
_chain_fatal = ""      # 建链路失败的原因;非空就整轮降级到 crispasr 主链


def chain_module():
    """导入本包 onnx/asr_chain.py,并把它的目录全局钉成本 CONFIG 的值。

    这一步很便宜:asr_chain 在 import 时只拉 numpy/soundfile,onnxruntime 是建会话时
    才 import 的。所以闸门读音频用它,真正吃几个 GB 的三个会话由 chain_engine() 负责。
    """
    global _chain_mod
    if _chain_mod is None:
        if ONNX_DIR not in sys.path:
            sys.path.insert(0, ONNX_DIR)
        import asr_chain as chain
        # 目录以本 CONFIG 为唯一权威(链路侧那几个全局只是默认值)。逐个赋值而不是只给
        # MODELS:cpu 分支踩过那个坑 —— 只搬 MODELS 时 VAD/Punc 还指在旧目录上,
        # 表现为 ASR 换了地方、断句和标点读别的机器留下的文件。
        chain.MODELS = Path(ONNX_MODELS_DIR)
        chain.ASR_DIR = ONNX_ASR_DIR
        chain.VAD_DIR = Path(ONNX_VAD_DIR)
        chain.PUNC_DIR = Path(ONNX_PUNC_DIR)
        chain.PUNC_FILE = ONNX_PUNC_FILE
        # 驱动自己要用 numpy(拼语音前缀、VAD 内存直喂),从链路那份直接取,不再 import 一次:
        # 装了 onnxruntime 就必然装了 numpy,而这条腿【不】是标准库依赖 —— 真缺了这里就抛
        # ImportError,由 main_loop 的起批前自检拦下,不会静默降级。
        globals()["np"] = chain.np
        _chain_mod = chain
    return _chain_mod


class OnnxChain:
    """FireRedASR2S ONNX 全链路在本进程常驻一份:FireRedVAD → AED → FireRedPunc。

    与 cpu 分支那个独立驱动(xhs-chain-cpu.py 的 Engine)同一口径,两点值得记:
      · VAD 不走"临时 wav 往返" —— 链路自带的 run_vad 会 sf.write 到 %TEMP%\\_chain_vad.wav
        再读回来,与生产批次同名的话会互相踩,这里改成内存直喂;
      · float→int16 用 rint,与 PCM_16 落盘的舍入口径一致,免得换条路就换了量化误差。
    标点恒在 CPU:它是 4 亿参数的 BERT,上卡只跟 AED 编码器抢显存,而每段只推一次。
    """

    def __init__(self):
        import onnxruntime as ort
        chain = chain_module()
        self.chain = chain
        sys.path.insert(0, str(ONNX_VAD_DIR))
        from infer_onnx import FireRedVadOnnx

        _orig = ort.InferenceSession

        def _vad_session(path_or_bytes, sess_options=None, *a, **kw):
            # vendor 的 infer_onnx.py 建会话时既没设 intra_op_num_threads(ORT 的 0 =
            # 吃满所有物理核),也没指定 providers —— 装了 onnxruntime-gpu 时它自己会挑
            # CUDA。VAD 只有 2.4 MB 权重、每个文件对整条音频提一次特征,放 CPU 更划算,
            # 而且"绝不用核显"这条要求最好处处都设防。这里临时替换工厂,不改 vendor 文件
            # (它是上游哈希锁定的原件,改了升级会冲突)。
            so = sess_options if sess_options is not None else ort.SessionOptions()
            if not getattr(so, "intra_op_num_threads", 0):
                so.intra_op_num_threads = ONNX_VAD_THREADS
            kw.setdefault("providers", ["CPUExecutionProvider"])
            return _orig(path_or_bytes, sess_options=so, *a, **kw)

        t_all = time.perf_counter()
        try:
            ort.InferenceSession = _vad_session
            self.vad = FireRedVadOnnx(model_dir=str(ONNX_VAD_DIR))
        finally:
            ort.InferenceSession = _orig
        t_vad = time.perf_counter() - t_all

        t0 = time.perf_counter()
        self.asr = chain.build_recognizer(
            "aed", ONNX_ASR_THREADS, mode=ONNX_MODE, cache=ONNX_CACHE_LEN,
            graph=ONNX_GRAPH, provider=ONNX_PROVIDER,
            dec_provider="cuda" if ONNX_DEC_ON_GPU else None)
        t_asr = time.perf_counter() - t0
        # 装的是 onnxruntime-gpu ≠ 真的跑在卡上:CUDA/cuDNN 运行库不齐时 ORT 只打一行
        # 警告就把会话整体退回 CPU。必须读 get_providers() 的实际结果,否则就是
        # "说好用 N 卡、实际在 CPU 上跑一整夜"而没人看得出来。
        if ONNX_PROVIDER == "cuda" and "CUDAExecutionProvider" not in self.asr.enc_ep:
            raise RuntimeError(
                f"请求了 cuda 但编码器实际落在 {self.asr.enc_ep}:"
                f"装的是 CPU 版 onnxruntime,或 CUDA/cuDNN 运行库不齐(看 ORT 的警告行)")
        t0 = time.perf_counter()
        self.punc = chain.Punc(threads=ONNX_PUNC_THREADS)
        t_punc = time.perf_counter() - t0
        logger.info(f"ONNX 主链加载:VAD {t_vad:.1f}s + AED({ONNX_GRAPH}/{ONNX_MODE}, "
                    f"{self.asr.weight_mb} MB) {t_asr:.1f}s + Punc {t_punc:.1f}s,"
                    f"合计 {time.perf_counter() - t_all:.1f}s")
        logger.info(f"执行提供者:enc={self.asr.enc_ep} dec={self.asr.dec_ep} "
                    f"punc={self.punc.ep}(VAD 恒定 CPU)")

    def segments(self, wav, sr):
        w16 = np.rint(np.clip(wav, -1.0, 1.0) * 32768.0).clip(-32768, 32767).astype(np.int16)
        feat = self.vad._extract_features(w16)
        probs = self.vad._run_model(feat)
        decisions = self.vad.postprocessor.process(probs.tolist())
        return self.vad.postprocessor.decisions_to_segments(decisions, len(wav) / sr)

    def transcribe(self, path):
        """一个文件走完 VAD → AED → Punc。返回 (文本, 统计);文本 "" = 无语音。"""
        t0 = time.perf_counter()
        wav, sr = self.chain.load_audio(path)
        dur = len(wav) / sr
        segs = self.segments(wav, sr)
        speech = sum(e - s for s, e in segs)
        texts, n_trunc = [], 0
        for s, e in segs:
            seg = wav[int(s * sr): int(e * sr)]
            if seg.size < int(0.1 * sr):
                continue
            r = self.asr.transcribe_wav(seg, sr)
            n_trunc += bool(r["truncated"])
            texts.append(r["text"].strip())
        raw = "".join(texts)
        if n_trunc:
            logger.warning(f"{os.path.basename(str(path))}:{n_trunc} 段撞上 cache 上限被截断")
        if raw and self.punc is not None:
            raw = "".join(self.punc.add(x) for x in texts)
        return raw, {"dur": round(dur, 2), "speech": round(speech, 2),
                     "segs": len(segs), "sec": round(time.perf_counter() - t0, 2),
                     "trunc": n_trunc}


def chain_engine():
    """惰性建一次 ONNX 主链(几个 GB 的会话,建起来要几十秒)。

    建不起来就整轮降级到 crispasr 主链并把原因记着 —— 宁可慢,不能把文件判成失败:
    失败进隔离区之后这批就再也不会被扫到,而"权重缺一件"和"音频本身坏了"在下游看不出区别。
    """
    global _chain_engine, _chain_fatal
    with _chain_lock:
        if _chain_engine is None and not _chain_fatal:
            try:
                _chain_engine = OnnxChain()
            except Exception as e:
                _chain_fatal = f"{type(e).__name__}:{e}"
                logger.critical(f"ONNX 主链起不来,本轮改走 crispasr 主链:{_chain_fatal}")
        return _chain_engine


def onnx_leg_setup() -> int:
    """PRIMARY_LEG="onnx" 的起批前校验。返回 0 = 通过,2 = 致命(不启动)。

    这里的判据是"会不会静默降级",不是"文件在不在":
      · 三张会话【真建一次】而不是查文件名 —— 缺哪张图、CUDA 请求了却落回 CPU、
        vendor 的 infer_onnx 换了接口,只有真建才看得见;建好的是进程级单例,
        起批前建和跑第一个文件时建是同一份,不重复花那几十秒。
      · 起批前拒绝启动,而不是等 chain_engine() 失败后整轮偷偷改走 crispasr:那个降级
        是给"跑到一半卡子坏了"准备的兜底,不是给"配置本来就不对"准备的。
    """
    if ONNX_PROVIDER not in ("cpu", "cuda"):
        logger.critical(f'ONNX_PROVIDER={ONNX_PROVIDER!r} 只认 "cpu" | "cuda":'
                        f'DirectML / OpenVINO 会挑中 AMD 核显,本链禁用')
        return 2
    if not os.path.isfile(GATE_VAD_MODEL):
        logger.critical(f"闸门 VAD 模型不存在:{GATE_VAD_MODEL}\n"
                        f"  闸门没有语音段就只能拿原始前 15 s 去判,静音开头的外语文件会判错")
        return 2
    if not os.path.isfile(CRISPASR_LID_MODEL):
        logger.critical(f"闸门判别模型不存在:{CRISPASR_LID_MODEL}\n"
                        f"  -dl 判不出语种 = 所有文件都按主链跑,AED 会对范围外语种编造汉字")
        return 2
    for d in (ONNX_MODELS_DIR, ONNX_VAD_DIR, ONNX_PUNC_DIR):
        if d and not os.path.isdir(d):
            logger.critical(f"ONNX 模型目录不存在:{d}(补全资源跑 onnx/fetch_assets.py)")
            return 2
    try:
        chain_module()
    except Exception as e:
        logger.critical(f"ONNX 链路代码/依赖起不来({ONNX_DIR}):{type(e).__name__}:{e}\n"
                        f"  要么按 onnx/requirements*.txt 装齐,要么把 PRIMARY_LEG 改回 \"crispasr\"")
        return 2
    if chain_engine() is None:
        logger.critical(f"ONNX 主链建不起来:{_chain_fatal}\n"
                        f"  修好后重跑,或把 PRIMARY_LEG 改成 \"crispasr\" 走原来的两条腿")
        return 2
    logger.info(f"主链:FireRedASR2S ONNX 全链路(VAD+AED+Punc 常驻),"
                f"provider={ONNX_PROVIDER} 图={ONNX_GRAPH} 模式={ONNX_MODE}")
    logger.info(f"语种闸门:crispasr silero VAD 导段 + whisper-tiny -dl"
                f"({os.path.basename(GATE_VAD_MODEL)} + {os.path.basename(CRISPASR_LID_MODEL)})")
    return 0


def gate_selfcheck(candidates: list) -> str:
    """拿前几个待处理文件实跑一次闸门,确认【能抓到判别行】。返回 '' = 通过。

    这一层必须实测而不能只查文件:闸门坏了不会产生任何错误 —— crispasr 照常返回 0,
    只是 stderr 里没有那行 `auto-detected language: xx (p=...)`,于是每个文件都判成
    "在范围内",整批外语素材被 AED 编成汉字并正常落盘,下游看不出区别(与 crispasr
    筛子那个 --no-prints 瞎掉的老坑同源)。逐个试,试出结论就放行。
    """
    tried = []
    for f in candidates:
        with staged_inputs([f]) as paths:
            gate = gate_language(paths[f])
        if gate["code"]:
            logger.info(f"闸门自检通过:{f.name} → {gate['why']}")
            return ""
        tried.append(f"{f.name}:{gate['why']}")
    return ("闸门跑不出语种结论,已试 " + str(len(tried)) + " 个文件:\n  "
            + "\n  ".join(tried) +
            "\n  继续跑等于所有文件都按主链走,范围外的语种会被 AED 编造。"
            "\n  查 CRISPASR_EXE 能否 -dl、GATE_THREADS/GATE_TIMEOUT_SEC 是否过短")


def _onnx_one(file_path: Path, staged: str = None):
    """主链跑一个文件,返回 (文本, 错误)。与 _crispasr_one 同一份契约。

    staged 是批量入口已经建好的 ASCII 硬链接路径:闸门那两次 crispasr 调用【必须】走它
    (crispasr 是 ANSI 程序,argv 经系统 ACP 转换,文件名里 GBK 表示不了的字符会变成 '?',
    路径随之失效),链路侧也顺带用它 —— libsndfile 在 Windows 上打不开非 ASCII 路径,
    这与 crispasr 那个坑同源,暂存区一次解决两处。
    """
    eng = chain_engine()
    if eng is None:
        return _crispasr_one(file_path, ENGINE_PRIMARY)
    src = staged or str(file_path)
    try:
        text, st = eng.transcribe(src)
    except Exception as e:
        # 读不了音频的容器交给 crispasr 的同一个模型:它自带 ffmpeg 解码,不换引擎。
        logger.warning(f"ONNX 链处理 {file_path.name} 失败,转 crispasr {ENGINE_PRIMARY}:"
                       f"{type(e).__name__}:{e}")
        return _crispasr_one(file_path, ENGINE_PRIMARY)
    logger.info(f"ONNX 链 {file_path.name}:音频 {st['dur']}s / 语音 {st['speech']}s / "
                f"{st['segs']} 段 / 用时 {st['sec']}s"
                + (f" / 截断 {st['trunc']} 段" if st["trunc"] else ""))
    return text, None


def transcribe_batch_onnx(files: list, on_done=None, flagged=None):
    """主链那一批:逐文件先过闸门,范围内的交给 ONNX 全链路,范围外的记进 flagged。

    与 _crispasr_batch 返回同一份契约 (results, error),process_batch 因此不用改形状:
      · results[f] = 文本(含合法空串) ⇒ 已产出;
      · f 在 flagged 里 ⇒ 由 process_batch 攒成兜底轮换 crispasr qwen3 重跑;
      · 两处都没有 ⇒ 没产出,进逐文件重跑。
    模型常驻进程,所以这里不需要 crispasr 那套"一次调用吃掉整批"的攒批逻辑,也不需要在
    stderr 里对号 —— 语种是【调用主链之前】按文件独立判的,天然不会错位。
    每跑完一个文件就回调 on_done,批内实时结算/删源文件的原行为不变。
    """
    if flagged is None:
        flagged = set()
    eng = chain_engine()
    if eng is None:
        return _crispasr_batch(files, on_done=on_done, engine=ENGINE_PRIMARY,
                               flagged=flagged)
    results = {}
    codes = {}
    with staged_inputs(files) as paths:
        for i, f in enumerate(files):
            if stop_requested():
                logger.info(f"已请求停止,主链在第 {i} 个文件处收工(已产出 {len(results)} 个)")
                break
            gate = gate_language(paths[f])
            code = gate["code"]
            if code:
                codes[i] = (code, gate["why"])
            if code and not _in_aed_range(code):
                flagged.add(f)
                logger.warning(f"语种闸门 [{f.name}] = {gate['why']} —— 不在 AED 可判范围,"
                               f"主链不跑,待 {ENGINE_FALLBACK} 兜底")
                continue
            if not code:
                logger.warning(f"语种闸门 [{f.name}] 没结论({gate['why']}),按主链跑")
            if gate["readable"]:
                text, error = _onnx_one(f, staged=paths[f])
            else:
                # 容器解不了不等于文件坏了:交给 crispasr 的【同一个】主引擎模型,
                # 它内部带 ffmpeg(m4a / mp4 / aac),不换语种、不换引擎。
                logger.info(f"语种闸门 [{f.name}] = {gate['why']};"
                            f"链路读不了这个容器,按 {ENGINE_PRIMARY} 走 crispasr")
                text, error = _crispasr_one(f, ENGINE_PRIMARY)
            if error:
                logger.error(f"主链失败 [{f.name}]:{error}")
                continue
            results[f] = text
            if on_done:
                on_done(f, text)
    _lid_summary(codes, files, f"{ENGINE_PRIMARY}/onnx")
    return results, None


def transcribe_batch(files: list, on_done=None, engine: str = ENGINE_PRIMARY,
                     flagged=None):
    """按 PRIMARY_LEG 分发到 ONNX 主链或 crispasr。兜底引擎永远走 crispasr。"""
    if PRIMARY_LEG == "onnx" and engine == ENGINE_PRIMARY:
        return transcribe_batch_onnx(files, on_done=on_done, flagged=flagged)
    return _crispasr_batch(files, on_done=on_done, engine=engine, flagged=flagged)


def transcribe_one(file_path: Path, engine: str = ENGINE_PRIMARY):
    """单文件版的同一分发。逐文件重跑阶段用得上:flagged 的文件已经在 process_batch
    里被指派成兜底引擎,不会被送到主链上重复过闸门。"""
    if PRIMARY_LEG == "onnx" and engine == ENGINE_PRIMARY:
        with staged_inputs([file_path]) as paths:
            src = paths[file_path]
            gate = gate_language(src)
            if gate["code"] and not _in_aed_range(gate["code"]):
                logger.warning(f"语种闸门 [{file_path.name}] = {gate['why']}"
                               f" —— 不在 AED 可判范围,改用 {ENGINE_FALLBACK}")
                return _crispasr_one(file_path, ENGINE_FALLBACK)
            if not gate["code"]:
                logger.warning(f"语种闸门 [{file_path.name}] 没结论({gate['why']}),按主链跑")
            if gate["readable"]:
                return _onnx_one(file_path, staged=src)
            logger.info(f"主链 [{file_path.name}] 换 crispasr 解容器(链路读不了这个格式)")
            return _crispasr_one(file_path, ENGINE_PRIMARY)
    return _crispasr_one(file_path, engine)


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
    # 被筛子扣下的文件(语种判到 AED 范围外)。逐文件重跑时靠它决定该直接用哪个引擎,
    # 免得把一个已经知道 AED 转不了的文件再喂给 AED 一次。
    flagged = set()

    if stats.batch_ok and len(batch) > 1:
        logger.info(f"批量转写 {len(batch)} 个文件 [{_engine_label(ENGINE_PRIMARY)}]")
        started = time.time()
        settled = set()
        settled_lock = threading.Lock()

        def on_done(f, raw):
            with settled_lock:
                if f in settled:
                    return
                settled.add(f)
            settle(out_txt, f, raw, rules_list, stats)

        def settle_leftovers(files_, mapping):
            """轮询点没来得及结算的(最新一个/中途缺号),批次结束后补上。"""
            for f in files_:
                if f in mapping:
                    with settled_lock:
                        if f in settled:
                            continue
                        settled.add(f)
                    settle(out_txt, f, mapping[f], rules_list, stats)

        results, error = transcribe_batch(batch, on_done=on_done,
                                          engine=ENGINE_PRIMARY, flagged=flagged)
        logger.info(f"批次耗时 {(time.time() - started) / 60:.1f} min,"
                    f"产出 {len(results)}/{len(batch)}")
        notify(f"asr批次耗时 {(time.time() - started) / 60:.1f} min,")
        settle_leftovers(batch, results)

        # ---------- 兜底轮 ----------
        # 被扣下的文件源音频还在原位,这里攒成一批换 qwen3 跑(一次模型加载吃掉整批)。
        hold = [f for f in batch if f in flagged and f.exists()]
        if hold:
            logger.info(f"{len(hold)} 个文件语种不在 AED 可判范围,"
                        f"换 {_engine_label(ENGINE_FALLBACK)} 兜底重跑")
            fb_started = time.time()
            fb_results, fb_error = transcribe_batch(hold, on_done=on_done,
                                                    engine=ENGINE_FALLBACK)
            logger.info(f"兜底批次耗时 {(time.time() - fb_started) / 60:.1f} min,"
                        f"产出 {len(fb_results)}/{len(hold)}")
            if fb_error:
                logger.warning(f"兜底批量调用异常:{fb_error}")
            settle_leftovers(hold, fb_results)
            results.update(fb_results)
            for f in hold:
                if f not in fb_results:
                    logger.warning(f"兜底也没产出,留在原位逐文件再试:{f.name}")

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
        # 已经知道 AED 转不了的文件直接走兜底引擎
        engine = ENGINE_FALLBACK if f in flagged else ENGINE_PRIMARY
        logger.info(f"单独处理:{f.name} [{engine}]")
        started = time.time()
        raw, error = transcribe_one(f, engine)
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
    for key in (ENGINE_PRIMARY, ENGINE_FALLBACK):
        if key not in ENGINES:
            logger.critical(f"引擎键 {key!r} 不在 ENGINES 里(可用:{'/'.join(ENGINES)})")
            return 2

    # 模型【全部】启动前校验。缺文件时 crispasr 未必报非零码:--punc-model 缺失会被
    # 静默跳过 = 整棵树没有标点;--lid-model 缺失则每个文件联网超时约 30 秒后按 en
    # 硬转,七八千个文件 = 上百小时白等。所以一律判致命。
    for key in (ENGINE_PRIMARY, ENGINE_FALLBACK):
        eng = ENGINES[key]
        if not os.path.isfile(eng["model"]):
            logger.critical(f"{key} 模型文件不存在:{eng['model']}")
            return 2
        if eng["punc"] and not os.path.isfile(eng["punc"]):
            logger.critical(f"{key} 标点模型不存在:{eng['punc']}\n"
                            f"  要么补上文件,要么把 ENGINES[{key!r}]['punc'] 置空(输出将没有标点)")
            return 2
    for key in (ENGINE_PRIMARY, ENGINE_FALLBACK):
        eng = ENGINES[key]
        punc = (f"+punc {os.path.basename(eng['punc'])}" if eng["punc"]
                else "不加标点后处理")
        logger.info(f"引擎 {key}:{eng['label']} / {eng['backend']}"
                    f" [{os.path.basename(eng['model'])}, {punc}]")
    if USE_LID_FALLBACK:
        leg = ("FireRedASR2S ONNX 全链路" if PRIMARY_LEG == "onnx" else "crispasr")
        logger.info(f"策略:{ENGINE_PRIMARY} 主力({leg}),{ENGINE_FALLBACK} 兜底"
                    f"(语种判到 AED 范围外的文件换引擎重跑)")
    else:
        logger.info(f"策略:只用 {ENGINE_PRIMARY},判到范围外仅在日志里报一句")

    if (CRISPASR_VAD_MODEL and CRISPASR_VAD_MODEL != "webrtc"
            and not os.path.isfile(CRISPASR_VAD_MODEL)):
        # 不致命:crispasr 会退回固定分块继续跑,但转写会在语句中途被切断
        logger.warning(f"VAD 模型不存在:{CRISPASR_VAD_MODEL}(VAD 将失效)")

    if CRISPASR_LANGUAGE == "auto":
        if _lid_is_off():
            logger.info("语种:-l auto,只用 AED 内置语种判别,不挂前置判别器"
                        f"(可判范围 {'/'.join(AED_COVERS)})—— 此时【兜底不会触发】")
            if USE_LID_FALLBACK:
                logger.warning("USE_LID_FALLBACK=True 但没有前置判别器,筛子读不到东西:"
                               "所有文件都会按 AED 落盘")
        elif not (CRISPASR_LID_MODEL and os.path.isfile(CRISPASR_LID_MODEL)):
            logger.critical(
                f'-l auto + --lid-backend {CRISPASR_LID_BACKEND} 需要前置模型,但文件不存在:'
                f'{CRISPASR_LID_MODEL or "(未配置)"}\n'
                f'  要么填对顶部资源块里的 CRISPASR_LID_MODEL,要么把 CRISPASR_LID_BACKEND'
                f' 改回 "off"(AED 内置判别照常工作)')
            return 2
        else:
            logger.info(f"语种:-l auto + 前置判别器 {CRISPASR_LID_BACKEND}"
                        f" ({CRISPASR_LID_MODEL}) —— AED 不会因判到的码改变输出,"
                        f"这一层只用来挑文件交给 {ENGINE_FALLBACK}")
            if USE_LID_FALLBACK and not LID_VERBOSE:
                logger.warning("筛子开着但 LID_VERBOSE=False:判别行被 --no-prints 吞掉,"
                               "兜底实际不会触发。要兜底就把 LID_VERBOSE 打开")
    else:
        logger.warning(f"CRISPASR_LANGUAGE={CRISPASR_LANGUAGE!r} 不是 auto:"
                       f"前置判别器不会挂上({ENGINE_FALLBACK} 兜底也随之失效)")
    if PRIMARY_LEG == "onnx":
        rc = onnx_leg_setup()
        if rc:
            return rc
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
    if PRIMARY_LEG == "onnx":
        head = [f for fl in groups.values() for f in fl][:GATE_SELFCHECK_FILES]
        why = gate_selfcheck(head)
        if why:
            logger.critical(why)
            return 2
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
#   python crisper-xhs-qwen-asr.py                前台跑(关窗口即断,但 Job Object 会连带杀掉
#                                      crispasr.exe,不会留孤儿占显存)
#   python crisper-xhs-qwen-asr.py --start    后台跑,控制台输出重定向到 txt\log\console_*.log
#   python crisper-xhs-qwen-asr.py --stop     建 STOP 标志,实例在下一批边界优雅退出
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
    print(f"停止:{os.path.abspath(__file__)} --stop")
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
    _USAGE = ("用法: python crisper-xhs-qwen-asr.py [--start | --stop]\n"
              "  (无参数)  前台转写整个队列\n"
              "  --start   后台转写(控制台输出 -> txt\\log\\console_*.log)\n"
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
