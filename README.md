# my-asr-toolchain —— 四条语音转写链的完整复原包

小红书口播素材的批转（batch ASR）工具链。四套引擎、五个驱动，每一套都带
**机器可读的资源清单 + 一键复原脚本**：fork 之后三条命令就能把引擎、运行库、模型权重
全部从公开链接拉到本机，逐件按 sha256 验收，不需要在网页上一页一页找。

仓库里不放任何大文件（模型权重、dll、zip 全部只存 URL + 字节数 + sha256）。

---

## 分支地图

| 分支 | 目录 | 驱动 | 跑在哪 |
|---|---|---|---|
| `main` | —— | —— | 只有这一页 |
| `cpu` | `FireRed-ONNX/` | `Oc-fired.py` | 无显卡机器（纯 onnxruntime，四件全链） |
| `cpu` | `Qwen3/` | `Cc-qwen.py` | 无显卡机器（CrispASR CPU 构建，单模型出成品） |
| `gpu` | `CrispASR-FireRed/` | `Wlid-Og-fired-Cg-qwen.py` | 有 CUDA 的机器（语种闸门判到范围内 → FireRedASR2 的 ONNX 全链，编码器可上 N 卡；范围外回退 crispasr qwen3。`PRIMARY_LEG="crispasr"` = 10-09 之前的双引擎走法，留着当降级与对拍参照） |
| `gpu` | `CrispASR-Qwen/` | `Cg-qwen.py` | 有 CUDA 的机器（qwen3 单引擎） |
| `gpu` | `faster_whisper/` | `Wg-large.py` | 有 CUDA 的机器（faster-whisper large-v3） |

每个目录自己是一份完整交付：`README.md`（复原三步 + 路径与环境变量表 + 组件出处等级表 +
已知洞）、`assets.json`（机器可读清单）、`fetch_assets.py`（通用下载/验收器，标准库实现）、
`requirements.txt`。选一条链照那个目录的 README 走就行，跨目录不用互相看。

## 驱动的命名记号法

文件名里就写清楚"哪条腿、什么引擎、跑在哪类机器、按什么次序"。一个 `-` 分隔的段 = 一条腿，
段内按 **引擎字母 + 机器档 + 模型词（全小写）** 拼，段的次序就是运行次序（最左边的先跑）：

| 记号 | 含义 |
|---|---|
| `W` | whisper 系（whisper-tiny 只做语种判别；faster-whisper 做转写） |
| `C` | CrispASR（gguf 权重那套引擎，`--backend` 选 fired / qwen） |
| `O` | FireRedASR2S 的 ONNX 链路（纯 onnxruntime，无 torch、无 sherpa-onnx） |
| `g` | 这条腿要吃 CUDA / N 卡 |
| `c` | 这条腿纯 CPU |
| `fired` | FireRedASR2-AED |
| `qwen` | qwen3-ASR |
| `large` | whisper large-v3 权重 |
| `lid` | 只出语种码、不出文本 |

五个驱动逐个拆读。旧名一并列着，因为 10-10 之前的记录、日志和外部引用用的全是旧名：

| 现名 | 拆读 | 10-10 前的旧名 |
|---|---|---|
| `Wlid-Og-fired-Cg-qwen.py` | whisper 先判语种 → 判到范围内走 FireRed ONNX 全链（N 卡）→ 判到范围外回退 crispasr qwen3（N 卡） | `crisper-xhs-qwen-asr.py`（10-08 前叫 `xhs-asr.py`） |
| `Cg-qwen.py` | crispasr qwen3，N 卡 | `crispasr-Qwen.py` |
| `Wg-large.py` | faster-whisper large-v3，N 卡 | `whisper-batch10.py`（`batch10` 是当时的批大小 10） |
| `Oc-fired.py` | FireRed ONNX 全链，CPU | `xhs-chain-cpu.py` |
| `Cc-qwen.py` | crispasr qwen3，CPU | `crispasr-Qwen-cpu.py` |

第三列是这张表存在的全部理由，所以它不是替换出来的：10-10 那轮改名按全文替换跑，脚本分不清一行是「指现在这个文件」还是「记它以前叫什么」，把这些旧名一起换成了新名，读起来就成了「旧名 = 现名」。上面那些旧名已逐处对 git 历史修回。

两处边界写在这，免得被当成漏改：

* 全套只有 `Wlid` 这一段没标机器档。它实际被驱动钉在 CPU 上跑 —— 闸门那两次 crispasr 短调用
  固定传 `--gpu-backend cpu`（独显要留给主链，这两步各只要半秒到一秒，为它们建一次 CUDA
  上下文纯属浪费，还多一份 ggml-cuda 常驻显存）。
* **被 `import` 的东西不进这套命名**：`asr_chain.py` / `aed_ort.py` / `lid_onnx.py` /
  `infer_onnx.py` 一律保持原名 —— 连字符不是合法的 Python 标识符字符，`import O-fired` 当场
  是语法错误，而这几件都是按模块名被 import 进来的。`fetch_assets.py` / `setup.py` /
  `dequant_aed.py` / `dequant_punc.py` / `lid_export*.py` 同样没改：它们的名字里不含
  "引擎 × 模型 × 机器"这三样，套不进这套字母（`dequant_*` 那两件还被 `fetch_assets.py`
  按文件名起子进程调用，动它们要连清单一起动）。

## 四条链怎么选

| 链 | 一句话 | 权重合计 | 实测速度 | 验证到哪一步 |
|---|---|---|---|---|
| `faster_whisper/`（whisper） | **老、快、部署面最宽**：pip 装得上就能跑，VAD 和标点都在库里，一个 ct2 模型目录就是一条链。代价是 whisper 系会"顺句读"，中文长口语的信息保留度不如 AED | 3.09 GB | 本轮没有对这一条做计时（**未测**） | 5 件权重的出处全是 A 级；4 个小文件对生产机实算过 sha256 |
| `CrispASR-FireRed/`（FireRed） | **慢但准、占用资源也小**：AED 结构上没有 LLM 解码器，不可能把一段话概括成一句；权重 1.07 GB（AED 962 MB + Punc 108 MB），比 qwen3 那份 1.49 GB 还小。纯 CPU 上它比 qwen3 慢多少**要看是哪台机器**：笔记本那组是 1.6–1.9 倍（n=1、30 s 片段，C 级），而 10-08 在校机（24 vCPU、无显卡）同机复测，FireRed-AED 贪心档 wall RTF 0.398（自报 0.357）、默认 beam=3 档 0.547（自报 0.501），跟 qwen3 那趟的 0.321 是同一量级——"慢近一倍"这个比例在校机上不成立。另外这条链的切片上限已由 30 改成 20：AED 每片解码预算写死 150 token，30 s 的切片必然撞顶并静默丢整句（详见该目录 README §5.1 ③）。10-09 把主腿换成 ONNX 之后"150 token 硬顶"这一条不再适用（撞顶会把 truncated 标出来、不静默），`20` 只留作标点分段的粒度 | 4.61 GB（crispasr 那侧 11 件，含程序与 CUDA 运行库）；10-09 起默认主腿是同一批模型的 ONNX 还原件，它那三件权重按精度档另计（上 N 卡要 f32/mixed 档，件数与 sha 在 `onnx/fetch_assets.py`） | 笔记本纯 CPU：1.44–1.80（同机 qwen3 0.88–0.97）；校机 24 vCPU：贪心 0.357（wall 0.398）/ 默认 beam=3 0.501（wall 0.547）；**10-09 笔记本 N 卡（RTX 4060 Laptop）同一批三件整腿：ONNX 主腿 cuda 0.427、纯 CPU 0.566、crispasr 腿 1.105** | 程序 14 件与生产机逐字节相同；3 件小权重对生产机实算过 sha256 |
| `CrispASR-Qwen/` + `Qwen3/`（Qwen） | **把功能全合进一个模型**（Qwen3-ASR-1.7B）：自带大小写、标点、中英混排，所以链路上不用后加标点模型、前端预处理也少。**吃满显卡性能**就是它的定位：GPU 上 RTF ≈ 0.08。有一个洞：CrispASR 这份构建把模型【内置】的语种判别吃了（`--list-backends` 里 `CAP_LANGUAGE_DETECT` 位没置），所以要用 `-l auto` + whisper-tiny 前置判别器补回来 | 1.57 GB（q4_k 1.49 GB + silero 0.85 MB + ggml-tiny 74 MB） | GPU：RTF ≈ 0.08（产线日志 2,984 文件 / 33.4 h 忙时算出来的，逐组差 10 倍，别拿均值外推）。纯 CPU：RTF 0.59（`-t` 12，8C/16T） | **`Qwen3/` 那份已在纯 CPU 机器上端到端跑通**（一键复原 → `--check` 5/5 → 驱动产出带标点的中文）；GPU 那份只做哈希级对账 |
| `FireRed-ONNX/`（FireRed 的 ONNX 还原） | **为了还原原组件、并在 CPU-only 环境下做到最佳性能**：FireRedASR2S 官方那四件（VAD / ASR / Punc / LID）直接导出成 ONNX 用 onnxruntime 跑，不走 gguf 量化那层。校机（24 vCPU EPYC，无显卡）实测最快档 RTF 0.361，但那只是 ASR 那一环；同机整条链 0.584~0.724、产线批内 0.155。同一台机器上的参照：crispasr 跑 qwen3 那条命令 RTF 0.321（63.7 s 音频、18 线程，含它自己的 VAD/标点/LID）—— 0.361 是单环、0.321 是整条，口径不同不能直接比大小。10-08 已把 crispasr 的 FireRed-AED 档在同一台机器、同一个 80.2 s 文件上跑齐（两边都 12 线程、都是 VAD + AED + 标点的整链）：本链单件 23.85 s（RTF 0.297，另有 6.4 s 一次性装载摊在两条素材上），crispasr 贪心档自报 0.357、每条重装模型后 wall 31.9 s，默认 beam=3 自报 0.501、wall 43.9 s。把装载一起算，本链约 27 s，比贪心少 15%、比 beam=3 少 38%；但同日更早那轮"单件对单件、两边各含一次装载"是 29.2 s 对 29.4 s，**打平** —— 所以这个差距本质是批次摊薄（crispasr 每个文件重装一次模型、批量 `-f` 省不下来，见 `gpu` 分支 `CrispASR-FireRed/README.md` §5.1 ①③），不是单件更快。去标点正文 463 vs 464 / 463（基本同文）。所以选这条链的理由现在多一条：还原官方件 + 长跑批次时同机更省 | 见该目录 README | RTF 0.361（20 s 段 + mixed 精度档，校机实测）；10-08 同机同件整链 0.297（crispasr FireRed-AED 贪心档自报 0.357） | **已在纯 CPU 校机上端到端跑通**，17/17 sha 与产线逐件对账、输出逐字相同 |

三条 crispasr 链（`CrispASR-FireRed/` / `CrispASR-Qwen/` / `Qwen3/`）和 `FireRed-ONNX/`
**共用同一份 I/O 契约**：输入 `ASRSOURCE\p`、输出按一级子目录分组的 `<组名>.txt` 追加写
`title:<相对路径>` + 正文、成功 send2trash、失败保结构隔离进 `f\`、`.lock` 跨引擎互斥、
`STOP` 文件在批边界优雅退出。所以数据面可以整体搬到另一条链上接着跑。
**`faster_whisper/` 那份是唯一例外**：它单文件轮询、输出全进一个 `ear.txt`、`rules.txt`
格式也不同（要 `正则 = 替换`），搬数据面前先看它 README 第 2 节。

三条 crispasr 链共用同一个 `crispasr.exe`（`CrispASR-FireRed/` 那份从 10-09 起主腿改成了
ONNX，但语种闸门那两步和 qwen3 兜底那一趟仍然是 crispasr.exe），其中有三处
**它自己的文档与实现不符**
（`PERFORMANCE.md` 的 P0 说 FireRed 解码器"没有 KV cache"、session API 的
`beam_size > 1` 守卫让"要贪心"落回默认 beam 档，以及 AED 每片解码预算写死 150 token、
`--chunk-seconds` 因此不是速度参数而是完整性参数），复算用的 `文件:行` 都在
`gpu` 分支 `CrispASR-FireRed/README.md` §5.1。这三处都不需要跑 GPU 就能核。

## 两个最新的驱动，简要说明

**`gpu` 分支 `CrispASR-FireRed/Wlid-Og-fired-Cg-qwen.py`** —— 生产机现役的那份。10-09 起
出厂主腿是 ONNX（`PRIMARY_LEG = "onnx"`），旧的 crispasr 双引擎走法留成
`PRIMARY_LEG = "crispasr"`，既是权重/闸门缺席时的降级路径，也是同机对拍的参照。
它解决的问题是"一个引擎的语种覆盖面不够"：FireRedASR2-AED 的中文信息保留度比 qwen3 稳
（AED 没有 LLM 解码器，结构上不可能把一段话概括成一句），但它内置的语种判别只覆盖
中文（+约 20 种汉语方言）/ 英语 / 粤语，**范围外不是差一点，是拿汉字编造**，而且 `-l`
改不了它。所以判语种必须在调用转写之前独立做一次，由一个闸门负责：两步 crispasr 短调用
（固定 `--gpu-backend cpu`、`-t 2`，校机实测合计约 1.5 s/文件），先用 silero 拿真语音段，
再按段拼出**前 15 秒语音**喂 `ggml-tiny.bin -dl`。为什么拼语音前缀而不是直接判原始文件：
这批素材的头 15 秒常常是前奏音乐 —— 正对照（合成件
15 s 器乐 + 20 s 中文语音，标准答案 zh）原始前缀判成 `en 0.784`（错），语音前缀判成
`zh 0.966`（对）。窗口长度这里**更正一句**（这几行先前写的是"whisper 那条判别路径本来就只
截前 15 秒"）：那个 15 秒截断属于 crispasr 的**外部判别器**（`crispasr_lid.cpp:284`
`kLidMaxSamples = 16000*15`），也就是旧腿批内筛子读的那行 `crispasr: LID ->`；闸门读的
是 whisper 后端自己那行 `auto-detected language:`，实测吃**前 30 秒**。10-09 用切片把两条
钉在一起：同一件 66.9 s 素材 head15 → `km 0.410626`（旧腿批次日志给的正是 `km p=0.411`）、
head30 → `zh 0.215178` ⇒ 两条腿判得不一样，主要变量是窗口长度，不是"有没有拼语音"
（那件素材 silero 判 0 段，闸门退回判的就是原始文件）。数字与逐条见该目录 README §6.6。
判到范围内的进 **FireRedASR2S ONNX 全链**（FireRedVAD → AED → FireRedPunc，
三个会话在本进程常驻一次、逐文件跑）；判到范围外的**不落盘**、攒下来立刻在同一批里用
crispasr qwen3 兜底重跑。闸门自己没结论时（没抓到判别行 / crispasr 非零退出）**不扣文件**，
按主链跑并记一行日志 —— 宁可留着已知的编造风险，也不能因为闸门瞎了就把整棵树拖去兜底引擎。

ONNX 这条腿出厂 `ONNX_PROVIDER = "cuda"`（10-09 从 `cpu` 改的，理由就是本节那段整腿实测），
只有 f32 编码器交给
CUDA EP —— 解码器也上卡在同一台机器上实测是**负收益**（同批三件 +28%，三件逐件都慢，
显存反而放得下），所以 `ONNX_DEC_ON_GPU` 保持 `False`。DirectML 与 OpenVINO 在代码里
**根本不列**，因为那两个能把 AMD 核显挑中。**没有 CUDA EP 的机器现在会当场停在起批前那道
`enc_ep` 断言上**，要在那种机器上跑得显式 `FIREDASR_ONNX_PROVIDER=cpu`。10-09 在同一台
笔记本上跑齐了同一批三件素材的
整腿对照：ONNX 主腿 cuda RTF 0.427、纯 CPU 0.566、crispasr 腿 1.105（逐趟数字、util 与
显存判据在该目录 README §6.6）。

另外几件它替人做了的事：批内每 30 秒轮询已写出的 `.txt` 落盘一个清一个（以前要等整批跑完
才统一结算，中途被杀等于白跑）、主链切段用 FireRedVAD 而不是 crispasr 默认的
silero（v6 会把唱歌素材判 0 段、整条静默丢弃）—— 闸门那一步虽然用 silero，但判到 0 段时
是退回对原始文件直接 `-dl`，不因此扣文件；无标点文本补标点在 ONNX 腿里由 FireRedPunc
自己承担；被判无语音的文件写进 `no_speech.txt` 记账
（rc 仍是 0、不落 `.txt`，转写树上什么都不留，只能另外记）。走 `PRIMARY_LEG = "crispasr"`
时仍是老形态：一次 crispasr 调用喂多个 `-f`（模型/VAD 只加载一次），筛子那套
`_lid_map` 从 stderr 逐文件对号。

**`cpu` 分支 `FireRed-ONNX/Oc-fired.py`** —— 无显卡机器上的那条链，四个环节
（FireRedVAD → FireRedASR2-AED → FireRedPunc → 可选 FireRedLID）**全部纯 onnxruntime**，
不需要 CUDA、不需要 gguf 量化那层。它和上面那份驱动 I/O 逐条相同，两套可互换；
`--check` 那套资源复原也是同一个形态（`assets.json` + `fetch_assets.py` + `setup.py`）。
建造侧多做了一件必须的事：官方权重是 fp32/fp16 的 ONNX，要在 CPU 上有可用吞吐必须
按档位反量化/降精度，实测最快的档是"20 s 段 + mixed"，RTF 0.361（校机 24 vCPU）。
它的已知洞和两处公开更正都写在该目录 README 第 9 节那张"已核/未核"表里。

## 复原的三条命令（每个目录都一样）

```bat
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python fetch_assets.py          :: 下载 + 逐件 sha256 验收
```

`fetch_assets.py` 只用标准库，跨平台，支持断点续传（`.part` + HTTP Range），
`--check` 离线核对现状、`--list` 打印全部 URL 与哈希、`--only NAME` 只补某一件、
`--set-root NAME=PATH` 换落点、`--proxy` 走代理。退出码 3 = 有组件没就位。
复原完之后**不用改驱动一行就能跑**：所有路径都是"环境变量优先，没给就落在本包目录下"，
要换位置改环境变量就行（见下面那一节）。

## 出处等级口径

清单里每一行都带 `grade`：

- **A** = 发布方给的哈希（GitHub release 的 digest、HuggingFace 的 LFS `oid`）与作者生产机
  那一份**逐字节相同** —— 照链接下载就是同一个文件。
- **B** = 哈希只取自生产机那一份，没有独立凭据可对照。
- **C** = 没有公开匿名下载件，必须人工补（本仓两处：微软 VC++ 运行库 4 个 dll、
  NVIDIA cuDNN9/cuBLAS12）。`--check` 对这种行只给 WARN，不算失败。

## 隐私与凭据

本仓是 **public**。里面**没有任何端点、token、口令，也没有任何绝对路径**（全树零个
盘符，用户名 / 内网 IP / 具体机器名一个都没有）：

- 收尾通知（ntfy）改成读环境变量：`NTFY_TOPIC_URL` 设了才发，不设就跳过，驱动里没有
  硬编码的 topic。
- 出现过的 `127.0.0.1:7890` 与 `127.0.0.1:7897` 全是 `fetch_assets.py --proxy` 的**写法示例**
  （回环地址＋mihomo/clash 常见的 mixed-port 默认值）。回环只有本机可达，指向不到任何远程
  服务；本仓不含远程代理地址、订阅链接或凭据。
- 路径的口径全仓统一 = **环境变量优先，没给就落在本包目录下**（相对名按脚本所在目录
  解析），所以整包克隆下来放哪儿都能跑。四个旋钮：`CRISPASR_BIN_DIR`（crispasr 程序
  目录）、`CRISPASR_MODEL_DIR`（权重目录）、`ASR_TEXT_DIR`（转写与记账输出）、
  `ASR_AUDIO_DIR`（待转音频库）；`faster_whisper/` 的模型目录用 `FASTER_WHISPER_MODEL_DIR`，
  `FireRed-ONNX/` 那条用它的 `--data-root` / `--models-dir` 等命令行参数（等价的环境变量
  前缀是 `FIREDASR_`）。每个包自己的 README 里有一张表逐条写着这些旋钮的默认值。
- 作者生产机上跑的那份原脚本仍按那台机器自己的绝对路径，那是产线实况，不在本仓里。

## 许可

引擎与权重各自的许可证在 `assets.json` 的 `license` 字段里逐件写着：CrispASR 是 MIT，
FireRedASR2 / Qwen3-ASR 是 Apache-2.0，FireRedPunc / FireRedVAD 是 BSD-2-Clause，
whisper.cpp 是 MIT，NVIDIA 那三件 dll 是 CUDA Toolkit EULA，cuDNN 是 NVIDIA 自己的 SLA。
faster-whisper large-v3 的模型权重是 MIT（OpenAI Whisper 上游也是 MIT）。
