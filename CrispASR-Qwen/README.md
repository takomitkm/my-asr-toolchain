# CrispASR-Qwen —— CrispASR / Qwen3-ASR 的 GPU 批转链

驱动是 `crispasr-Qwen.py`，引擎是 CrispASR 的 `qwen3` backend（Whisper 式音频编码器 +
Qwen3 1.7B 解码器），跑在 CUDA 上。**一个模型直接出成品文本**：自带大小写、标点、
中英混排，所以链路上没有独立的标点模型，也不做前端预处理。

和 `cpu` 分支 `Qwen3/` 的关系：同一个驱动的两个构建。那份是 CPU 版
（`crispasr-Qwen-cpu.py` + `crispasr-windows-x86_64-cpu` 那套件），这份是 CUDA 版；
两边的 I/O 契约逐条相同，数据面可以互换。仓库根 README 有四引擎对比表。

CrispASR 这份 CUDA 构建把模型【内置】的语种判别吃掉了（`--list-backends` 的
`CAP_LANGUAGE_DETECT` 位根本没置），所以这套仍然是 `-l auto` + whisper-tiny 前置判别器补回来，
细节见第 5 节。

---

## 1. 三步复原

```bat
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python fetch_assets.py          :: 下载 + 逐件 sha256 验收
```

`fetch_assets.py` 是通用的，读同目录 `assets.json`（机器可读的资源清单：URL、字节数、
sha256、出处等级、许可证都在里面）。落点默认在本目录下两个文件夹：

| 根 | 默认 | env | 装什么 |
|---|---|---|---|
| `bin` | `./crispasr` | `CRISPASR_BIN_DIR` | `crispasr.exe` + 8 个 ggml/crispasr dll + CUDA 运行库 3 件（+ 4 个 MSVC 运行库 dll，需人工，见第 3 节） |
| `model` | `./model` | `CRISPASR_MODEL_DIR` | `qwen3-asr-1.7b-q4_k.gguf` / `ggml-silero-v6.2.0.bin` / `ggml-tiny.bin` |

想落到别处：设清单里写的同名环境变量（`CRISPASR_BIN_DIR` / `CRISPASR_MODEL_DIR`），下载与驱动读同一个值；
只想改下载落点就用 `--set-root model=<别的目录>`（相对按包目录算、绝对照用，驱动那边要用同一个值）。
核对现状不联网用 `--check`，看全部 URL 与哈希用 `--list`，只补某一件用 `--only NAME`。

上面那两个 env 就是驱动 CONFIG 里 `CRISPASR_BIN_DIR` / `MODEL_DIR` 读的两个；数据面另有
`ASR_TEXT_DIR` / `ASR_AUDIO_DIR`。四个都不设时全部落在包目录下，所以**复原完不改一行就能跑**：

| env | 默认（相对包目录） | 是什么 |
|---|---|---|
| `CRISPASR_BIN_DIR` | `./crispasr` | 上面 `bin` 那个根 |
| `CRISPASR_MODEL_DIR` | `./model` | 上面 `model` 那个根 |
| `ASR_TEXT_DIR` | `./txt` | 转写文本与日志的输出目录（`rules.txt`、`tmplist.txt`、`log\` 都在这） |
| `ASR_AUDIO_DIR` | `./audio` | 音频数据面根目录，驱动用它的 `p\` 当输入队列、`f\` 当失败隔离区、`t\` 当 ASCII 暂存区 |

值给相对路径就按包目录解析，给绝对路径就照用；驱动与 `fetch_assets.py` 读同一套 env，
所以只要 env 设了，下载落点和运行读点不会分家。
GPU 档的三个开关在驱动 `CONFIG · CrispASR` 一节：`CRISPASR_BACKEND = "qwen3"`、
`CRISPASR_GPU_BACKEND = "cuda"`、`CRISPASR_THREADS = 6`（CPU 侧解码线程，混合后端时才起作用），
一般不用改。`ASR_ROOT` / `ASRSOURCE` 必须在同一块盘上（`t\` 用硬链接，跨盘会失败）。

## 2. 跑

```bat
.venv\Scripts\python crispasr-Qwen.py --start   :: 分离后台启动
.venv\Scripts\python crispasr-Qwen.py --stop     :: 写 STOP 标志，下一批边界干净退出
.venv\Scripts\python crispasr-Qwen.py            :: 前台跑（Ctrl+C 一次=本批跑完退，两次=立刻杀子进程）
```

**关掉启动它的那个控制台 = 连坐杀 crispasr 子进程**（Job Object 是故意绑上去的，
用来防孤儿；实测在 Windows Terminal 下会整链一起走）。要停就用 `--stop`，别关窗。

I/O 契约（与 `Qwen3/`、`../FireRed-ONNX/` 那两份逐条相同）：

- 递归扫 `ASRSOURCE\p` 下的音频（`.mp3 .m4a .mp4 .wav .oga .ogg .opus .flac .aac`），
  按**一级子目录分组**，组内按 mtime 升序；根上的散文件归 `p` 组。
- 输出 `ASR_ROOT\<组名>.txt`，**追加**写：`title:<相对 p 的路径>` + 正文 + 空行。
- 成功 send2trash；空转写即删且不落记录；失败按相对路径隔离进 `f\`（保目录结构）。
- 一次调用喂多个 `-f`（`BATCH_SIZE = 28`），模型只加载一次；批内每 30 秒轮询
  已写出的 `.txt`，落盘一个清一个，中途被杀不白跑。
- `.lock` 单实例锁，**与 `ASR_ROOT` 相同的其它 crispasr 驱动互斥**。
- `rules.txt`（同目录有 `rules.example.txt`）逐条容错，单条非法正则只跳过该行；
  `tmplist.txt` 是幻觉复读的黑名单。
- 收尾推送可选：`set NTFY_TOPIC_URL=https://…` 才发，不设就跳过（端点不入库）；
  Ctrl+N 在开/关之间切换，不必重启。

## 3. 组件清单与出处等级

`assets.json` 是机器可读的那份（`python fetch_assets.py --list` 打印全表）。摘要：

| 件 | 来源 | 等级 | 说明 |
|---|---|---|---|
| `crispasr.exe` + `crispasr.dll` + `crispasr-quantize.exe` + `ggml.dll` + `ggml-base.dll` + `ggml-cpu.dll` + `ggml-cuda.dll` + `whisper.dll` + LICENSE + `README-CUDA.txt` + THIRD_PARTY_NOTICES（zip 的 11 个成员） | GitHub release `v0.8.37/crispasr-windows-x86_64-cuda-non-cuda.zip`（146,153,022 B / `74d04c30…`） | **A** | zip 的 sha256 是 release 给的 digest；11 个成员各自的 sha256 从 zip 里算，且与作者生产机那份逐字节相同 |
| `cublas64_12.dll` / `cublasLt64_12.dll` / `cudart64_12.dll` | 同一个 release 页的独立资源（113,712,640 / 692,441,600 / 573,952 B） | **A** | 哈希取自 release 附带的 `crispasr-windows-x86_64-cuda-runtime-sha256.txt`（251 B / `92e42561…`），与生产机那份逐字节相同 |
| `msvcp140/vcomp140/vcruntime140/vcruntime140_1.dll` | 微软 `vc_redist.x64.exe`（aka.ms 链接） | **C** | CPU/CUDA 构建都动态链 MSVC 运行库。装了 redist 就不用管；不能装就拷这 4 份到 exe 同目录。`--check` 只给 WARN |
| `qwen3-asr-1.7b-q4_k.gguf` 1,490,915,200 B | `huggingface.co/cstr/qwen3-asr-1.7b-GGUF` | **A** | sha256 `ec197cef…` = HF 的 LFS oid，也就是 CrispASR 自己 `-m auto` 注册表 `qwen3-1.7b` 那一行 |
| `ggml-silero-v6.2.0.bin` 885,098 B | `huggingface.co/ggml-org/whisper-vad` | **A** | crispasr 自己的默认 VAD（`examples/cli/crispasr_vad_cli.cpp:19-20`）。换 v5.1.2 要同时改 path 和 sha256（同字节数、不同内容） |
| `ggml-tiny.bin` 77,691,713 B | `huggingface.co/ggerganov/whisper.cpp` | **A** | `-l auto` 的前置语种判别器。必须给成本地文件，否则 crispasr 会联网下、离线机器上卡满超时 |

等级口径：**A** = 发布方给的哈希与作者生产机那份逐字节相同，照链接下就是同一个文件；
**B** = 哈希只取自生产机那份、没有独立凭据可对照；**C** = 没有公开匿名下载件，需人工。
自动下载合计 2,522,373,225 B（`--list` 末尾那个数）。

### 3.1 上传前对"必须组件是否完善"做了什么

这台机器和校机都跑不了 CUDA（校机无独显、无 `nvidia-smi`），所以这一轮不是端到端实测，
而是三条静态/哈希级对账，全部通过：

1. **引用闭包**：把驱动里 `os.path.join(CRISPASR_BIN_DIR/…, MODEL_DIR, "…")` 拼出来的
   运行时引用逐条抽出来（bin 侧 1 件 `crispasr.exe`、model 侧 3 件权重），核对清单能不能
   把它们全部落地 —— 没有缺口。驱动会自己创建的目录/文件（`log\`、`.lock`、`STOP`、
   `t\`、`f\`）和要人手写的 `rules.txt`/`tmplist.txt` 单独归类，不算组件。
2. **清单 ↔ 生产机目录**：`bin` 根 14 件、`model` 根相关件的**字节数逐件相符**；
   清单能落 18 件里多出的 4 件正是 MSVC dll（生产机装了系统级 redist，exe 同目录没有它们，
   这是符合预期的，不是缺件）。
3. **真算 sha256 的那几件**：拿 `fetch_assets.py --check --only …` 直接对生产机目录算过
   本文件夹的 `ggml-silero-v6.2.0.bin`（864.35 KB）与 `ggml-tiny.bin`（74.09 MB）——
   都报"sha256 对上"。大权重（1.49 GB 那一件）本轮没有重算哈希，依据是发布方凭据
   （HF 的 LFS oid）+ 字节数相符，也就是上表的 A 级口径本身。
   同一批里另外几件（`firered-vad.gguf` 2.25 MB、`fireredpunc-q8_0.gguf` 103.63 MB、
   faster-whisper 的 4 个 json）也真算过，但那些属于隔壁 `CrispASR-FireRed/`
   和 `faster_whisper/` 两个文件夹。

**没有做过的**：这一份 CUDA 驱动在 GPU 上的端到端跑通。同一支 `crispasr.exe` 的 CPU 路径
已在纯 CPU 机器上跑通（`cpu` 分支 `Qwen3/` 第 3.1 节，含 rc=127→rc=0 那个 MSVC 门槛），
CUDA 侧的行为差异只在 `CRISPASR_GPU_BACKEND` 这一个开关和 `ggml-cuda.dll` + 三件 cuda dll。

## 4. 关键 CONFIG 值（为什么是这个值）

| 常量 | 值 | 为什么 |
|---|---|---|
| `CRISPASR_MODEL` | `qwen3-asr-1.7b-q4_k.gguf` | 显存吃紧就换 q8_0（2,506,723,200 B，同仓库另一档）。实测过：**q8 反而慢 25%**，所以默认停在 q4_k |
| `CRISPASR_VAD_MODEL` | `ggml-silero-v6.2.0.bin` | crispasr 自己的默认。按作者的决定【不换】。已知洞见第 5 节 |
| `CRISPASR_LANGUAGE` | `"auto"` | 改成 `"zh"` 时 CONFIG 里 `CRISPASR_LID_MODEL` 那个前置判别器就不参与拼命令了，`ggml-tiny.bin` 可以不下 |
| `BATCH_SIZE` | `28` | 一次调用喂多个 `-f`，模型只加载一次；显存不够就调小 |

实测过的数（GPU，4060）：单文件 40 s 量级的中文素材 **RTF ≈ 0.08**。多实例只 +33%，
批量 `-f` 之外再堆并行没有意义。

## 5. 已知洞与边界

- **"CrispASR 把内置 LID 吃了"**：qwen3-ASR 模型自带语种判别，但 CrispASR 的 CUDA 构建
  在 `--list-backends` 里没有置 `CAP_LANGUAGE_DETECT` 位，`-l` 传的语种码被拼成一条
  assistant 前缀而不是真判别。所以这套用 `-l auto` + `--lid-backend whisper
  --lid-model ggml-tiny.bin` 前置补一层（只听前 15 秒、不出字）。两条路实测各有已证的坏法：
  写死 `"zh"` 在整段非中文文件上复读崩盘；判别错码会把内容摘要化丢掉。逐条依据在驱动的
  CONFIG · 语种一节。
- **silero v6.2.0 对唱歌素材判 0 段 → 整条静默丢弃**（rc 仍为 0、不落 `.txt`）。
  口语素材上 v5/v6 打平（同一份 89.84 s 中文：切 5 段/75.78 s 对 7 段/75.88 s，墙钟
  55.9 对 56.1 s），但"口语打平"不等于"全语料打平"。**而且这份驱动不写 `no_speech.txt`**
  ——记账逻辑在 `xhs-asr.py` 和 CPU 版那份里，这份不改（驱动 CONFIG 一节末尾那条注释就写着这件事）。
  要吞了多少的账，用 `CrispASR-FireRed/` 那份驱动跑，或换 `firered-vad.gguf`。
- `--stop` / STOP 文件是**批边界**退出，一批最长可能等 `BATCH_SIZE` × 单文件时长。
- MSVC 4 件和（GPU 档才需要的）CUDA 运行库是两处人工门槛；前者 rc=127 已实测，
  后者缺了会退化成 CPU 后端或直接报错。
