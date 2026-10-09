# CrispASR-FireRed —— FireRedASR2S 全链路（ONNX，可上 N 卡）+ qwen3 兜底的批转链

驱动是 `crisper-xhs-qwen-asr.py`（10-08 前的旧名是 `xhs-asr.py`，两份是同一个文件；名字里的
qwen 指兜底那台引擎，主力一直是 FireRed 那套）。

**10-09 起这条链有两段实现，由驱动顶部一个常量挑：**

| `PRIMARY_LEG` | 主力那一路（"小红书 ASR"部分）由谁跑 | 语种怎么定 |
|---|---|---|
| `"onnx"`（默认） | 本包 `onnx/` 里的 **FireRedASR2S ONNX 全链路**：FireRedVAD → FireRedASR2-AED（纯 onnxruntime）→ FireRedPunc，三个会话在驱动进程里常驻一次 | **闸门**：crispasr 的默认 silero VAD 导语音段 → 拼前 15 s 语音 → `whisper-tiny -dl` 判语种 |
| `"crispasr"` | CrispASR 的 `firered-asr` backend（同一模型 AED 的 gguf q4_k 量化件，CUDA） | crispasr 自己的 `--lid-backend whisper` 筛子（判到范围外照样换 qwen3） |

两档共用同一份兜底：判到 AED 语种范围外 → 这批文件立刻用 `qwen3` 重跑（同一批内完成）。
换档不改任何其它参数，那 8 行 crispasr 代码一行没删。

形状没变的部分：AED 出**无标点**的字、标点由 `fireredpunc` 加（`"onnx"` 档是 ONNX 图的
FireRedPunc、`"crispasr"` 档是 gguf 那份），VAD 用 `firered-vad`（`"onnx"` 档用它的 ONNX 图）。
四件事各有各的文件，所以这一份是三个 CrispASR 方案里组件最多的一套
（crispasr 侧自动下载合计 4,612,011,305 B，`"onnx"` 档还要再加 `onnx/` 那三件权重，见第 1 节）。

---

## 1. 复原

```bat
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python fetch_assets.py          :: crispasr 侧：下载 + 逐件 sha256 验收
.venv\Scripts\python onnx\fetch_assets.py --models onnx\models --graph mixed   :: ONNX 侧（PRIMARY_LEG="onnx" 才要）
```

`fetch_assets.py` 是通用的，读同目录 `assets.json`（URL、字节数、sha256、出处等级、许可证
都在里面）。落点默认在本目录下两个文件夹：

| 根 | 默认 | env | 装什么 |
|---|---|---|---|
| `bin` | `./crispasr` | `CRISPASR_BIN_DIR` | `crispasr.exe` + 7 个 dll + `crispasr-quantize.exe` + 文档 3 件 + CUDA 运行库 3 件（+ 4 个 MSVC 运行库 dll，需人工，见第 3 节） |
| `model` | `./model` | `CRISPASR_MODEL_DIR` | AED q4_k / FireRedPunc q8_0 / FireRedVAD / ggml-tiny / silero-v6.2.0（闸门）/ qwen3 q8_0 共 6 份权重 |

**ONNX 那三件权重不在 `assets.json` 里**，由 `onnx/fetch_assets.py` 自己那份清单管
（`python onnx/fetch_assets.py --list` 打印全表）。分两份是因为两侧的建造方式不同：
crispasr 侧全部是"下载即终件"，ONNX 侧有两件（`punc.f32.onnx`、`encoder.f32.onnx`）
是本地 `dequant_*.py` 从公开原料造出来的，逐字节可复现（10-06 从 int8/q8w 原料重跑，
五件产物与产线文件同 sha256）。`--graph` 决定造到哪一步，必须与驱动里的 `ONNX_GRAPH`
一致：默认 `mixed` = f32 编码器 + int8 解码器 = 建 `punc_f32` + `aed_encoder_f32`。
落盘约 5.5 GB，其中 `encoder.f32.onnx` 只有 934 KB、权重在旁边的 `.onnx.data` 里。
这一条不进库（`onnx/.gitignore` 的 `models/*`），克隆后必须跑这一步。

想落到别处：设清单里写的同名环境变量（`CRISPASR_BIN_DIR` / `CRISPASR_MODEL_DIR`），下载与驱动读同一个值；
只想改下载落点就用 `--set-root model=<别的目录>`（相对按包目录算、绝对照用，驱动那边要用同一个值）。
核对现状不联网用 `--check`，看全部 URL 与哈希用 `--list`，只补某一件用 `--only NAME`。

上面那两个 env 就是驱动 CONFIG 里 `CRISPASR_BIN_DIR` / `MODEL_DIR` 读的两个；数据面另有
`ASR_TEXT_DIR` / `ASR_AUDIO_DIR`；`"onnx"` 档另有下表四个。都不设时全部落在包目录下，
所以**复原完不改一行就能跑**：

| env | 默认（相对包目录） | 是什么 |
|---|---|---|
| `CRISPASR_BIN_DIR` | `./crispasr` | 上面 `bin` 那个根 |
| `CRISPASR_MODEL_DIR` | `./model` | 上面 `model` 那个根 |
| `ASR_TEXT_DIR` | `./txt` | 转写文本与日志的输出目录（`rules.txt`、`tmplist.txt`、`no_speech.txt`、`log\` 都在这） |
| `ASR_AUDIO_DIR` | `./audio` | 音频数据面根目录，驱动用它的 `p\` 当输入队列、`f\` 当失败隔离区、`t\` 当 ASCII 暂存区 |
| `ASR_PRIMARY_LEG` | `onnx` | 主力那一路由谁跑：`onnx` / `crispasr`（见第 6 节） |
| `FIREDASR_ONNX_MODELS_DIR` | `./onnx/models` | ONNX 三件权重的根目录 |
| `FIREDASR_ONNX_PROVIDER` | `cpu` | ONNX 的执行提供者：`cpu` / `cuda`。**只有这两个值**，见第 6 节那条"只用 N 卡" |
| `FIREDASR_ONNX_GRAPH` | `mixed` | AED 图形状，必须与建权重时 `--graph` 给的那个一致 |

六份 crispasr 侧权重的文件名分别写在 `CRISPASR_MODEL_AED` / `CRISPASR_MODEL_QWEN3` /
`CRISPASR_PUNC_MODEL` / `CRISPASR_LID_MODEL` / `CRISPASR_VAD_MODEL` / `GATE_VAD_MODEL`
（见第 4 节），拼的都是 `MODEL_DIR` 下那几个名字，与 `fetch_assets.py` 的落点一一对得上。
值给相对路径就按包目录解析，给绝对路径就照用。`ASR_ROOT` / `ASRSOURCE` 必须在同一块盘上
（`t\` 用硬链接，跨盘会失败）。

## 2. 跑

```bat
.venv\Scripts\python crisper-xhs-qwen-asr.py --start   :: 分离后台启动
.venv\Scripts\python crisper-xhs-qwen-asr.py --stop     :: 写 STOP 标志，下一批边界干净退出
.venv\Scripts\python crisper-xhs-qwen-asr.py            :: 前台跑（Ctrl+C 一次=本批跑完退，两次=立刻杀子进程）
```

**关掉启动它的那个控制台 = 连坐杀 crispasr 子进程**（Job Object 是故意绑上去的，
用来防孤儿）。要停就用 `--stop`，别关窗。

I/O 契约（与 `CrispASR-Qwen/`、`cpu` 分支那两份逐条相同）：

- 递归扫 `ASRSOURCE\p` 下的音频（`.mp3 .m4a .mp4 .wav .oga .ogg .opus .flac .aac`），
  按**一级子目录分组**，组内按 mtime 升序；根上的散文件归 `p` 组。
- 输出 `ASR_ROOT\<组名>.txt`，**追加**写：`title:<相对 p 的路径>` + 正文 + 空行。
- 成功 send2trash；空转写即删且不落记录；失败按相对路径隔离进 `f\`（保目录结构）。
- `BATCH_SIZE = 28`；批内每 30 秒轮询已写出的 `.txt`，落盘一个清一个。
- 语种筛子判到 AED 范围外的文件**不落盘**，攒下来立刻用 qwen3 兜底重跑（同一批内完成）。
- VAD 判"整段无语音"的文件 rc 仍是 0、不落 `.txt`，转写树上什么都不留，
  所以在这里记账：`ASR_ROOT\no_speech.txt`（`NO_SPEECH_LOG`），格式是 `时间 \t 原因 \t 相对路径`。
- `.lock` 单实例锁，**与同一 `ASR_ROOT` 下的其它 crispasr 驱动互斥**（同一份数据面
  不能两个引擎同时吃）。
- `rules.txt`（同目录有 `rules.example.txt`）逐条容错，单条非法正则只跳过该行；
  `tmplist.txt` 是幻觉复读的黑名单。
- 收尾推送可选：`set NTFY_TOPIC_URL=https://…` 才发，不设就跳过（端点不入库）；
  Ctrl+N 在开/关之间切换，不必重启。

## 3. 组件清单与出处等级

`assets.json` 是机器可读的那份（`python fetch_assets.py --list` 打印全表）。摘要：

| 件 | 来源 | 等级 | 说明 |
|---|---|---|---|
| `crispasr.exe` + `crispasr.dll` + `crispasr-quantize.exe` + `ggml.dll` + `ggml-base.dll` + `ggml-cpu.dll` + `ggml-cuda.dll` + `whisper.dll` + LICENSE + `README-CUDA.txt` + THIRD_PARTY_NOTICES（zip 的 11 个成员） | GitHub release `v0.8.37/crispasr-windows-x86_64-cuda-non-cuda.zip`（146,153,022 B / `74d04c30…`） | **A** | zip 的 sha256 是 release 给的 digest；11 个成员各自的 sha256 从 zip 里算，且与作者生产机那份逐字节相同 |
| `cublas64_12.dll` / `cublasLt64_12.dll` / `cudart64_12.dll` | 同一个 release 页的独立资源（113,712,640 / 692,441,600 / 573,952 B） | **A** | 哈希取自 release 附带的 `crispasr-windows-x86_64-cuda-runtime-sha256.txt`（251 B / `92e42561…`） |
| `msvcp140/vcomp140/vcruntime140/vcruntime140_1.dll` | 微软 `vc_redist.x64.exe`（aka.ms 链接） | **C** | 装了 redist 就不用管；不能装就拷这 4 份到 exe 同目录。`--check` 只给 WARN |
| `firered-asr2-aed-q4_k.gguf` 962,807,328 B | `huggingface.co/cstr/firered-asr2-aed-GGUF` | **A** | sha256 = HF 的 LFS oid；也就是 CrispASR `-m auto` 注册表 `firered-asr` 那一行。上游 FireRedASR2 是 Apache-2.0 |
| `fireredpunc-q8_0.gguf` 108,664,800 B | `huggingface.co/cstr/fireredpunc-GGUF` | **A** | 书面标点后处理，**只给 AED 用**。必须给本地路径：留空时 crispasr 取 `auto` 并联网下 FireRedPunc（`crispasr_run.cpp:3940`），拉不动的机器每批白等一次超时。上游 BSD-2-Clause |
| `firered-vad.gguf` 2,357,952 B | `huggingface.co/cstr/firered-vad-GGUF` | **A** | **转写侧**的 VAD（`_base_cmd` 里 `-vm` 那一行，qwen3 兜底与容器退回都走它）。**转写不用 crispasr 默认的 silero v6.2.0**，理由见第 5 节；silero v6 在 `"onnx"` 档里只出现在闸门（见下面那行）。上游 BSD-2-Clause |
| `ggml-tiny.bin` 77,691,713 B | `huggingface.co/ggerganov/whisper.cpp` | **A** | 语种判别的模型：`"onnx"` 档是闸门的 `-dl`，`"crispasr"` 档是 `--lid-backend whisper`。必须本地文件，否则联网下、离线机器卡满超时 |
| `ggml-silero-v6.2.0.bin` 885,098 B | `huggingface.co/ggml-org/whisper-vad` | **A** | **闸门**第 1 步的 VAD，也就是 crispasr `--vad` 的默认那个（`crispasr_vad_cli.cpp:19` 的 URL 就是这条）。只用于"判语种前导个段"，不参与转写。sha256 两头对上：HF 的 LFS oid 与纯 CPU 测试机上实跑那份相同（10-09 `sha256sum` 直读） |
| `qwen3-asr-1.7b-q8_0.gguf` 2,506,723,200 B | `huggingface.co/cstr/qwen3-asr-1.7b-GGUF` | **A** | 兜底引擎权重。显存不够换同仓 q4_k（1,490,915,200 B / `ec197cef…`），换时 path 与 sha256 一起改 |
| `onnx/models/**`（AED 图 / FireRedVAD / FireRedPunc，约 5.5 GB） | `onnx/fetch_assets.py` 自己那份清单：sherpa-onnx 的 GitHub release + 两个 HuggingFace 仓，另有两件本地 `dequant_*` 建造 | **A**（下载件）/**A**（建造件按 sha256 逐字节复现） | `"onnx"` 档才需要。与 crispasr 侧那份清单**没有重叠**，所以不存在两处对齐问题；`--list` 打全表、`--check` 不联网核对现状 |

等级口径：**A** = 发布方给的哈希与作者生产机那份逐字节相同，照链接下就是同一个文件；
**B** = 哈希只取自生产机那份、没有独立凭据可对照；**C** = 没有公开匿名下载件，需人工。

### 3.1 上传前对"必须组件是否完善"做了什么

这台机器和校机都跑不了 CUDA（校机无独显、无 `nvidia-smi`），所以这一轮不是端到端实测，
而是三条静态/哈希级对账，全部通过：

1. **引用闭包**：驱动里 `os.path.join(CRISPASR_BIN_DIR/…, MODEL_DIR, "…")` 拼出来的运行时
   引用逐条抽出来（bin 侧 1 件 `crispasr.exe`、model 侧 5 件权重，与 155/158/160/162/167
   行一一对上），核对清单能否全部落地 —— 没有缺口。驱动自己创建的
   （`log\`、`.lock`、`STOP`、`t\`、`f\`、`no_speech.txt`）和要人手写的
   `rules.txt`/`tmplist.txt`/`<组名>.txt` 单独归类，不算组件。
2. **清单 ↔ 生产机目录**：`bin` 根 14 件、`model` 根 5 件的**字节数逐件相符**；
   清单能落 18 件里多出的 4 件正是 MSVC dll（生产机装了系统级 redist，exe 同目录没有它们，
   符合预期）。生产机 `crispasr-model` 目录另有 9 件，比这份清单多的 4 件是另外两条链用的
   （silero 两份、q4_k 那份、canary 对齐器），本链不引用。
3. **真算 sha256 的那几件**：拿 `fetch_assets.py --check --only …` 直接对生产机目录算过
   `firered-vad.gguf`（2.25 MB）、`fireredpunc-q8_0.gguf`（103.63 MB）、`ggml-tiny.bin`
   （74.09 MB）—— 都报"sha256 对上"。AED 962 MB 与 qwen3 q8_0 2.5 GB 本轮没有重算哈希，
   依据是发布方凭据（HF 的 LFS oid）+ 字节数相符，也就是上表的 A 级口径本身。

**没有做过的**：这一份 CUDA 驱动在 GPU 上的端到端跑通。同一支 `crispasr.exe` 的 CPU 路径
已在纯 CPU 机器上跑通（`cpu` 分支 `Qwen3/` 第 3.1 节，含 rc=127→rc=0 那个 MSVC 门槛）。

## 4. 关键 CONFIG 值（为什么是这个值）

| 常量 | 值 | 为什么 |
|---|---|---|
| `ENGINES["aed"]` | backend `firered-asr` + `CRISPASR_MODEL_AED` + `CRISPASR_PUNC_MODEL` | 主引擎。AED 的中文信息保留度比 qwen3 稳（没有 LLM 解码器，结构上不可能把一段话概括成一句） |
| `ENGINES["qwen3"]` | backend `qwen3` + `CRISPASR_MODEL_QWEN3`，`punc` 留空 | 兜底档，只接筛子判到 AED 范围外的文件；qwen3 自己出标点，所以 `punc` 必须是空串 |
| `CRISPASR_GPU_BACKEND` | `"cuda"` | 改 `"cpu"` 就退回 CPU 构建那套（程序目录要一起换成 cpu zip） |
| `CRISPASR_THREADS` | `6` | CPU 侧解码线程。实测口径：**纯 CPU 时取 0.75 × 逻辑线程**；GPU 档这个值影响小 |
| `CRISPASR_LANGUAGE` | `"auto"` | AED 内置语种判别只覆盖 中文(+约20种汉语方言)/英语/粤语，范围外**不是差一点，是拿汉字编造**，而且 `-l` 改不了它 |
| `CRISPASR_LID_BACKEND` | `"whisper"` | 前置筛子；配合 CONFIG 里 `CRISPASR_LID_MODEL` 那个本地 `ggml-tiny.bin`。每文件判一次。`"onnx"` 档不用它拼转写命令，但 `PRIMARY_LEG="crispasr"` 时仍是主力那一路的筛子 |
| `BATCH_SIZE` | `28` | 一次调用喂多个 `-f`，模型/VAD 只加载一次 |
| `PRIMARY_LEG` | `"onnx"` | 主力那一路由谁跑（第 6 节）。改 `"crispasr"` 就回到 10-09 之前的形状 |
| `ONNX_PROVIDER` | `"cpu"` | ONNX 的执行提供者，**只认 `cpu` / `cuda`**。写别的（DirectML、OpenVINO）驱动直接退出 —— 那两个能挑中 AMD 核显 |
| `ONNX_GRAPH` | `"mixed"` | f32 编码器 + int8 解码器。10-06 产线实测最快的那档，也是 `cuda` 下唯一有意义的选择（int8 图里那些整型算子 CUDA EP 跑不了） |
| `ONNX_DEC_ON_GPU` | `False` | 编码器上卡、解码器留 CPU。10-09 在 N 卡上实测：解码器也上卡**更慢**（整批 418.9 s → 537.2 s，+28%），显存倒是放得下（峰值 4,788 MiB / 8,188 MiB）。机制与 crispasr 那边同源结论见 §5.1，数值见 §6.6 |
| `ONNX_ASR_THREADS` / `ONNX_VAD_THREADS` / `ONNX_PUNC_THREADS` | 逻辑核一半 / ≤8 / 4 | 与 `cpu` 分支那个分发包同口径（10-06 实测并回去的） |
| `GATE_VAD_MODEL` | `model/ggml-silero-v6.2.0.bin` | 闸门第 1 步 = crispasr `--vad` 的**默认**那个模型（本地路径，不让它联网下） |
| `GATE_LID_SEC` / `GATE_MIN_SPEECH_SEC` | 15.0 / 1.0 | 判别用 15 s 语音（whisper 那条路径本来就只截 15 s）；拼不出 1 s 语音就退回判原始文件 |
| `GATE_THREADS` / `GATE_TIMEOUT_SEC` | 2 / 300 | 闸门两步各是一次短调用，固定 `--gpu-backend cpu`（独显留给主链） |

实测过的数（GPU，4060）：单文件 40 s 量级的中文素材 **RTF ≈ 0.08**。

## 5. 已知洞与边界

- **VAD 为什么不用 crispasr 默认的 silero v6.2.0**：v6 对唱歌素材判 0 段 → 整条静默丢弃
  （rc 仍为 0、不落 `.txt`）。同一条 60 s 日推歌曲副歌：`firered-vad` 出 97 字真歌词、
  silero-v5 捡回 59 字、silero-v6 出 0 段。**代价是多花约 31% 墙钟**。
  FireRedVAD 的 voice 类是"语音 ∪ 唱歌"，所以它不会把唱歌整段丢掉。
- **CrispASR 自带文档与实现不符的三处**见下面 5.1，读源码就能复算，不需要跑 GPU。

### 5.1 与 CrispASR 自带文档不符的三处（读 v0.8.37 源码核出来的）

三条都是**代码级**断言，复算方法：clone 上游后 `git checkout d08ec2d`（= 0.8.37，发布件
`crispasr.exe --version` 打的 git sha 就是它），照下面给的 `文件:行` 逐行读。等级 **A**
（本包内可复算的读码结论，不依赖任何一台机器）。

**① `PERFORMANCE.md` 里 P0 那行说 firered 解码器自注意力"没有 KV cache —— 增长向量、O(T²) 重算"，
这条不成立。**

* 每个 beam、每一层都带一份历史 K/V：`src/firered_asr.cpp:2247-2258` 声明 `sa_k` / `sa_v`
  并按 `max_len` 预留容量；`:2340-2343`（贪心路）与 `:2563-2568`（beam 路）每步只把**当前这一步**
  的 K/V `insert` 到末尾，历史部分一次都不重算。
* 交叉注意力那侧的编码器 K/V 也是每层算一次、全部步复用（`K_enc[li]` / `V_enc[li]`，`:2372-2395` 里只读不重算）。
* 随长度增长的是 attention **打分本身**：每步 O(T)、整句 O(T²)。这是注意力的固有代价，
  和"没做 KV cache"是两件事，P0 那行把它们混成一件了。
* 误判可能的来源：context 结构体里确实有 `kv_self_k` / `kv_self_v` / `kv_cross_k` / `kv_cross_v`
  四个张量字段（`src/firered_asr.cpp:254-259`），但**全文件除声明外没有任何使用点**，是死字段。
  "结构体里有 KV cache 字段却没接上"看起来正好是 P0 那条说法的样子，实际在用的缓存是上面
  那两份 per-beam vector。
* 上游自己已经推翻过这条，而且就在同一份发布件里：`HISTORY.md:3449-3458`（2026-07-12 那节标题
  直接写着 "profiling debunked the handover"）用 `FIRERED_BENCH` 逐节点量下来是——self-attn 的
  K/V **已经缓存**（`beam.sa_k/sa_v` 每步只 append 当前 token）、单步耗时**与历史长度无关**，
  所以解码是 **dispatch-bound** 而不是注意力复杂度问题，真实开销是每步 90 多个小 matvec
  （8 个投影 × 16 层）。他们的修法是常驻图缓存 `CRISPASR_FIRERED_MATVEC_CACHE`
  （`src/firered_asr.cpp:385-391`，**默认开**，关掉走原来的每次建图路径，转写逐字节相同）。
  也就是说 0.8.37 到手时那条 P0 早已不成立，`PERFORMANCE.md` 那张表只是没跟着改。
* 对本包的意义：**别**用"CrispASR 没治 KV cache，所以 `cpu` 分支那条纯 ONNX 链更快是治了这个"
  来解释速度差——两侧都是增量 K/V，这条解释已经作废（那半边也有公开更正记录）。10-08 同机同件
  （80.2 s 素材、两边都 12 线程、都是 VAD + AED + 标点的整链）实测：ONNX 单件 23.85 s（RTF 0.297，
  另加 6.4 s 一次性装载），crispasr 贪心档 wall 31.9 s（自报 0.357）、默认 beam=3 wall 43.9 s
  （自报 0.501）；但同日更早那轮"单件对单件、两边各含一次装载"是 29.2 s 对 29.4 s，**打平**。
  也就是说差距来自**批次摊薄**（crispasr 每个文件重装一次模型、多个 `-f` 又省不下来），
  过去那个跨机器拼出来的"5 倍"从来不存在。剩下的差要归到计算图与执行器上（f32 编码器 + int8
  解码器的 mixed 档，见 `cpu` 分支 `FireRed-ONNX/README.md` §4.3）。

**② session API 没法把 FireRed 的解码固定在贪心：`beam_size = 1` 被当成"不设"。**

* `src/crispasr_c_api.cpp:1832` 里 `s->beam_size` 的初值就是 1；`:7551-7553` 写的是
  `if (s->beam_size > 1) firered_asr_set_beam_size(ctx, s->beam_size)`。于是调用方给 1 = 不调 setter
  = 落回 firered 自己的默认，而 `firered_asr_context_default_params()` 给的默认是 **3**
  （`src/firered_asr.cpp:297`）。"要贪心"和"没意见"在这个接口上是同一个值。
* 连带的第二条：firered 的 F1 复读熔断只接在 `beam_size == 1` 那条分支上
  （`src/firered_asr.cpp:2274-2283` 的注释明写 beam 路什么都没有，并举了硬音频上 "OOH"×35、
  跑满 `max_len` 烧掉约 350 s 解码的例子）。走 session API 的调用方同时拿不到贪心**和**复读熔断。
* **CLI 是另一条路，`-bs 1` 能进贪心**：`examples/cli/crispasr_backend_firered_asr.cpp:47` 直接
  `cp.beam_size = params.beam_size > 0 ? params.beam_size : 3`，1 会被传到
  `src/firered_asr.cpp:2311` 的 `beam_size == 1` 分支。（一处自我更正：早期我把这条写成
  "CLI 永远跑 beam 3、`-bs 1` 走不到贪心"，**CLI 那半句是错的**，走不通的是 session API 那条。）
* 本包这份驱动走的是 CLI（`crispasr.exe` 子进程），命令里**没有** `-bs`，所以取 CLI 的默认档。
  10-08 校机实测已核到：**不给 `-bs` 时日志逐条是 `firered_asr: decoder starting (max_len=150, beam=3, layers=16)`
  ⇒ 默认档就是 3**；给 `-bs 1` 才变 `beam=1`。`--help` 里那行 `-bs N [greedy]` 与运行时不一致，以运行时为准。
  10-08 之后把"CLI 不给 `-bs` 时那个默认值到底是几"这条从推断升级成了读码（等级 A，`git show d08ec2d` 可复现）：
  CLI 的活参数表是 `examples/cli/whisper_params.h:31`，`int32_t beam_size = -1;`，行尾注释写着
  "-1 = greedy; beam search only when explicitly set via -bs N" —— **这句注释对 qwen3 成立（`-1 > 1` 为假 ⇒ 不进
  beam 分支），对 AED 不成立**，因为上面那条适配器三元式把 -1 折成了 3。另：`cli.cpp:69` 那份带
  `whisper_full_default_params(...)` 的参数表整体在 `#if 0` 死块里（`:56` 起），早先"CLI 默认取自 whisper 默认参数"
  那句是读错了活代码，作废。
  要知道自己拿到的是哪一档，就对着日志里 `firered_asr: decoder starting (max_len=…, beam=…, …)`
  那一行看（`src/firered_asr.cpp:2269-2271`，verbosity ≥ 1 就打 = 默认会打，加了 `-q` / 静默档就不打）。
  速度代价也实测了：beam=3 比 `-bs 1` 慢 35%（12 线程）～49%（18 线程），加线程救不回来；
  "beam=3 更准"这一条**没有实测**，只是先验，动它之前自己复验。
* **温度（`-tp/--temperature`）不是 AED 的旋钮**（A，读码）：`--help` 那行
  （`examples/cli/cli.cpp:1046-1047`，默认 0.00）是通用的，但 firered 适配器全文不读
  `params.temperature`，`src/firered_asr.cpp` 也没有这个符号，能力表里没置 `CAP_TEMPERATURE`
  （`crispasr_backend_firered_asr.cpp:36-37`）⇒ 对 AED 传 `-tp` 是被静默忽略。
  真读它的是 LLM 那类后端：`crispasr_backend_qwen3.cpp:370` 塞进解码配置，`:396-399` 只在
  `temperature > 0` 时才把 argmax 换成 `sample_temp`（`src/core/greedy_decode.h:114`：logits 除以 T
  → softmax → 按分布抽，配 `--seed`），`:374` 的 `n_runs = (T > 0 && best_of > 1) ? best_of : 1`
  说明 **qwen3 的"多条路径"是 best-of-N 独立采样再取累计分（`:429-430` 打 `best-of-N picked score=`），
  不是 beam**；beam 那路要 `-bs > 1` 才走（`:269-305`，`core_beam_decode`）。
  `--temperature-inc`（默认 0.2）全文只被 whisper 家族读（`cli.cpp:2997`、
  `crispasr_backend_crispasr.cpp:120`），AED 与 qwen3 都不认。

**③ 每个切片的解码上限是硬顶 150 token，没有任何 CLI 参数能改 —— 所以 `--chunk-seconds` 不是速度参数。**

* `src/firered_asr.cpp:2075`：`int max_len = is_lid ? 2 : std::min(T_sub, 150);` —— 150 写死在源码里。
  `-n/--max-new-tokens`（默认 512）是 LLM 后端的路径（`src/core/greedy_decode.h:71`、
  `src/core/beam_decode.h:112`、`examples/cli/crispasr_backend_qwen3.cpp:253`），firered 适配器全文
  不读它，所以对 AED 无效。
* 该文件 `:2072-2073` 的注释按"3-4 BPE token/s"估算，认为 150 够用。**中文连续旁白实测到
  5.4-5.5 token/s**，150 token ≈ 27 s，所以 30 s 的切片会顶到 150 并静默丢尾。10-08 同机同链
  只差这一个参数、跑两条素材（逐片 token 数取自 crispasr 自己的日志）：
  s1.wav 80.2 s —— `30` 切 3 片 token 150/150/118（两片撞顶），去标点 445 字；`20` 切 5 片
  token 109/100/109/108/10（零撞顶），去标点 **463 字，+18 字 / +4.0%**（原文 1367→1423 B）。
  s2.wav 344.9 s —— `30` 切 12 片 token 150/150/115/146/130/145/150/150/150/150/150/141
  （**12 片里 7 片撞顶**），去标点 1771 字；`20` 切 19 片最长 135，去标点 **1850 字，+79 字 / +4.5%**
  （原文 5542→5818 B）。丢的都是整句尾巴：s1 缺"遍地""定独自持枪突入在所有民众的镜头中""毙"，
  s2 找回来三个完整句子。撞顶时 rc 仍为 0、日志只打 token 数、不报截断。
  同轮 `20` 档重复跑一遍逐字复现（43.9 / 43.6 s，token 序列与字数相同）。
  **速度代价接近零**：beam 档 s1 44.0→43.9 s、s2 184.4→181.4 s，贪心档 s1 36.2→31.9 s（−11.9%）。
* 对照：`--vad-export-raw` 那条文档说 VAD 段是 chunk-length-independent 的，但 crispasr 的 firered VAD
  实现（`src/firered_vad.cpp:446` 起）只有 `min_speech_sec` / `min_silence_sec`，**没有** 上游 FireRedVAD
  那个 `max_speech_frame=2000`（= 20 s）的长段强切。也就是说在这份构建里，切片的唯一上限就是
  `--chunk-seconds`，它必须 ≤ 20 s 才落在 150 token 之下。
* 补一层机制（10-09 读码，A）：**不传 `--chunk-seconds` 时这个上限不是"30"，而是没有**。
  `examples/cli/crispasr_run.cpp:1058-1062` 见到 AED 的 `CAP_UNBOUNDED_INPUT` 就把
  `effective_chunk_seconds` 置 0，而 `examples/cli/crispasr_long_audio_fallback.h:57` 的
  `if (wants_vad) return false` 让那条 30 s 兜底在开了 `--vad` 时不触发 ⇒ VAD 段多长就整段喂多长
  （`:1153 → :1251` 把 0 原样传进 `crispasr_compute_audio_slices`）。所以这条链上"传不传 `-ck`"
  是内容完整性参数，不是速度参数；上游对同一件事的措辞见 `CrispASR-Qwen/README.md` §4.1 ④ 引的
  `HISTORY.md:6210-6216`。
* 本包的处置：`crisper-xhs-qwen-asr.py` 的 `VAD_MAX_SEGMENT_SEC` 由 30 改成 **20**。
* 这个坑**只咬 AED**：qwen3 那条解码预算是 `max_new_tokens`（默认 512，`-n` 能改），
  30 s 档离顶还有 2~3 倍，所以 `CrispASR-Qwen/` 与 `cpu` 分支 `Qwen3/` 的 30 没动 ——
  推导与复核方法记在 `../CrispASR-Qwen/README.md`。

- **CrispASR 这份构建把 qwen3 模型内置的语种判别吃掉了**（`--list-backends` 没置
  `CAP_LANGUAGE_DETECT`），兜底那趟仍然靠 `-l auto` + whisper 前置判别。
  把驱动判到的语种码再传一次会把筛子的错误固化，所以不传。
- `--stop` / STOP 文件是**批边界**退出，一批最长可能等 `BATCH_SIZE` × 单文件时长。
- MSVC 4 件与 CUDA 运行库 3 件：前者是启动门槛（不补 rc=127，实测过），后者缺了会退化成
  CPU 后端或直接报错。

## 6. `"onnx"` 档：语种闸门 + FireRedASR2S 全链路（10-09）

### 6.1 形状

```
一个文件
  ├─ 闸门（crispasr 的两次短调用，固定 --gpu-backend cpu）
  │    1) --vad -vm <silero v6.2.0> --vad-export-raw <json> --strict-pipeline --require-vad
  │         → 真语音段表。0.46 s（80.2 s 素材）
  │    2) 按语音段拼前 15 s → 临时 wav → -m ggml-tiny.bin -dl
  │         → auto-detected language: xx (p = …)。1.07 s
  │    合计约 1.5 s/文件
  ├─ 判到 _AED_IN_RANGE 内 → ONNX 全链路（进程内常驻，不重载模型）
  │    FireRedVAD(ONNX) → AED(aed_ort.py, mixed 图) → FireRedPunc(ONNX)
  ├─ 判到范围外 → 不进主链，攒进 flagged → 同批换 crispasr qwen3 兜底（原逻辑不变）
  └─ 容器读不了（m4a / mp4 / aac，/libsndfile 不认）→ crispasr 的【同一个】主引擎模型
        它内部带 ffmpeg；不换语种、不换引擎，只换个解码器
```

改动只在"主力那一路由谁实现"。队列、硬链接暂存、批内实时结算、`no_speech.txt` 记账、
隔离区、熔断回搬、`.lock` 单实例、Job Object、`--start/--stop`、通知全部原样；
`_crispasr_batch` / `_crispasr_one` 一行没改（只是加了 `_` 前缀），把 `PRIMARY_LEG`
改成 `"crispasr"` 就是改前的行为。

### 6.2 闸门为什么是这两步，以及它的失手方式

* 第 1 步必须 `--vad-export-raw`：不带 `-raw` 导出来是 `kind="chunks"`（30 s 网格），
  不是语音段（10-09 实测）。段表里 `start/end` 是**采样点**、`t0_cs/t1_cs` 是**百分秒**，
  文件里不写单位；驱动取带 `_cs` 的那两个字段除 100 —— 名字自己说明单位，不靠猜。
* 第 1 步不带转写：给它 `-otxt -of` 也不落 `.txt`、不跑一个 token（rc=0 / 0.46 s /
  输出文件"无"，实测）。所以闸门不会"顺手先转一遍"。
* **两步合成一次调用不行**：`--vad-export-raw` 与 `-dl` 同时给时 crispasr 0.46 s 退出、
  rc 仍是 0，但**没有判别行** = 静默假成功（实测）。所以这里是两次进程调用。
* **silero 一条语音都没找到时，crispasr 自己不空手退出**：长素材（实测 120 s）会打一行
  `VAD returned no speech at all on a Ns clip — falling back to full-clip chunks`，然后
  **把整条按 30 s 网格当段表写出来**（`kind` 仍是 `vad_segments`，JSON 里没有任何标记能区分）；
  20 s 的纯器乐则老实地落 `slices: []`。所以段表"有段"不等于"有语音"。驱动按 stderr 那句
  （取纯 ASCII 片段 `full-clip chunks`，那行里的破折号过编码转换会花）把网格丢掉，让已经写好
  的 0 段分支接手 —— 两条分支的处置本来就相同（都判原始文件），但不丢掉的话日志会把
  "原始前 15 s"说成"语音前缀"，恰好与实际相反（10-09 校机实测，`tmp/c02.vad.json` 与
  `tmp/c04.vad.json` 两份段表都在）。
* 正对照（合成件 `c03_ctrl_music.wav` = 15 s 合成器乐 + 20 s 中文语音，标准答案 zh）：
  原始前 15 s → **en 0.784（错）**；语音前 15 s → **zh 0.966（对）**。探针
  `b1009_scan/gate5.py` 先测出这两个数，10-09 校机又用**出厂驱动的 `gate_language()` 本身**
  复跑了一遍（`sbh1009/probe2.py`），数值逐字相同 = 这条改进真的在跑，不是只在探针里成立。
  同一批 7 件里 VAD-first 只改判了这 1 件，另外 6 件两问一致，且**没有一件被改坏方向**
  （没有"原始判在内、闸门判到范围外"，也没有反向）。这就是第 1 步的全部价值 ——
  不加 VAD 的 `-dl` 会被片头音乐带偏（她记录里第 (6) 条那个老洞）。
* 判据沿用 `_AED_IN_RANGE` 那份名单，没有另立一套；ONNX 这份 AED 与 crispasr 那个是同一个模型，
  覆盖范围一样，而且**同样不吃语言标记**（判到的码只用来挑文件）。
* 闸门没有结论（抓不到判别行 / crispasr 非零 / 超时）→ **不扣文件，按主链跑**并记一行日志。
  宁可让 AED 编造一段（老风险），也不能因为闸门自己瞎了就把整棵树拖去慢 5 倍的兜底引擎。
  这一层的"瞎了"由起批前的 `gate_selfcheck()` 拦：拿前 3 个待处理文件实跑闸门，
  一个判别行都抓不到就 **rc=2 不启动**（`onnx_leg_setup()` 同样在起批前建一次三个会话，
  缺权重、provider 没落在卡上、依赖没装，都在这里当场停，不会偷偷改走 crispasr 跑完整夜）。

### 6.3 只用 N 卡，不碰核显

`ONNX_PROVIDER` 只认 `cpu` | `cuda`，`ep_list()` 对其它值直接退出。理由是**装了什么包不等于
跑在什么设备上**，三条都是为此：

* **不列 DirectML / OpenVINO**：那两个可以挑中 AMD 核显（这台笔记本的 iGPU 是 Radeon 780M，
  `Win32_VideoController` 打的型号，10-09 读的；集显跑重内核还会整机崩），ONNX 主链里唯一的 python 侧入口就是这一个常量。
* **VAD 与 Punc 恒定 CPU**：驱动在建 FireRedVadOnnx 会话时临时替换 `InferenceSession`
  工厂，把 `providers` 钉成 `["CPUExecutionProvider"]`（vendor 的 `infer_onnx.py` 自己没给
  providers，装了 onnxruntime-gpu 时它会自己挑设备），顺带补 `intra_op_num_threads`
  —— 那个文件建会话时没设线程数，ORT 的 0 = 吃满所有物理核。Punc 是 4 亿参数的 BERT、
  每段只推一次，上卡只跟编码器抢显存。不改 vendor 文件本体。
* **建完会话读 `get_providers()` 验一次**：`ONNX_PROVIDER="cuda"` 而 `enc_ep` 里没有
  `CUDAExecutionProvider` 就抛错。CUDA/cuDNN 运行库不齐时 ORT 只打一行警告就把整个会话
  退回 CPU，不读实际结果就是"说好用 N 卡、实际在 CPU 上跑一整夜"而没人看得出来。

显存与解码器上不上卡：默认 `ONNX_DEC_ON_GPU = False`（编码器上卡、解码器留 CPU）。
10-09 在 N 卡上自己测过（§6.6）：**放不下不是理由** —— 编码器 + int8 解码器都上卡时峰值
4,788 MiB，8 GB 档放得下（只放编码器是 4,256 MiB）；不让解码器上卡是因为**它上卡更慢**
（同一批三件素材 wall 418.9 s → 537.2 s，慢 28%）。逐 token 自回归的解码器每步都要发一批
kernel launch，crispasr 那边同源结论是 per-token GPU launch 20 ms 对 CPU 一整步 60 ms
（§5.1），两边方向一致。

### 6.4 为什么"整段无语音"和"判到范围外"处置不同

* 闸门判 0 段（silero 对唱歌素材的已知形态）→ 拼不出语音前缀 → 退回对**原始文件**直接 `-dl`，
  照样有语种结论，不会因此扣文件。crispasr 把 0 段换成 30 s 整段网格的那一种也走这里（§6.2）。
* ONNX 链自己那步 FireRedVAD 判 0 段 → 转写是合法空串 → `settle()` 走 `no_speech.txt` 记账、
  源文件照删。与 crispasr 那条 `no speech detected in '…'` 的记账口径相同。
* 转写侧的 VAD **没有**换成默认 silero：v6 会把唱歌素材整条吞掉（§5 第一条）。
  silero 在这里只出现在闸门，因为闸门要的恰好是"人在说话的那 15 秒"，不是"整条内容都在"。
* 撞 cache 上限被截断的段数会随统计打出来（`st["trunc"]`），与 `cpu` 分支那个驱动同口径；
  ONNX 这条的 VAD 自带 20 s 强切（上游 `max_speech_frame=2000`），结构上到不了 §5.1 ③ 那个 150 token 顶。

### 6.5 校机 CPU 冒烟（10-09 已跑完）

沙箱 = 一台 24 逻辑核、无独显的机器，`provider=cpu`、`graph=mixed`、`threads=4`，7 件素材
每件打一条分支，与生产批次同机共存 —— 所以下面的耗时是**被抢占中的读数**，只会偏高不会偏低。

* 建链：VAD 0.1 s + AED（mixed 图，权重 3358.3 MB）5.7 s + Punc 1.7 s = **7.4 s**，
  一次建好整批常驻。
* 主链单件：80.23 s 音频 / 79.27 s 语音 → **30.3 s**（对语音 RTF 0.38）；
  120.0 s / 79.25 s 语音 → **23.8 s**（RTF 0.30）；7.81 s / 7.72 s → 2.25 s；
  20 s 纯器乐 / 0 段 → **0.04 s**（无语音基本不花钱）。
* 闸门：每件 **1.1–1.8 s**（silero 导段 + `-dl`，两次 crispasr 短调用）。
* 分支覆盖：zh/en 进 ONNX 主链；ja 不进主链、同批攒给 crispasr qwen3 兜底（120 s 那件
  兜底 42.8 s）；`.m4a` 被 libsndfile 拒了 → 交给 crispasr 的**同一个** aed 模型（它内部带
  ffmpeg）；20 s 纯器乐 → 主链 FireRedVAD 判 0 段 → 合法空串 → `no_speech.txt` 记账 + 源文件
  删除。整批 7/7 有结论，`rc=0`。
* 回归：三次运行的 `p.txt` **逐字节相同**（4917 B，sha256 前缀 `eef6ba05`），其中第一次运行
  闸门是坏的（下面第 3 条）—— 印证闸门只改**路由**、不改主链出字。c05 与 `cpu` 分支那个独立
  链路驱动在同一件上的输出，去空白、去标点后 **463 / 463** 字符一致，再归一大小写就逐字相等
  （唯一的差是 Punc 把英文代词 `i'm` 写成 `I'm`，而参照 JSON 存的是加标点**之前**的分段文本）。

改造加这一轮冒烟一共逮到四个缺陷，全部已修，而且**都在新增代码里**（`_crispasr_batch` /
`_crispasr_one` 一行没动）：

1. **缩进错误**：合并兜底分支时把一处 `if/else` 改坏了，文件 import 就 `IndentationError`。
   先前"这份已经能编译"的说法是**错的，在此公开更正** —— 当时只看了 diff，没跑 `py_compile`。
2. **`FIREDASR_ONNX_ASR_DIR` 的空值哨兵被吃掉**：`resolve_dir(name, "")` 把 `""` 拼成了本包
   目录，链路于是去包目录下找 `encoder.f32.onnx`，起链当场 `FileNotFoundError`。这个空值的
   含义是"在 `ONNX_MODELS_DIR` 下按 `sherpa-onnx-fire-red-asr2*` 自动找"，现在只有环境变量
   真给了值才展开。现象是**起批前 `rc=2` 不启动**，不是偷偷换腿 —— 那道门禁本来就是为了
   这个才立的。
3. **闸门漏 `import json`**：`_gate_vad_segments` 读段表用 `json`，顶部却没 import，于是每件
   都抛 `name 'json' is not defined`、被 `except` 吞成一行 WARNING。闸门**看着在工作**
   （照样吐出 zh/ja 结论），实际全程走的是"判原始文件"那条退路 —— 功能没坏、改进没生效、
   日志只有 7 行 warning，是最阴的一种。修法是补 import，并把 7 件素材的两种问法并排打出来
   （`sbh1009/probe2.py`，就是上面那条"只改判 1/7"的表）来确认 VAD-first 真的在跑。
4. **`原始前 15s 那段没用上` 这句在单长段素材上是假的**（c05 的段表只有 0.03–79.55 s 一条，
   拼出来的前缀就是原始的前 15 s，一秒都没跳过）。日志改成给数：
   `语音前缀 15.0s(取自原始 0.0–15.0s 的 1 段)`；`_speech_prefix` 因此多返回
   `{head, tail, k}`。改完 `pyflakes` 在这份文件上只剩 `np` 那四处 —— 那是运行时从链路模块
   取的（`chain_module()` 里 `globals()["np"] = chain.np`），不是漏 import。

### 6.6 N 卡实机（10-09 已跑完，笔记本 RTX 4060 Laptop）

沙箱 = 作者自己的笔记本：独显 **RTX 4060 Laptop（`nvidia-smi` 报 8,188 MiB）**、核显
**Radeon 780M**（型号取自 `Win32_VideoController`，10-09 读的 —— 先前本节写的是 680M，
**那是错的，在此公开更正**）。环境 = 单独一个 venv（CPython 3.13.11，不带系统
site-packages），里面装 `onnxruntime-gpu==1.30.0`；系统侧是 CUDA toolkit 13.2 加
System32 里的 `cudnn64_9`。链路代码与三件权重从校机经 sftp 原样搬来，>1 MB 的 8 件
逐件 sha256 与校机那份对上（`mismatches: 0`）。

跑的件是**桌面那一份驱动**（与仓库这份同代码，只有 CONFIG 里的路径写法和本节这类说明
文字不同），输入是生产库里三件真素材的**副本**（66.92 + 296.86 + 617.78 = **981.56 s**
音频），输出与暂存都在沙箱目录，生产树一个字没动。

先单验运行时（`python -c "import onnxruntime"` + 三个会话各建一次）：1.30.0 在这台机器上
import 干净、`CUDAExecutionProvider` 在列、3.1 GB 的 f32 编码器会话确实落在卡上 ——
校机 10-05 那个"1.21 以上 import 就崩"的形态**没有复现**，所以那条是"看机器"，不是
"跨版本就不能用"（`requirements.txt` 最后那段已按这个改）。

四趟，同一份代码只换开关（g3 那份是把 `ONNX_DEC_ON_GPU` 改成 `True` 的副本，其余逐字相同），
输入是同一批副本：

| 趟 | 腿 | provider | 解码器 | rc | 整批 wall | RTF | 三件逐件用时 | `nvidia-smi` 峰值 |
|---|---|---|---|---|---|---|---|---|
| g1 | onnx | cuda | 留 CPU | 0 | **418.9 s** | **0.427** | 17.32 / 80.01 / 244.48 s | util 99%、4,256 MiB |
| g2 | onnx | cpu | 留 CPU | 0 | 555.4 s | 0.566 | 21.63 / 129.59 / 358.69 s | util 0%、210 MiB |
| g3 | onnx | cuda | **上卡** | 0 | 537.2 s | 0.547 | 18.98 / 136.67 / 336.26 s | util 100%、4,788 MiB |
| g4 | crispasr（参照） | `--gpu-backend cuda` | —（它没有这个开关） | 0 | 1084.2 s | 1.105 | AED 段 840 s 只出 1 件 / 兜底段 242 s 出 2 件 | util 峰值 96%、4,104 MiB（**这峰不是 AED 段造成的，见 g5**） |
| g5 | crispasr AED 腿单件定位 | `--gpu-backend cuda` | — | 0 | 253.4 s | 0.853 | `[魅族17]` 一件 296.86 s | util 中位 **0%**、p95 9%、单点最高 50%；显存恒 1,116 MiB |

* **编码器上卡有效，比纯 CPU 快 1.32 倍**（wall 555.4 → 418.9 s；逐件 1.25×/1.62×/1.47×）。
  `执行提供者:enc=['CUDAExecutionProvider', 'CPUExecutionProvider'] dec=['CPUExecutionProvider']
  punc=['CPUExecutionProvider']（VAD 恒定 CPU）` 这一行是驱动自己读的 `get_providers()`，
  不是推断。装载：VAD 0.1 s + AED 12.6 s + Punc 4.2 s = 17.0 s（整批常驻，一次）。
* **解码器上卡是负收益**：g3 比 g1 慢 28%，而且三件逐件都慢（1.10×/1.71×/1.38×）。
  显存反而不是瓶颈 —— 两个都上卡峰值 4,788 MiB，8 GB 档放得下（只放编码器 4,256 MiB）。
  所以 `ONNX_DEC_ON_GPU` 保持 `False` 的理由是**速度**，先前写"放不下"那条已经改掉（§6.3）。
  crispasr 那边同源的结论（per-token GPU launch 20 ms 对 CPU 一整步 60 ms）方向一致。
* **同机整腿对照（g4）**：同一批三件副本走 `PRIMARY_LEG="crispasr"` 是 **1084.2 s（RTF 1.105）**，
  ONNX 腿是 418.9 s（0.427）⇒ **ONNX 这条腿在这台机器上少 61%**。拆开看这 1084.2 s 花在哪：
  AED 段 11:46:41→12:00:42 共 **840 s 只产出 1 件**（三件的前置 LID 都在这一段跑，其中两件被判到
  范围外、扣下不转写），兜底段 12:00:42→12:04:45 共 **242 s 出 2 件**（684.7 s 音频 ⇒ 那一段 RTF 0.354）。
  口径要讲清：这是**腿对腿**的账（两边都把 3 件输入变成 3 件产出），不是"AED 对 AED"—— g4 里
  只有 1 件真由 AED 转写，另 2 件是 qwen3。crispasr 每个文件重装模型、批次 `-f` 摊不开那件事
  （`cpu` 分支 §12 的校机实测）在这台机器上同样成立，而且它这里还多了一层"LID 判完再换引擎重跑"。
* **g4 那个 util 96%/4,104 MiB 不属于 AED 腿 —— g5 单独定位过**（这是本节的第二条更正：
  我先前只留峰值、没时间戳，读起来像"AED 腿也在用卡"，和 §6.6 之外那份"显卡整天空闲"的旧读数冲突）。
  g5 = 只放 g4 里判到 `zh`、确实走了 AED 的那一件（296.86 s），`PRIMARY_LEG="crispasr"`，
  采样器每 2 s 取一次并记时刻（`gpu1009/rungpu_aed1.py` → `sb2/run/g5_samples.json`）：
  **120 次采样里 util>5% 只有 6 次**（离散在 t=2.3/61.9/76.8/143/183.4/236.4 s），单点最高 50%、
  中位 0%、p95 9%，显存全程 1,116 MiB 常驻。⇒ **AED 腿确实只把切片编码器零星丢给卡、解码在 CPU**，
  那条"整卡 2-6 W、显存占着 1,230 MiB"的旧读数与这次对得上；而 4,104 MiB 只能是 qwen3 兜底段
  （1.7B q8 权重上卡）留下的。RTF 口径也顺手提一句：g5 单件 253.4 s（0.853）比 g4 里 AED 段的
  840 s/1 件便宜得多，因为那 840 s 装了**三件**的前置 LID。
* **换腿会改文本，不只是改速度**（g1 对 g4，尺子＝去空白 + Unicode `P*`/`S*`）：
  唯一两边都由 AED 转写的那件（`[魅族17]`）**1,454 字 对 1,458 字、相似 0.9657**，逐处差异是
  近音近形摆动加一类系统性差别 —— **crispasr 保留英文大写（`FLYME`/`NFC`/`IPHONE的X`/`MX`/`OS`），
  ONNX 这条链出小写并与中文黏住**（`flyme`/`nfc`…，与校机 10-08 那条 B 级观察同向）。
  另两件在两条腿上根本不是同一个引擎（ONNX 腿判 zh 全进主链；crispasr 腿判 km/ko 转 qwen3），
  所以只有一处值得记：数字形态 qwen3 给 `2015`/`31`/`220V`/`5C`，ONNX 给"二零一五""二百二十伏"，
  且 qwen3 会把短句压掉（`sb模块走功率十五瓦` → `SB15W`）。唱歌那件 0.164，不参与判据。
* **核显一次都没被选中**，判据是机制：`ep_list()` 只给 `cpu`/`cuda`（DirectML、OpenVINO 不列），
  VAD 会话被钉成 `["CPUExecutionProvider"]`，而 `nvidia-smi` 只看得到 NVIDIA 设备 —— 上面那几个
  util/显存读数按构造就是 4060 的。g2 是正对照（util 0%、只有 210 MiB 上下文）。核显侧的占用
  事后没有读数，所以"780M 全程没干活"是**机制推断**，不是仪表证据。
* **功率那一列不采信**：这台机器的 `nvidia-smi` 里 `power.limit` 是 `[N/A]`，util 同刻 99–100%
  时 `power.draw` 最大只读到 23–24 W，显然不是真值。
* **文本层面 GPU 与 CPU 两趟几乎同字**：ONNX 那三趟里两件中文素材**字数完全相同**
  （1,454 / 3,365），两两相似度 ≥0.9988，逐处差异都是单字级的近音/近形摆动（继↔即、拓↔踏、
  的↔了、业↔叶、安↔n），没有丢句、没有多出整段。第三件是唱歌素材，两腿都把它编成英文歌词且彼此相距很远
  （相似 0.10–0.27），这一件不能当 EP 之间的判据 —— §5 第一条说的就是它。
* **闸门那句"silero 判 0 段"是 silero 的真实判定，不是报错被吞**：三趟日志里
  `闸门 VAD 失败` 出现 **0** 次；拿同一个文件的 ASCII 名副本单独复跑那条 VAD 调用，段表
  照常落盘、`kind="vad_segments"`、`num_slices: 0`。同一条素材主链那步 FireRedVAD 出
  43.16 s 语音 / 5 段 —— 正好是"FireRed 的 voice = 语音 ∪ 唱歌、silero 会把唱歌整段吞掉"
  在真素材上的一个实例（§5 第一条、§6.4）。
  顺带一条**探针自己的坑**（不是驱动的洞）：crispasr 是 ANSI 程序，`-f` 直接给中文名一律
  `error: input file not found`（rc=2，`-dl` 也一样），所以自写探针必须走驱动那套
  `staged_inputs()` 的 ASCII 暂存再去抄命令，否则测出来的"闸门坏了"是假的。
* **两腿的语种筛子在这批素材上给出不同的路由**（未解释，B 级观察）：同一批三件，ONNX 腿的闸门
  判 `zh×3`（全进主链，批次 6.1 min、产出 3/3）；crispasr 腿自带的前置筛子判
  `km p=0.411`（唱歌那件）/ `zh` / `ko p=0.405`（617.78 s 那件）—— 于是 14.0 min 只产出 1/3，
  两件转 qwen3。两次用的都是同一个 `ggml-tiny.bin` 的 `-dl`，p 都在 0.2~0.4 这个低置信区，
  差别在喂进去的音频范围与执行后端（闸门那两步固定 `--gpu-backend cpu`、只给 15 s 语音前缀，
  而 crispasr 内部那条走的是它自己的取段）。这批素材按内容都是中文科技视频，所以 `ko` 那个结论
  几乎肯定是筛错了，但**我没有真值**（没有人工标注）⇒ 这条只记到"两条腿不一致"，不写"哪边对"。
  实践上的含义：把主腿换成 ONNX 之后，**有多少文件进主链这件事会变**，不是纯粹的换后端。

### 6.7 还没做的事

* `crispasr` 腿在这台笔记本上只量了整腿（g4）和 AED 腿单件的 GPU 占用（g5）；
  **beam 档、qwen3 兜底腿单独的速度都还没在这台机器上量过**。
* `ONNX_MODE` 仍是 `greedy`：beam 在 ONNX 链里实现了但未调通（慢 9 倍、输出退化）。
* 分语种验证集仍没有（与 `cpu` 分支同一条洞）。
* Linux 侧只到机制，没有实机跑过。
