# CrispASR-FireRed —— CrispASR / FireRedASR2-AED 的 GPU 双引擎批转链

驱动是 `xhs-asr.py`，主力引擎是 CrispASR 的 `firered-asr` backend
（FireRedASR2-AED：Conformer 编码器 + AED 解码器，**小红书自家**那套的 gguf 量化件），
兜底引擎是 `qwen3`。跑在 CUDA 上，一次调用喂多个 `-f`、模型只加载一次。

这条链的形状：AED 出**无标点**的字，标点后处理由 CLI 侧的 `fireredpunc` 加；
语种靠 `whisper-tiny` 前置筛子判；VAD 用 `firered-vad`。四件事各有各的文件，
所以这一份是三个 CrispASR 方案里组件最多的一套（自动下载合计 4,611,126,207 B）。

换引擎这件事在这份驱动里是**自动**的：前置筛子判到 AED 语种范围外的文件不落盘、
攒下来立刻用 qwen3 在同一批里重跑。以前那是"她自己切回 `CrispASR-Qwen/` 那份手工跑"。

---

## 1. 三步复原

```bat
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python fetch_assets.py          :: 下载 + 逐件 sha256 验收
```

`fetch_assets.py` 是通用的，读同目录 `assets.json`（URL、字节数、sha256、出处等级、许可证
都在里面）。落点默认在本目录下两个文件夹：

| 根 | 默认 | env | 装什么 |
|---|---|---|---|
| `bin` | `./crispasr` | `CRISPASR_BIN_DIR` | `crispasr.exe` + 7 个 dll + `crispasr-quantize.exe` + 文档 3 件 + CUDA 运行库 3 件（+ 4 个 MSVC 运行库 dll，需人工，见第 3 节） |
| `model` | `./model` | `CRISPASR_MODEL_DIR` | AED q4_k / FireRedPunc q8_0 / FireRedVAD / ggml-tiny / qwen3 q8_0 共 5 份权重 |

想落到别处：设清单里写的同名环境变量（`CRISPASR_BIN_DIR` / `CRISPASR_MODEL_DIR`），下载与驱动读同一个值；
只想改下载落点就用 `--set-root model=<别的目录>`（相对按包目录算、绝对照用，驱动那边要用同一个值）。
核对现状不联网用 `--check`，看全部 URL 与哈希用 `--list`，只补某一件用 `--only NAME`。

上面那两个 env 就是驱动 CONFIG 里 `CRISPASR_BIN_DIR` / `MODEL_DIR` 读的两个；数据面另有
`ASR_TEXT_DIR` / `ASR_AUDIO_DIR`。四个都不设时全部落在包目录下，所以**复原完不改一行就能跑**：

| env | 默认（相对包目录） | 是什么 |
|---|---|---|
| `CRISPASR_BIN_DIR` | `./crispasr` | 上面 `bin` 那个根 |
| `CRISPASR_MODEL_DIR` | `./model` | 上面 `model` 那个根 |
| `ASR_TEXT_DIR` | `./txt` | 转写文本与日志的输出目录（`rules.txt`、`tmplist.txt`、`no_speech.txt`、`log\` 都在这） |
| `ASR_AUDIO_DIR` | `./audio` | 音频数据面根目录，驱动用它的 `p\` 当输入队列、`f\` 当失败隔离区、`t\` 当 ASCII 暂存区 |

五份权重的文件名分别写在 `CRISPASR_MODEL_AED` / `CRISPASR_MODEL_QWEN3` /
`CRISPASR_PUNC_MODEL` / `CRISPASR_LID_MODEL` / `CRISPASR_VAD_MODEL`（见第 4 节），拼的都是
`MODEL_DIR` 下那几个名字，与 `fetch_assets.py` 的落点一一对得上。
值给相对路径就按包目录解析，给绝对路径就照用。`ASR_ROOT` / `ASRSOURCE` 必须在同一块盘上
（`t\` 用硬链接，跨盘会失败）。

## 2. 跑

```bat
.venv\Scripts\python xhs-asr.py --start   :: 分离后台启动
.venv\Scripts\python xhs-asr.py --stop     :: 写 STOP 标志，下一批边界干净退出
.venv\Scripts\python xhs-asr.py            :: 前台跑（Ctrl+C 一次=本批跑完退，两次=立刻杀子进程）
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
| `firered-vad.gguf` 2,357,952 B | `huggingface.co/cstr/firered-vad-GGUF` | **A** | VAD。**刻意不用 crispasr 默认的 silero v6.2.0**，理由见第 5 节。上游 BSD-2-Clause |
| `ggml-tiny.bin` 77,691,713 B | `huggingface.co/ggerganov/whisper.cpp` | **A** | `-l auto` 的前置语种判别器。必须本地文件，否则联网下、离线机器卡满超时 |
| `qwen3-asr-1.7b-q8_0.gguf` 2,506,723,200 B | `huggingface.co/cstr/qwen3-asr-1.7b-GGUF` | **A** | 兜底引擎权重。显存不够换同仓 q4_k（1,490,915,200 B / `ec197cef…`），换时 path 与 sha256 一起改 |

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
| `CRISPASR_LID_BACKEND` | `"whisper"` | 前置筛子；配合 CONFIG 里 `CRISPASR_LID_MODEL` 那个本地 `ggml-tiny.bin`。每文件判一次 |
| `BATCH_SIZE` | `28` | 一次调用喂多个 `-f`，模型/VAD 只加载一次 |

实测过的数（GPU，4060）：单文件 40 s 量级的中文素材 **RTF ≈ 0.08**。

## 5. 已知洞与边界

- **VAD 为什么不用 crispasr 默认的 silero v6.2.0**：v6 对唱歌素材判 0 段 → 整条静默丢弃
  （rc 仍为 0、不落 `.txt`）。同一条 60 s 日推歌曲副歌：`firered-vad` 出 97 字真歌词、
  silero-v5 捡回 59 字、silero-v6 出 0 段。**代价是多花约 31% 墙钟**。
  FireRedVAD 的 voice 类是"语音 ∪ 唱歌"，所以它不会把唱歌整段丢掉。
- **CrispASR 自带文档与实现不符的三处**见下面 5.1，读源码就能复算，不需要跑 GPU。

### 5.1 与 CrispASR 自带文档不符的三处（读 v0.8.37 源码核出来的）

两条都是**代码级**断言，复算方法：clone 上游后 `git checkout d08ec2d`（= 0.8.37，发布件
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
* 对本包的意义：**别**用"CrispASR 没治 KV cache，所以 `cpu` 分支那条纯 ONNX 链快 5 倍是治了这个"
  来解释速度差——两侧都是增量 K/V，这条解释已经作废（那半边也有公开更正记录）。速度差要归到
  计算图与执行器上（f32 编码器 + int8 解码器的 mixed 档，见 `cpu` 分支 `FireRed-ONNX/README.md` §4.3）。

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
  要知道自己拿到的是哪一档，就对着日志里 `firered_asr: decoder starting (max_len=…, beam=…, …)`
  那一行看（`src/firered_asr.cpp:2269-2271`，verbosity ≥ 1 就打 = 默认会打，加了 `-q` / 静默档就不打）。
  速度代价也实测了：beam=3 比 `-bs 1` 慢 35%（12 线程）～49%（18 线程），加线程救不回来；
  "beam=3 更准"这一条**没有实测**，只是先验，动它之前自己复验。

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
* 本包的处置：`xhs-asr.py` 的 `VAD_MAX_SEGMENT_SEC` 由 30 改成 **20**。
* 这个坑**只咬 AED**：qwen3 那条解码预算是 `max_new_tokens`（默认 512，`-n` 能改），
  30 s 档离顶还有 2~3 倍，所以 `CrispASR-Qwen/` 与 `cpu` 分支 `Qwen3/` 的 30 没动 ——
  推导与复核方法记在 `../CrispASR-Qwen/README.md`。

- **CrispASR 这份构建把 qwen3 模型内置的语种判别吃掉了**（`--list-backends` 没置
  `CAP_LANGUAGE_DETECT`），兜底那趟仍然靠 `-l auto` + whisper 前置判别。
  把驱动判到的语种码再传一次会把筛子的错误固化，所以不传。
- `--stop` / STOP 文件是**批边界**退出，一批最长可能等 `BATCH_SIZE` × 单文件时长。
- MSVC 4 件与 CUDA 运行库 3 件：前者是启动门槛（不补 rc=127，实测过），后者缺了会退化成
  CPU 后端或直接报错。
