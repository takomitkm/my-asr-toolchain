# CrispASR-Qwen —— CrispASR / Qwen3-ASR 的 GPU 批转链

驱动是 `Cg-qwen.py`（旧名 `crispasr-Qwen.py`），引擎是 CrispASR 的 `qwen3` backend（Whisper 式音频编码器 +
Qwen3 1.7B 解码器），跑在 CUDA 上。**一个模型直接出成品文本**：自带大小写、标点、
中英混排，所以链路上没有独立的标点模型，也不做前端预处理。

和 `cpu` 分支 `Qwen3/` 的关系：同一个驱动的两个构建。那份是 CPU 版
（`Cc-qwen.py` + `crispasr-windows-x86_64-cpu` 那套件），这份是 CUDA 版；
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
.venv\Scripts\python Cg-qwen.py --start   :: 分离后台启动
.venv\Scripts\python Cg-qwen.py --stop     :: 写 STOP 标志，下一批边界干净退出
.venv\Scripts\python Cg-qwen.py            :: 前台跑（Ctrl+C 一次=本批跑完退，两次=立刻杀子进程）
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

### 4.1 `--chunk-seconds 30`：Qwen 这条路没有 AED 那个丢字的坑

`../CrispASR-FireRed/` 那条 10-08 实测出"每个切片解码上限硬顶 150 token、`-n` 管不到、
撞顶静默丢尾"，那边的 `VAD_MAX_SEGMENT_SEC` 已改成 20（详见 `../CrispASR-FireRed/README.md` §5.1 ③）。
本包和 `cpu` 分支 `Qwen3/` 的 30 **不动**，理由分三层：

- **① 预算不是硬顶**（A，读 v0.8.37 源码）。`examples/cli/crispasr_backend_qwen3.cpp:253`
  写的是 `const int max_new = params.max_new_tokens > 0 ? params.max_new_tokens : 512;`，
  那个默认值本身在 `src/core/greedy_decode.h:71` 和 `src/core/beam_decode.h:112`
  （两处都是 `int max_new_tokens = 512; // hard cap on generated tokens`）。
  与 AED 的关键差别只有两点：**512 不随切片长度收紧**（AED 是 `min(T_sub, 150)`，切片越长顶越死），
  **而且 `-n/--max-new-tokens` 改得动**（AED 那条 CLI 上无路可走）。
  KV cache 按 `max(4096, prompt_len + max_new + 16)` 现场分配（`:254`），不存在"预算装不下"；
  不分块的代价是显存与时间随音频线性增长，不是丢字。
- **② 30 s 离顶还有两三倍**（B，外推，不是直测）。解码循环 `while (size < max_new_tokens &&
  最后一个 != eos)`，正常切片停在 EOS（`:262-267` 取 `<||im_end|>` 的 id）而不是停在 512。
  同权重同素材的本机数字是连续中文旁白约 5.5 token/s、每 30 s 用量 100-135 token
  （AED 自己日志报的，是 AED 词表的密度），Qwen3 的词表对中文更碎（只会比这个数大不会小），
  保守按 2 倍算，30 s ≈ 200-270 token，占 512 的 39%-53%。
  **顶真正会咬人的地方是长音频整趟解码**：VAD 漏切的长段、或 `--chunk-seconds` 调到 0（不分块），
  约 **75-90 s 连续语音**才够到 512。
- **③ 截断同样是静默的**（A）：`crispasr_backend_qwen3.cpp:807-818` 那个 `for (step < max_new)`
  的流式循环走到 `step + 1 == max_new` 就 `break`，全程没有一行"我撞预算了"的输出。
  所以真要动这条链的分块，**别指望日志告警**，只能自己数 token 或比对字数。
- **④ "qwen3 有没有默认分块值"= 有，恒为 30 s；"整条全吃"要主动写 `--chunk-seconds 0`**（A，读码）。
  CLI 的活默认在 `examples/cli/whisper_params.h:230-231`：`chunk_seconds = 30` +
  `chunk_seconds_explicit = false`。`examples/cli/crispasr_run.cpp:1057-1063` 那条"没显式传 `-ck`
  就把上限置 0"的豁免**只给声明了 `CAP_UNBOUNDED_INPUT` 或 `CAP_INTERNAL_CHUNKING` 的后端**，
  而 qwen3 的能力表（`crispasr_backend_qwen3.cpp:56-58`）两个位都没置 ⇒ **qwen3 不传 `-ck` 也是 30 s，
  开不开 VAD 都一样**（VAD 段仍被 30 s 再切，`crispasr_run.cpp:1153 → :1251`）。
  AED 相反，它声明了 `CAP_UNBOUNDED_INPUT`（`crispasr_backend_firered_asr.cpp:36`），于是不传 `-ck` 时
  上限被置 0，**而 0 在 VAD 模式下就是"没有上限"**：`crispasr_long_audio_fallback.h:57` 的
  `if (wants_vad) return false` 让兜底不触发，VAD 段不管多长都整段进解码器，配 `min(T_sub, 150)` 的硬顶
  —— 这才是 `../CrispASR-FireRed/README.md` §5.1 ③ 那个丢字洞的完整机制（不是"全吃"这一个词能说清的）。
  上游自己的措辞在 `HISTORY.md:6210-6216`："VAD slices on a `CAP_UNBOUNDED_INPUT` backend were capped at
  30 s … Mirror the CLI: VAD on + `CAP_UNBOUNDED_INPUT` + `chunk_seconds` not explicit ⇒
  `effective_chunk_seconds=0` (VAD bounds the slices)"。⇒ **这就是两份驱动都必须显式传 `--chunk-seconds` 的原因**：
  AED 不传就没有上限、直接撞 150，qwen3 不传就是写死 30。
  另外三条同一口径的读码：
  ① **不开** VAD、音频 >30 s、也没传 `-ck` 时，`crispasr_run.cpp:1095`（`kLongAudioFallbackChunkSeconds = 30`）
  与 `:1121-1133` 兜底按 30 s 定长切，并打 `auto-chunking at 30 s to keep encoder in its safe window`，
  提示语里明写"要整趟就 `--chunk-seconds 0`"；这条兜底被 `!(capabilities & CAP_UNBOUNDED_INPUT)`（`fallback.h:61`）
  挡住，所以只对 AED 那类后端有意义，对 qwen3 是空转（它的 `effective` 本来就不是 0）。
  ② qwen3 没有覆写 `prefers_vad()`（基类 `crispasr_backend.h:358` 返回 false，全文只有 cohere / gemma4 /
  parakeet 覆写），所以它不会自动帮你开 VAD。
  ③ `crispasr_run.cpp:1153-1157` 的 `slice_chunk_seconds` 只在后端自己声明 `vad_slice_cap_seconds() > 0`
  时才额外收紧（全文只有 `crispasr_backend_parakeet.cpp:335` 给了），qwen3 与 firered 都是基类的 0 ⇒
  传进去的 `--chunk-seconds` 就是 VAD 段的唯一再切上限。
  ⇒ 结论：**qwen3 默认 30 s 分块，全吃要主动 `--chunk-seconds 0`（`:1122` 的 `!chunk_seconds_explicit`
  闸也是因为它，显式 0 才不被兜底覆盖），代价是 512 token 预算顶在约 75-90 s 连续语音（②③两层），
  加上 KV 随音频线性涨。**
- 复算路径：上游 clone 后 `git checkout d08ec2d`（= 0.8.37），逐行看上面给的 `文件:行`。
  本机没有 qwen3 权重，②这一档没做直测；要把 ② 从 B 提到 A，就找一段 >90 s 不被 VAD
  切断的连续语音，同一条素材在 `-n 512` 和 `-n 1024` 下各跑一遍比字数 —— 字数变了就是顶到了。

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
  ——记账逻辑在 `Wlid-Og-fired-Cg-qwen.py` 和 CPU 版那份里，这份不改（驱动 CONFIG 一节末尾那条注释就写着这件事）。
  要吞了多少的账，用 `CrispASR-FireRed/` 那份驱动跑，或换 `firered-vad.gguf`。
- `--stop` / STOP 文件是**批边界**退出，一批最长可能等 `BATCH_SIZE` × 单文件时长。
- MSVC 4 件和（GPU 档才需要的）CUDA 运行库是两处人工门槛；前者 rc=127 已实测，
  后者缺了会退化成 CPU 后端或直接报错。
