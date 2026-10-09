# Qwen3 —— CrispASR / Qwen3-ASR 的纯 CPU 批转链

这一套是**没有显卡、只有 CPU 的机器**上跑的 Qwen3-ASR 批转驱动。
引擎是 CrispASR 的 `qwen3` backend（Whisper 式音频编码器 + Qwen3 1.7B 解码器），
**一个模型直接出成品文本**：自带大小写、标点、中英混排，所以链路上没有独立的标点模型，
也不做前端预处理。CrispASR 那份 CUDA 构建把内置的语种判别吃掉了（详见第 6 节），
所以这套用 `-l auto` + whisper-tiny 前置判别器补回来。

仓库里另一套 CPU 方案是 `../FireRed-ONNX/`（FireRedASR2S 全链纯 onnxruntime）。
两者的取舍见**仓库根 README 的引擎对比表**（main 分支）。

---

## 1. 三步复原

```bat
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python fetch_assets.py          :: 下载 + 逐件 sha256 验收
```

Linux / macOS 同三条，把 `.venv\Scripts\` 换成 `.venv/bin/`（驱动本身跨平台，
后台启动走 `start_new_session`；但 crispasr 的 Windows 构建只在 Windows 上有，
换系统要去 CrispASR 的 release 页取对应 tar.gz，URL 形态见 `assets.json` 那条）。

`fetch_assets.py` 落点默认在本目录下两个文件夹：

| 根 | 默认 | 装什么 |
|---|---|---|
| `bin` | `./crispasr` | `crispasr.exe` + `openblas.dll`（+ 4 个 MSVC 运行库 dll，需人工，见第 3 节） |
| `model` | `./model` | `qwen3-asr-1.7b-q4_k.gguf` / `ggml-silero-v6.2.0.bin` / `ggml-tiny.bin` |

想落到别处：设清单里写的同名环境变量（`CRISPASR_BIN_DIR` / `CRISPASR_MODEL_DIR`），下载和驱动读的是
同一个值；只想改下载落点就用 `--set-root model=<别的目录>`（相对路径按包目录算，绝对路径照用，但记得
驱动那边要用同一个值）。核对现状不联网用 `--check`，看全部 URL 与哈希用 `--list`。

上面那两个 env 就是驱动 CONFIG 里 `CRISPASR_BIN_DIR` / `MODEL_DIR` 读的两个；数据面另有
`ASR_TEXT_DIR` / `ASR_AUDIO_DIR`。四个都不设时全部落在包目录下，所以**复原完不改一行就能跑**：

| env | 默认（相对包目录） | 是什么 |
|---|---|---|
| `CRISPASR_BIN_DIR` | `./crispasr` | 上面 `bin` 那个根 |
| `CRISPASR_MODEL_DIR` | `./model` | 上面 `model` 那个根 |
| `ASR_TEXT_DIR` | `./txt` | 转写文本与日志的输出目录（`log\`、`rules.txt`、`tmplist.txt`、`no_speech.txt` 都在这） |
| `ASR_AUDIO_DIR` | `./audio` | 音频数据面根目录，驱动用它的 `p\` 当输入队列、`f\` 当失败隔离区、`t\` 当 ASCII 暂存区 |

`ASR_ROOT` / `ASRSOURCE` 必须在同一块盘上（`t\` 用硬链接，跨盘会失败）。

## 2. 跑

```bat
.venv\Scripts\python Cc-qwen.py --start     :: 分离后台启动
.venv\Scripts\python Cc-qwen.py --stop       :: 写 STOP 标志，下一批边界干净退出
.venv\Scripts\python Cc-qwen.py              :: 前台跑（Ctrl+C 一次=本批跑完退，两次=立刻杀子进程）
```

I/O 契约（与 `../FireRed-ONNX/Oc-fired.py` **逐条相同**，两套可互换）：

- 递归扫 `ASRSOURCE\p` 下的音频（`.mp3 .m4a .mp4 .wav .oga .ogg .opus .flac .aac`），
  按**一级子目录分组**，组内按 mtime 升序；根上的散文件归 `p` 组。
- 输出 `ASR_ROOT\<组名>.txt`，**追加**写：`title:<相对 p 的路径>` + 正文 + 空行。
- 成功 send2trash；空转写即删且不落记录；失败按相对路径隔离进 `f\`（保目录结构）。
- `.lock` 单实例锁，**与 `../FireRed-ONNX/` 那份驱动互斥**（同一份数据面不能两个引擎同时吃）。
- `rules.txt`（同目录有 `rules.example.txt`）逐条容错，单条非法正则只跳过该行；
  `tmplist.txt` 是幻觉复读的黑名单。
- `no_speech.txt`：VAD 判"整段无语音"的文件 rc 仍是 0、不落 `.txt`，转写树上什么都不留，
  所以在这里记账（时间 \t 原因 \t 相对路径）。
- ntfy 收尾推送可选：`set NTFY_TOPIC_URL=https://…` 才发，不设就跳过（端点不入库）。

## 3. 组件清单与出处等级

`assets.json` 是机器可读的那份（`python fetch_assets.py --list` 打印全表）。摘要：

| 件 | 来源 | 等级 | 说明 |
|---|---|---|---|
| `crispasr.exe` + `openblas.dll` + `crispasr-quantize.exe` + LICENSE + THIRD_PARTY_NOTICES | GitHub release `v0.8.37/crispasr-windows-x86_64-cpu.zip`（8,705,179 B / `630ffec1…`） | **A** | zip 的 sha256 是 release 给的 digest；5 个成员各自的 sha256 从 zip 里算，且与作者生产机那份相同 |
| `msvcp140/vcomp140/vcruntime140/vcruntime140_1.dll` | 微软 `vc_redist.x64.exe`（aka.ms 链接） | **C** | CPU 构建动态链 MSVC 运行库。装了 redist 就不用管；不能装就拷这 4 份到 exe 同目录。`--check` 只给 WARN |
| `qwen3-asr-1.7b-q4_k.gguf` 1,490,915,200 B | `huggingface.co/cstr/qwen3-asr-1.7b-GGUF` | **A** | sha256 `ec197cef…` = HF 的 LFS oid = 作者生产机实测。也就是 CrispASR 自己 `-m auto` 注册表 `qwen3-1.7b` 那一行（`src/crispasr_model_registry.cpp:182-184`），不是另找的路子 |
| `ggml-silero-v6.2.0.bin` 885,098 B | `huggingface.co/ggml-org/whisper-vad` | **A** | crispasr 自己的默认 VAD（`examples/cli/crispasr_vad_cli.cpp:19-20`）。同仓 v5.1.2 是同字节数不同内容的另一份（`29940d98…`），换版本要同时改 path 和 sha256 |
| `ggml-tiny.bin` 77,691,713 B | `huggingface.co/ggerganov/whisper.cpp` | **A** | `-l auto` 的前置语种判别器（`models/download-ggml-model.sh:9` 的 src 同仓）。必须给成本地文件，否则 crispasr 会去联网下、离线机器上卡满超时。语种写死（如 `"zh"`）时这件不参与拼命令，可以不下 |

等级口径：**A** = 发布方给的哈希与作者生产机那份逐字节相同，照链接下就是同一个文件；
**B** = 哈希只取自生产机那份、没有独立凭据可对照；**C** = 没有公开匿名下载件，需人工。
这一套 5 件里 4 件 A、1 件 C —— 也就是说 CPU 这条链**零本地专有件**，全部可从公开链接复原。

### 3.1 一键复原已在纯 CPU 机器上实测（2026-10-07）

在一台没有任何显卡的 Windows 11 机器上、在一个空目录里照第 1 节那三条命令跑，全程联网走公开链接：

| 步骤 | 结果 |
|---|---|
| `fetch_assets.py` 真下载 | 通过 4 项、警告 1 项（只有 MSVC 那件是 `manual`）、不通过 0 项。`crispasr-windows-x86_64-cpu.zip` 8,705,179 B 整包 sha 对上，解压后 5 个成员**逐个** sha256 对上 |
| 1.39 GB 那个 gguf 中途断流 | 收到 174,634,371 B 时连接断，脚本用 HTTP Range 从断点接上下完，最终 sha256 仍对上（`.part` 续传逻辑实测有效） |
| `fetch_assets.py --check`（不联网） | 5 项全通过 |
| 补那 4 个 MSVC dll 前后 | **不补：`crispasr.exe --version` 直接 rc=127**，报 `api-ms-win-crt-utility-l1-1-0.dll: cannot open shared object file`。拷进 exe 同目录后 rc=0，打印 `version 0.8.37 / git sha d08ec2dd`。所以 C 那一件是**真门禁**，不是可选洁癖 —— Windows 10/11 上装过 `vc_redist` 的机器一般已经在 System32 里，没装过就必须手动补这一件 |
| 驱动端到端 | 按第 1 节把四个目录指到复原出来的位置、输入队列里放音频，跑通出中文转写：`<组名>.txt` 里 `title:` 头 + 带标点正文，`tmplist.txt` 记账，日志在 `log\` |

Python 侧只用到 `Send2Trash`（驱动顶部 `from send2trash import send2trash` 是模块级 import，缺了直接崩），
系统自带的 Python 或免安装的 embeddable Python 都能跑，不需要 CUDA、不需要 torch。

## 4. 关键 CONFIG 值（为什么是这个值）

| 常量 | 值 | 依据 |
|---|---|---|
| `CRISPASR_BACKEND` | `"qwen3"` | 引擎选择 |
| `CRISPASR_GPU_BACKEND` | `"cpu"` | 无显卡机器必须显式改；不改会去找 ggml-cuda |
| `CRISPASR_THREADS` | `0` | 0 = 交给 crispasr 自动取"逻辑线程数 × 0.75"。系数来自 16 逻辑线程机器上的 `-t` 扫点（12 最快，RTF 0.59），B 级读数 |
| `CRISPASR_LANGUAGE` | `"auto"` | `zh` 与 `auto` 不是谁更准，而是各自有已证的坏法：`zh` 在整段非中文的文件上复读崩盘，判别器错码会把内容摘要化丢掉。逐条依据在驱动 CONFIG · 语种一节 |
| `CRISPASR_LID_BACKEND` | `"whisper"` | 配 CONFIG 里 `CRISPASR_LID_MODEL` 那个本地 `ggml-tiny.bin` |
| `BATCH_SIZE` | `28` | 一次 `crispasr.exe` 调用喂多个 `-f`，模型与 VAD 只加载一次（重载 1.7B 权重每次约 6 s） |

### 4.1 `--chunk-seconds 30`：这条链没有 AED 那个丢字的坑（10-08 读码核过）

`gpu` 分支 `CrispASR-FireRed/` 那条 10-08 实测出"FireRed-AED 每个切片的解码上限硬顶 150 token、`-n` 管不到、撞顶静默丢整句"，那边的 `VAD_MAX_SEGMENT_SEC` 已由 30 改成 20（逐片 token 数、推导与复核方法都在 `CrispASR-FireRed/README.md` §5.1 ③）。本包和 `gpu` 分支 `CrispASR-Qwen/` 的 30 **不动**，三条理由：

- **预算不是硬顶，而且改得动**（A，读 v0.8.37 源码）。`examples/cli/crispasr_backend_qwen3.cpp:253` 写的是 `const int max_new = params.max_new_tokens > 0 ? params.max_new_tokens : 512;`，那个默认值本身在 `src/core/greedy_decode.h:71` 和 `src/core/beam_decode.h:112`（两处都是 `int max_new_tokens = 512; // hard cap on generated tokens`）。与 AED 的关键差别只有两点：**512 不随切片长度收紧**（AED 是 `min(T_sub, 150)`，切片越长顶得越死），**而且 `-n/--max-new-tokens` 改得动（AED 那条在 CLI 上没有入口）。KV cache 按 `max(4096, prompt_len + max_new + 16)` 现场分配（`:254`），不存在"预算装不下"；不分块的代价是内存与时间随音频线性增长，不是丢字。
- **30 s 离顶还有两三倍**（B，外推，不是直测——校机上那份 1.39 GB 的 qwen3 gguf 已随旧测试目录删掉，没有权重就测不了这一档）。正常切片是停在 EOS 而不是停在 512（`crispasr_backend_qwen3.cpp:262-267` 取 `<|im_end|>` 的 id，循环条件 `gen.back() != cfg.eos_id`）。同权重同素材的本机实测是连续中文旁白约 5.5 token/s、每 30 s 用量 100-135 token，Qwen3 的词表对中文更碎（只会比这个数大不会小），保守按 2 倍算，30 s ≈ 200-270 token，占 512 的 39%-53%。**顶真正会咬人的地方是长音频整趟解码**：VAD 漏切的长段、或把 `--chunk-seconds` 设成 0（不分块），大约 **75-90 s 连续语音**才够到 512。
- **截断同样是静默的**（A）：`crispasr_backend_qwen3.cpp:807-818` 那个 `for (step &lt; max_new)` 的流式循环走到 `step + 1 == max_new` 就直接 `break`，全程没有一行"我撞预算了"的输出。所以真要改这条链的分块，别指望日志告警，只能自己数 token 或比对字数。

**"qwen3 有没有默认分块值"= 有，恒为 30 s；"整条全吃"要主动写 `--chunk-seconds 0`**（10-09 读码，A）。
上面那句"30 不动"容易被读成"30 是我们挑的"，实际不是：**30 是 crispasr 写死的默认，本包只是没去改它**。
CLI 的活默认在 `examples/cli/whisper_params.h:230-231`（`chunk_seconds = 30` + `chunk_seconds_explicit = false`），
而 `examples/cli/crispasr_run.cpp:1057-1063` 那条"没显式传 `-ck` 就把上限置 0"的豁免**只给声明了
`CAP_UNBOUNDED_INPUT` 或 `CAP_INTERNAL_CHUNKING` 的后端**，qwen3 的能力表（`crispasr_backend_qwen3.cpp:56-58`）
两个位都没置 ⇒ **不传 `-ck` 也是 30 s，开不开 VAD 都一样**（VAD 段仍被 30 s 再切，`crispasr_run.cpp:1153 → :1251`）。
AED 相反：它声明了 `CAP_UNBOUNDED_INPUT`（`crispasr_backend_firered_asr.cpp:36`），不传 `-ck` 时上限被置 0，
**而 0 在 VAD 模式下就是"没有上限"** —— `crispasr_long_audio_fallback.h:57` 的 `if (wants_vad) return false`
让兜底不触发，VAD 段不管多长都整段进解码器，配 `min(T_sub, 150)` 的硬顶；这就是 `gpu` 分支
`CrispASR-FireRed/README.md` §5.1 ③ 那个丢字洞的完整机制，不是"全吃"一个词能说清的。上游自己的措辞在
`HISTORY.md:6210-6216`："VAD slices on a `CAP_UNBOUNDED_INPUT` backend were capped at 30 s … Mirror the CLI:
VAD on + `CAP_UNBOUNDED_INPUT` + `chunk_seconds` not explicit ⇒ `effective_chunk_seconds=0` (VAD bounds the slices)"。
⇒ **仓里所有 crispasr 驱动都必须显式传 `--chunk-seconds`**：AED 不传就没有上限、直接撞 150，qwen3 不传就是写死 30。
再加三条同一口径的读码：① **不开** VAD、音频 >30 s、也没传 `-ck` 时，`crispasr_run.cpp:1095`
（`kLongAudioFallbackChunkSeconds = 30`）与 `:1121-1133` 兜底按 30 s 定长切，并打
`auto-chunking at 30 s to keep encoder in its safe window`，提示语里明写"要整趟就 `--chunk-seconds 0`"；
这条兜底被 `fallback.h:61` 的 `!(capabilities & CAP_UNBOUNDED_INPUT)` 挡住，对 qwen3 是空转（它的 `effective` 本来就不是 0）。
② qwen3 没有覆写 `prefers_vad()`（基类 `crispasr_backend.h:358` 返回 false，全文只有 cohere / gemma4 / parakeet 覆写），
所以它不会自动帮你开 VAD。③ `crispasr_run.cpp:1153-1157` 的 `slice_chunk_seconds` 只在后端自己声明
`vad_slice_cap_seconds() > 0` 时才额外收紧（全文只有 `crispasr_backend_parakeet.cpp:335` 给了），qwen3 与 firered
都是基类的 0 ⇒ 传进去的 `--chunk-seconds` 就是 VAD 段的唯一再切上限。
⇒ 结论：**qwen3 默认 30 s 分块；"整条全吃"要主动写 `--chunk-seconds 0`，代价是 512 token 预算顶在约 75-90 s
连续语音（上面第二条与这条 ④），加上 KV 随音频线性涨。**

复算路径：上游 clone 后 `git checkout d08ec2d`（= 0.8.37，发布件 `crispasr.exe --version` 打的 git sha 就是它），照上面给的 `文件:行` 逐行读。要把第二条从 B 提到 A：找一段 &gt;90 s 不被 VAD 切断的连续语音，同一条素材在 `-n 512` 与 `-n 1024` 下各跑一遍比字数——字数变了就是顶到了。
## 5. 已知洞与边界

- **silero v6.2.0 对唱歌素材判 0 段**：rc 仍 0、不落 `.txt`、日志只有 "no speech detected"，
  整条被静默丢弃。口语素材上 v5/v6 打平（断句与标点有别、内容零进出），但**这不等于全语料打平**。
  作者的记账方式是 `no_speech.txt`（见第 2 节）。换成 `firered-vad.gguf` 能捡回唱歌内容，
  代价是在 CPU 上多花约 31% 墙钟（B 级，n=1 的两次读数）。
- **待转写文件先硬链接到 `ASRSOURCE\t` 下的 ASCII 名**：`crispasr.exe` 是 ANSI 程序，
  argv 经系统 ACP（中文 Windows = cp936）转换，文件名里 GBK 表示不了的字符会变成 `?`、路径随之失效。
  这是 crispasr 子进程路线独有的限制；`../FireRed-ONNX/` 那份进程内驱动不需要这一套。
- **长跑期间不要关启动它的那个控制台窗口**：Windows Terminal 的进程树连坐会一起杀掉 worker。
  停止只用 `--stop`。
- 校机那台 24 vCPU 无显卡机器上，同一份 89.84 s 中文素材、`-t 12`：q4_k 67.6 s（RTF≈0.75），
  q8_0 84.3 s 且两份文本不逐字相同 —— 所以量化固定 q4_k，别"换大量化换精度"。B 级读数（驱动注释里记的那次）。

## 6. "CrispASR 把内置 LID 吃了"是什么意思

Qwen3-ASR 上游的做法是把语种写进解码前缀（`language Chinese<asr_text>` 这类），模型自己带多语种能力。
CrispASR 的 gguf 移植在 CLI 路径上不接收这个前缀，所以**驱动只能靠外部前置判别器**决定语种：
`-l auto` + `--lid-backend whisper --lid-model ggml-tiny.bin`，只听开头几百毫秒、出一个语种码、
不改一个字。判别器给的码错了，后续解码就在错的语种假设下走 —— 已证的坏法是**把内容摘要化丢掉**。
这条在本仓库的四套引擎里只有 Qwen3 有（FireRed 的 AED 完全无语言输入，见 `../FireRed-ONNX/README.md`；
`faster-whisper` 与 `CrispASR-FireRed` 用的是各自的 LID 做法）。
