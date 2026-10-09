# faster_whisper —— faster-whisper / large-v3 的 GPU 批转链（最老、最轻的一条）

驱动是 `Wg-large.py`（旧名 `whisper-batch10.py`，`batch10` 是当时的批大小 10），引擎是
faster-whisper（OpenAI Whisper large-v3 的
CTranslate2 移植）+ `BatchedInferencePipeline`。四套方案里它最老：一个 ct2 模型目录
+ 一组 pip 包就能跑，VAD（silero，库内置）和标点（whisper 自己出）都不用外部件。

代价也在这：whisper 系有著名的"顺句读/概括"倾向，中文长口语的信息保留度不如
`CrispASR-FireRed/` 那条 AED 链，而且 `large-v3` 的权重（3.0 GB）比 qwen3-1.7B 那份
（1.49 GB）还大一倍。它的优势是**部署面最宽**：只要 pip 装得上。

---

## 1. 三步复原

```bat
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python fetch_assets.py          :: 下载 ct2 模型目录 + 逐件 sha256 验收
```

`fetch_assets.py` 是通用的，读同目录 `assets.json`。这一套只有一个需要落地的根：

| 根 | 默认 | env | 装什么 |
|---|---|---|---|
| `model` | `./model` | `FASTER_WHISPER_MODEL_DIR` | `model.bin` / `config.json` / `preprocessor_config.json` / `tokenizer.json` / `vocabulary.json` 共 5 件 |

默认落点和驱动 CONFIG 里 `MODEL_PATH` 算出来的路径是同一个（都在包目录下，两边都不含机器
专属路径），所以**这一条链复原完不用改一行就能跑**。要换地方就设环境变量
`FASTER_WHISPER_MODEL_DIR`，并且让下载与驱动用同一个值（相对路径按包目录算、绝对路径照用；
`--set-root model=<同一个值>` 只改下载落点，不会写进驱动）。
核对现状不联网用 `--check`，看全部 URL 与哈希用 `--list`，只补某一件用 `--only NAME`。

数据面同样靠环境变量，不给就用包内的默认：

| env | 默认 | 装什么 |
|---|---|---|
| `ASR_TEXT_DIR` | `./txt` | 转写文本与日志的输出目录（`ear.txt`、`a.txt`、`rules.txt`、`log\` 都在这） |
| `ASR_AUDIO_DIR` | `./audio` | 音频数据面根目录，驱动用它的 `converted\` 当输入队列、`failed\` 当失败隔离区 |

## 2. 跑

```bat
.venv\Scripts\python Wg-large.py
```

**这份驱动没有 `--start` / `--stop`、没有单实例锁、没有 STOP 文件**（那三件是 CrispASR
两份驱动后来加的）。前台跑，Ctrl+C 记一条"用户中断"就退。要后台就自己套
`Start-Process`，但注意：关掉启动它的控制台会连坐杀子进程这条在这份上不存在
（模型是在 python 进程里跑的，没有子进程），反过来也一样——进程一旦被外部终止，
正在转的那个文件就没了。

I/O 契约（与前两条链**不一样**的地方不少，别当同一条用）：

- 输入：递归扫 `ASRSOURCE\converted` 下的 `.mp3 .m4a .mp4 .wav .oga .ogg .opus`
  （`EXTENSIONS` 那个集合，**没有 `.flac` / `.aac`**）。
- 取文件方式：**每次只取 mtime 最早的那一个**（`find_oldest_audio`），转完再找下一个 ——
  不是批量调用，也没有"一级子目录分组"的概念。
- 输出：**单一文件** `ASR_ROOT\ear.txt`，追加写：`title:<相对 converted 的路径>` + 每段
  一行 `[起秒 -> 止秒] 文本`（带时间码）+ 空行。
- 成功 send2trash 源文件；失败 `shutil.move` 到 `ASRSOURCE\failed\`（**平铺，不保目录结构**，
  同名会撞），并继续跑下一个；每轮结束在日志里打错误汇总。
- `rules.txt` 在 `ASR_ROOT`（默认 `<包目录>/txt/rules.txt`），格式是
  **每行 `正则 = 替换`**，分隔符要求两侧各有一个空格（`load_rules` 里那个 `if " = " in line`）——
  和 CrispASR 那两份的 `原词=替换词` 不是同一个格式，两边不能直接共用一份文件。
  `#` 开头是注释。
- 没有 `tmplist.txt` 这类复读黑名单，也没有 `no_speech.txt` 记账：VAD 判空的直接不出字，
  但 `ear.txt` 里仍会留下一条只有 `title:` 和时间码为空正文的记录。
- `handle_no_file()`（把 `ear.txt` 归档进 `a.txt` 再把 `ear.txt` 删掉）在 `main_loop` 里
  是**注释掉的状态**（`# handle_no_file()`），所以现在 `ear.txt` 只会一直长。要归档就
  自己把那一行取消注释，或手动搬。

## 3. 组件清单与出处等级

`assets.json` 是机器可读的那份（`python fetch_assets.py --list` 打印全表）。摘要：

| 件 | 来源 | 等级 | 说明 |
|---|---|---|---|
| `model.bin` 3,087,284,237 B | `huggingface.co/Systran/faster-whisper-large-v3` | **A** | sha256 = HF 的 LFS oid（发布方凭据），与作者生产机那份逐字节相同 |
| `config.json` 2,394 B | 同上 | **A** | 非 LFS、仓库里没有 oid，这份 sha256 是直接从 huggingface.co 取的线上件算的，与生产机那份逐字节相同 |
| `preprocessor_config.json` 340 B | 同上 | **A** | 同上 |
| `tokenizer.json` 2,480,617 B | 同上 | **A** | 同上 |
| `vocabulary.json` 1,068,114 B | 同上 | **A** | 同上 |
| cuDNN 9 + cuBLAS 12 | NVIDIA（`developer.nvidia.com/rdp/cudnn-download`） | **C** | 要登录接受条款才能下，没有匿名直链。`--check` 对这一条只给 WARN |

等级口径：**A** = 发布方给的哈希与作者生产机那份逐字节相同；**B** = 哈希只取自生产机那份；
**C** = 没有公开匿名下载件，需人工。自动下载合计 3,090,835,702 B。

程序本体不在 `assets.json` 里：`faster-whisper` / `ctranslate2` 都是 pip 包，
钉死的版本和实测在位的传递依赖见 `requirements.txt`。

### 3.1 上传前对"必须组件是否完善"做了什么

这台机器和校机都跑不了 CUDA（校机无独显、无 `nvidia-smi`），所以这一轮不是端到端实测：

1. **闭包**：这份驱动只引用一个**目录**（CONFIG 里的 `MODEL_PATH`），5 个文件名是
   faster-whisper 的模型加载器自己拼的。对法改成"清单 ↔ 生产机目录"：两边都是 5 件，
   字节数逐件相符，没有一边多一边少的情况。
2. **真算 sha256 的那几件**：`fetch_assets.py --check --only …` 直接对生产机目录算过
   `config.json`（2.34 KB）、`preprocessor_config.json`（340 B）、`tokenizer.json`
   （2.37 MB）、`vocabulary.json`（1.02 MB）—— 全部"sha256 对上"。
   `model.bin` 3.0 GB 本轮没有重算，依据是 HF 的 LFS oid + 字节数相符。
3. **依赖面**：从生产机的 site-packages 目录名读出实际在位的版本（`faster_whisper-1.2.1` /
   `ctranslate2-4.7.1` / `av-16.1.0` / `tokenizers-0.22.2` / `huggingface_hub-1.5.0` /
   `onnxruntime-1.24.2` / `numpy-2.4.2`），并把 faster-whisper 1.2.1 的 `METADATA`
   里那组 `Requires-Dist` 一起抄进 `requirements.txt`，所以"装得上"这件事是有依据的。

**没有做过的**：GPU 上的端到端跑通，以及 cuDNN/cuBLAS 那两件人工件在干净机器上的补齐流程。
纯 CPU 的机器请直接用仓库 `cpu` 分支（`Qwen3/` 或 `FireRed-ONNX/`），不要靠这条链 ——
它没有 CUDA 运行库就只是慢，不是不能跑，但那就完全违背选它的理由了。

## 4. 关键 CONFIG 值（为什么是这个值）

| 常量 | 值 | 为什么 |
|---|---|---|
| `DEVICE` / `DEVICE_INDEX` | `"cuda"` / `0` | 多卡才需要动 `DEVICE_INDEX` |
| `COMPUTE_TYPE` | `"int8_float16"` | 8 GB 档显存的选择：权重走 int8、激活走 fp16。换 `"float16"` 更准更吃显存，换 `"int8"` 更省但慢 |
| `LOCAL_FILES_ONLY` | `True` | **必须 True**：False 时 `WhisperModel` 会去联网取 HF，离线机器上就是每轮白等一次超时；True 保证用的就是 `fetch_assets.py` 落地那 5 件 |
| `BATCH_SIZE` | `10` | 注意语义：这是 `BatchedInferencePipeline` 的**音频批大小**（一次前向吃几个 VAD 段），不是 CrispASR 那两份的"一次调用喂几个文件" |
| `LANGUAGE` | `None` | 让 whisper 自己判语种。这份驱动**没有**独立的前置判别器，语种只取 whisper 首段那一次判定 |
| `INITIAL_PROMPT` | `""` | 留空。想给热词就填这里（whisper 的 prompt 是"上文样式"的提示，不是硬替换） |
| `vad_parameters`（transcribe 调用里） | `min_silence_duration_ms=500` | 库内置 silero VAD；判空的那段直接不出字 |

## 5. 已知洞与边界

- **cuDNN9 / cuBLAS12 必须是进程能搜到的位置**（第 3 节那条 C 级）。生产机上这两个是装在
  系统的 `System32` 里的（随 NVIDIA CUDA Toolkit / cuDNN 安装包进去）。另一条路是 pip 装
  `nvidia-cudnn-cu12` / `nvidia-cublas-cu12` 再把 `site-packages\nvidia\*\bin` 加进 PATH。
  **别和 CrispASR 那两套混**：CrispASR release 附带的 `cublas64_12.dll` 是 113,712,640 B，
  System32 里这套是 113,716,224 B，差 3,584 B，不是同一版。
- **失败隔离是平铺的**：`failed\` 下同名文件会互相覆盖（`shutil.move` 到
  `FAILED_FOLDER / source_file.name`）。CrispASR 那两份是按相对路径保结构的。
- **没有单实例锁**：同时起两份会抢同一批文件，且两份都会往 `ear.txt` 追加。
- **无语音/空转写的文件仍会在 `ear.txt` 留一条记录**：`title:` 那一行照写，正文一段都没有
  （空转写的 `full_text` 是 `"\n"`，非空、判不出"没转出来"）。CrispASR 那两份的做法是
  空转写即删、不落记录（`CrispASR-FireRed/` 那份另记 `no_speech.txt`）。
- **`rules.txt` 格式不通用**（要 `正则 = 替换`，空格是分隔符的一部分），而且这里的
  "规则"是纯 `re.sub`，非法正则会在 `re.compile` 那一步抛、被 `load_rules` 的外层 try 接住
  并当场结束解析 —— 只打一条 `Failed to load rules`，**坏行之后的规则全都不生效**
  （前面的还在）。CrispASR 那两份是逐条容错、单条非法只跳过该行。写规则文件时这一条最容易踩。
- whisper 系的"顺句读"倾向：它可能把一段话改写或概括。要信息保留度用 AED 那条链。
