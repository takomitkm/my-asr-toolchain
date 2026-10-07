# firedasr-onnx-win-aigc

FireRedASR2 纯 ONNX 批量转写链（CPU · 不需要 torch / 不需要 sherpa-onnx / 不需要 GPU）。
这里是产线正在跑的那套脚本的**副本**：去掉了所有机器专属路径，每一项可配置的东西都能从外部指定，
包内**不带** dll / onnx 这类大文件 —— 它们要么有匿名下载入口，要么由包内脚本从下载件建造，
一个脚本就能把缺的全补上。

主战场是 Windows，但**建环境和转写脚本都是跨平台的**，Linux/macOS 用 `python3 setup.py`
（`setup.bat` 现在只是 Windows 上双击着方便的转发壳）。平台差异集中在第 3 节末尾。

三步跑通 —— Windows：

```bat
setup.bat                                    :: 等价于 python setup.py：建 venv + 装依赖（两份 requirements 全装，下载约 73 MB）
.venv\Scripts\python.exe fetch_assets.py --graph mixed   :: 下载 + 建造全部模型（常驻约 4.6 GB；int8 档 1.68 GB，见 4.1）
.venv\Scripts\python.exe xhs-chain-cpu.py --input-dir .\audio --data-root .\data
```

Linux / macOS：

```sh
python3 setup.py                             :: 同上，venv 建在 .venv/（x86_64 下载约 79 MB，见 3.2）
.venv/bin/python fetch_assets.py --graph mixed
.venv/bin/python xhs-chain-cpu.py --input-dir /srv/audio --data-root /srv/asr
```

前两条也可以用系统 python；`fetch_assets.py` 不带 `--models` 时默认就落在包旁的 `models/`，
所以第三条不用额外指模型目录。

这套"一键"不是纸面承诺：2026-10-06 在产线那台机器上做过对拍，资源清单 17 项逐 sha256 全对、
三件建造件从 int8/q8w 原料重跑得到与产线文件同哈希的产物，并拿这批重跑出来的模型跑了一条音频
——除墙钟耗时外，输出与产线模型逐字相同。全表见 12.4。

---

## 1. 这条链的形状

```
音频文件 ──► FireRedVAD(ONNX)      判哪里有人说话，顺带把长音频切成 <=20 s 的段
         ──► FireRedASR2-AED(ONNX) 纯 onnxruntime 跑 encoder+decoder，贪心解码出原始文本
         ──► FireRedPunc(ONNX)     逐段加标点（官方口径），再走官方 RuleBaedTxtFix 收尾
         ──► <out-dir>\<组名>.txt   title:<相对路径> + 正文 + 空行，追加写
```

* 三个模型进程内**只加载一次**、跨文件复用；链内不起任何外部子进程。
* 批量驱动 `xhs-chain-cpu.py` 直接 import `asr_chain.py` 复用同一份实现，
  差别只有两点：模型保活、VAD 走内存直喂（不落临时 wav；float→int16 用 `rint`，
  与链路 `sf.write` 的 PCM_16 舍入口径一致）。
* `asr_chain.py` 可以单独跑一条音频（`--json` 出结构化结果，含逐段边界和分环节耗时），
  批量不需要它，但留着当"最小可跑核"和排障工具。
* 语种判定（LID）这一环**默认不跑**：FireRedASR2 不吃语言标记，LID 只是事后标注。
  想要就见第 8 节。

数据面契约（每条都是"跑完一个文件才动那个文件"，全程可断可续）：

| 环节 | 行为 |
|---|---|
| 输入 | `--input-dir` 递归扫音频（后缀集合 `--extensions`），按**第一级子目录**分组，组内按 mtime **升序**（时间顺序）；根目录散文件归入 `--root-group`（默认取输入目录名） |
| 输出 | 每组追加写 `<out-dir>\<组名>.txt`；一条记录 = `title:<相对输入根的路径>` + 正文 + 空行（含子目录时路径带反斜杠） |
| 结算 | 成功或空转写 → 按 `--on-done` 处理源文件；失败 → 按相对路径隔离进 `--fail-dir` |
| 熔断 | 连续 `--abort-consec-fail` 个文件一个都没成功 = 判为环境故障：本段被隔离的文件**全部搬回**输入目录，剩下的原样留着，退出码 3，绝不报"转完了" |
| 单实例 | 锁 `<out-dir>\.lock`；停止用 `--stop` 或 Ctrl+C（一次=当前文件跑完再停，两次=立即停） |
| 队列快照 | `--tmplist`；日志 `<log-dir>\chain_*.log`（UTF-8） |

**默认 `--on-done trash`，也就是转写成功后把源文件丢进回收站**（产线在用这一档）。
想让文件留在原地就显式加 `--on-done keep`。

---

## 2. 包里有什么

随包的源码（`sha256` 取前 16 位，2026-10-06 本地计算）：

| 文件 | 作用 | 必须 | sha256 |
|---|---|---|---|
| `xhs-chain-cpu.py` | 批量驱动：扫描/分组/续跑/结算/熔断/停止/后台化（Windows + Linux/macOS） | 是 | `c55c8d2d7948b52f` |
| `asr_chain.py` | 三环核：VAD→AED→Punc 的加载与推理，单文件 CLI | 是 | `5ec36d4e60bcd443` |
| `aed_ort.py` | FireRedASR2-AED 的纯 onnxruntime 实现（int8/f32/mixed 三种图，贪心与 beam） | 是 | `2d6018194602eb06` |
| `models/fireredvad-onnx/infer_onnx.py` | FireRedVAD 的 ONNX 推理（上游原文件：HF `tardigrade-doc/FireRedVAD_onnx` 上那份，2026-10-06 在校机上实取逐字节相同） | 是 | `dda73c30f190956e` |
| `fetch_assets.py` | 一键补全资源：下载 + 建造 + 逐件 sha256 验收 | 是 | `2c0cff9c63f83877` |
| `dequant_punc.py` | 把 `punc.q8w.onnx` 的 8-bit `MatMulNBits` 逆量化成普通 `MatMul`，产出 `punc.f32.onnx` | 建造期 | `631241ce963cfb21` |
| `dequant_aed.py` | 把 sherpa 的动态量化 int8 图重写成纯 f32 图（外置权重 `.data`） | 建造期 | `b0637acb0979dd6e` |
| `requirements.txt` | 转写运行时依赖（钉死版本，理由见第 7 节） | 是 | `b73c4f1978bc6892` |
| `requirements-build.txt` | 建造期依赖（只要 `onnx`+`numpy`） | 建造期 | `7b7fc899cbddc8c0` |
| `setup.py` | 建 venv + 装依赖 + 导入自检 + 列出还缺什么资源（Windows / Linux / macOS 同一个文件） | 方便用 | `2b2b5a0fee6a2280` |
| `setup.bat` | Windows 上的壳：找到 python 就转过来调 `setup.py`，逻辑不在这里 | 方便用 | `e9c5e7062c6f92be` |
| `rules.example.txt` | 后处理规则文件格式示例（**不是**产线那份，产线那份含具体词表、不随包发） | 否 | `e6b1bc6a0de3765b` |
| `.gitignore`、`.gitattributes` | 仓库元数据：把 `.venv/`、`__pycache__/` 和 `fetch_assets.py` 落下来的整棵 `models/` 资源树挡在库外（只放回上游那两份文本）；`.gitattributes` 就一行规则 `* -text` = **禁止 git 转换任何换行**，所以上面这张哈希表在 `git clone` 之后仍然逐字节可复算（bat 是 CRLF、`optional-lid/lid_onnx.py` 也带着产线那会的 CRLF、其余 LF）。手动拷包用时这两件不参与运行 | 只有进 git 才有用 | `daad8816a20170de` / `1ba7a18741202a76` |
| `optional-lid/lid_onnx.py` | 自导 FireRedLID ONNX 的运行侧 | 否 | `26ca6bcd8441395b` |
| `optional-lid/lid_export.py`、`lid_export_dec.py` | 从官方 torch 权重导出上面两个图（需要 torch） | 否 | `e48345553fdf2ce0` / `1d40f71034183b01` |

包内**没有**的东西，全部由 `fetch_assets.py` 补：`.onnx`、`.onnx.data`、`cmvn.bin`、
`tokenizer.json`、`tokens.txt`、`out_dict` 等。目录里只剩一个 270 B 的
`models\fireredvad-onnx\README.md`（上游模型卡片；随包的 `infer_onnx.py` 和它来自
**同一个 HuggingFace 仓** `tardigrade-doc/FireRedVAD_onnx`，见第 10 节）。

上面这张表随时能自己重算（改过脚本后请以重算结果为准）。这条在两个平台都能直接跑，
Windows 上把开头的解释器换成 `.venv\Scripts\python.exe`：

```bat
.venv\Scripts\python.exe -c "import hashlib,glob;[print(f, hashlib.sha256(open(f,'rb').read()).hexdigest()[:16]) for f in glob.glob('*.py')+glob.glob('optional-lid/*.py')+['setup.bat','rules.example.txt','requirements.txt','requirements-build.txt','models/fireredvad-onnx/infer_onnx.py']]"
```
```sh
.venv/bin/python -c "import hashlib,glob;[print(f, hashlib.sha256(open(f,'rb').read()).hexdigest()[:16]) for f in glob.glob('*.py')+glob.glob('optional-lid/*.py')+['setup.bat','rules.example.txt','requirements.txt','requirements-build.txt','models/fireredvad-onnx/infer_onnx.py']]"
```

---

## 3. 第 0 步：环境

`setup.py` 会：检查解释器版本 → 在包目录建 `.venv` → 装 `requirements.txt` 和
`requirements-build.txt` → 导入自检（打印 onnxruntime 版本和它认到的 provider）→ 打印"还缺哪些资源"。
**三个平台同一个文件**，venv 里的解释器路径按平台选（Windows `.venv\Scripts\python.exe`，
Linux/macOS `.venv/bin/python`）。`setup.bat` 现在只剩一层壳：找到 python 就转过来调 `setup.py`，
逻辑不在 bat 里（旧版 bat 那套"找 `py -3.12`→建 venv→装包"已经整体搬进 python 了）。

```bat
setup.bat                    :: Windows，等价于下面那行
```
```sh
python3 setup.py             :: Linux / macOS
```

变体（参数三个平台一样）：

```
setup.py nobuilder          不装 onnx（只要转写、且不打算造 f32；用 --graph int8 就够）
setup.py noruntime          只装建造侧
setup.py --python <解释器>   用它来建 venv，默认拿当前解释器
setup.py --force            跳过 3.1 那个版本窗口检查
setup.py --skip-check       不跑最后那条资源核对
```

退出码：`0` = 环境就绪（资源还缺也算就绪，那是第 1 步的事）；`6` = 环境不对。
最后一步 `fetch_assets.py --check` 在未补资源时返回 1，这是预期的，`setup.py` 不把它当失败，
让你先看完那张表。想单独核对随时：

```
.venv\Scripts\python.exe fetch_assets.py --check     :: Windows
.venv/bin/python fetch_assets.py --check             :: Linux / macOS
```

手搓环境也行，只要 `pip install -r requirements.txt`。

### 3.1 解释器版本窗口：3.12–3.13

两个实测点：**3.12.15（产线在用）和 3.13.11（本包干净机那轮）全通**；3.10/3.11 这条链上没验过，
别当依据。（更正：这一段原先写的是"3.10–3.12 最稳"，那是照产线版本推的，不是实测——实际用来建
`.venv` 的那台干净机是 3.13.11。）

上下界不只靠实测，钉死的版本自己就写着（下列都取到 2026-10-06 的 PyPI 官方 JSON，A 级）：

* 下界：`numpy==2.5.3`、`scipy==1.18.1` 的 `requires_python` 都是 `>=3.12`；
* 上界：`onnxruntime==1.20.1` 的轮子只发到 **cp313**，3.14 没有轮子。

所以 `setup.py` 里 `PY_MIN=(3,12)` / `PY_MAX=(3,13)`，不在窗口内它当场退 6 并告诉你换解释器，
而不是让 pip 撞一半再失败。你的解释器确实在窗口外但想硬试，加 `--force`（解析失败别当本包的 bug 报）。

### 3.2 Linux 额外的三件事

* **glibc ≥ 2.28**：`onnxruntime==1.20.1`、`numpy==2.5.3`、`scipy==1.18.1` 的轮子标签是
  `manylinux_2_28`（≈ Debian 10+ / Ubuntu 18.10+ / RHEL 8+）。CentOS 7 那类老系统装得上但
  `import` 会报 GLIBC 版本不够 —— 第 3 步的导入自检会抓到并把这句打给你。
  `tokenizers` 是 `manylinux_2_17`、`kaldi-native-fbank` 是 `manylinux2014_2_17`，都不是瓶颈。
* **aarch64 有轮子**：这 8 件加建造侧的 `onnx==1.23.1` 都发了 `manylinux_*aarch64`，
  ARM 机器（树莓派 4/5、ARM 云机）不用编译；`kaldiio`、`Send2Trash` 是纯 python。
  所以下载量：x86_64 约 79 MB、aarch64 约 75 MB（Windows 约 73 MB，全部含建造侧；按 PyPI 报的
  轮子字节数加的，未算 pip 自己的依赖）。
* **venv 常被发行版拆包**：Debian/Ubuntu 上 `python3 -m venv` 会报 `ensurepip is not available`，
  补 `sudo apt update && sudo apt install -y python3-venv python3-pip` 再跑；
  `setup.py` 失败时打的也是这句。

### 3.3 平台差异（读代码得到，Linux 一侧没有实跑过）

链路里只有批量驱动碰操作系统，四处都已经按平台分支：

| 位置 | Windows | Linux / macOS | 说明 |
|---|---|---|---|
| 单实例锁 `\.lock` | `msvcrt.locking`（`xhs-chain-cpu.py:295`） | `fcntl.flock`（`:299`） | 都能防重复启动；POSIX 侧靠关句柄放锁（`:320`） |
| `--start` 后台化 | `DETACHED_PROCESS\|CREATE_NEW_PROCESS_GROUP`（`:880`） | `start_new_session=True` = setsid（`:883`） | Linux 上子进程脱离控制终端，关终端不会带走它 |
| Ctrl+N 静音 | `msvcrt` 轮询按键（`:365`） | 没有 `msvcrt` → 那个线程 `ImportError` 直接 return | **Linux/macOS 只能用 `--stop` 或 Ctrl+C** |
| `--on-done trash` | Send2Trash → 回收站 | Send2Trash → `~/.local/share/Trash`；库缺或抛异常 → 源文件保留 + 写一行 error 日志（`:542`） | 想留文件就显式 `--on-done keep` |
| 记录里的路径 | `title:<相对路径>` 含子目录时用 `\` | 同一行代码用 `Path.relative_to`，分隔符是本机 `/` | 分隔符不同，正文一样 |

模型件（`.onnx` / `.onnx.data` / `cmvn.bin` / `tokenizer.json`）**不含平台代码**，
Windows 上下好的模型目录整个拷到 Linux 也能用，反之亦然。但**整条 Linux 链路没有跑过**
（产线是 Windows，见第 9 节最后两行）——上表每一行都是从代码读出来的机制，不是行为实测。

---

## 4. 第 1 步：一键补全资源

```bat
python fetch_assets.py --graph mixed                 :: 下载 + 建造 + 验收
python fetch_assets.py --list                        :: 只打印清单（URL / 字节数 / sha256）
python fetch_assets.py --check                       :: 只核对现状，不下不建
python fetch_assets.py --models .\m --graph int8    :: 换模型根目录 / 换档
python fetch_assets.py --proxy http://127.0.0.1:7890 :: 需要代理（也认 https_proxy 环境变量）
```

要点：

* **逐件按 sha256 验收**，对不上就删掉重下一次，第二次还不对就硬失败 —— 不会把坏文件留在盘上让链路去猜。
* **断点续传**：半截文件写成 `<目标>.part`，按 `Range` 接着下（服务器不认 Range 就推倒重来）。
* **建造是确定性的**：造完逐件比对产线哈希，不一致会拒绝让你用（`--check` 也会报）。
  这条不是嘴上说说——2026-10-06 在校机上把三件建造件从 int8/q8w 原料**重跑了一遍**，
  产物（含两张外置权重 `.data`）与产线正在用的逐字节相同，见 12.4。
* 开始前先算空间：脚本按"常驻 + 解包瞬时"两个数报（`--graph mixed` 常驻 4.57 GB、峰值 5.35 GB，
  见 4.1 末表），不够就直接报"空间不够"并退出，不会下到一半写满盘。
  注意 AED 是 tar.bz2：**解压出来的成员比压缩包更大**（1.23 GB > 0.80 GB），
  所以别按"压缩包多大"估。
* `--dry-run` 打印将要执行的建造命令；`--force` 强制重下/重建；`--only name,name` 只做某几件；
  `--keep-tar` 解包后保留 tar.bz2（默认删）。

### 4.1 资源总表

下载件（全部匿名可取，2026-10-06 逐条实测 HTTP 200，不需要 token / 不需要登录）：

| 名字 | 落地位置（相对 `--models`） | 字节 | sha256 |
|---|---|---|---|
| `aed_int8_tar` | `fire-red-asr2-aed-int8.tar.bz2`（解包后默认删） | 838,589,068 | `43015b3f1643a5688b4821e8ed323473d38b798c4ec291471fe00df1bcfc4f1c` |
| ↳ 解出 | `sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26\encoder.int8.onnx` | 817,286,833 | `54048d66b6e8f3c80ea7ce95efe794587b0fd81d7271651d0decd3803852ae82` |
| ↳ 解出 | `…\decoder.int8.onnx` | 417,291,928 | `b840ce7196ae4a14d05ae84bbf56082b6b61ccec5610fda907dddbcea37354ff` |
| ↳ 解出 | `…\tokens.txt` | 79,172 | `1bc613de2112d257e61a349c3e72d1b1a9cf19c33d3ca954197ad2171e5ea07b` |
| `vad_model` | `fireredvad-onnx\model.onnx` | 2,461,278 | `517e9c6207618407da41fc274b1e3f09e8cde531db42f039a52be93b29a49151` |
| `vad_cmvn_bin` | `fireredvad-onnx\cmvn.bin` | 644 | `b020eb6a57b01993c7aa032fbb0e33d257359ef1bdcb4b66e3dc360f11b42d4e` |
| `vad_cmvn_json` | `fireredvad-onnx\cmvn.json` | 3,293 | `662d07cfe6e111ef386ce5b932def7e6c8218eb3fdd73db7588f2ab66851e825` |
| `vad_config` | `fireredvad-onnx\config.json` | 2 | `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` |
| `punc_q8w` | `fireredpunc-onnx\punc.q8w.onnx` | 162,771,205 | `5b7cfdd8a8b7228c56b4d2123b4b09a4af34d70cd43f613d1fcccb35bd2ece8f` |
| `punc_tokenizer` | `fireredpunc-onnx\tokenizer.json` | 268,961 | `53ff61207898738bbdc000f38abebef01041c8d23b6270c11855fc692d0a3ad6` |
| `punc_out_dict` | `fireredpunc-onnx\out_dict` | 33 | `6f0f7e0004881d617bc6e1d7b5b39972da80dcb49576bca489b1603ee55e20bb` |
| `punc_license` | `fireredpunc-onnx\LICENSE` | 11,357 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `punc_readme` | `fireredpunc-onnx\README.md` | 4,227 | `2ed2edffe0e405a0441e7e9e3c4efae11b10eb6e74131461754d3745a0c9c495` |

URL 前缀（`fetch_assets.py --list` 会逐条打全 URL）：

* AED：`https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26.tar.bz2`
* VAD：`https://huggingface.co/tardigrade-doc/FireRedVAD_onnx/resolve/main/<文件>`
* Punc：`https://huggingface.co/jiangzhuo9357/fireredpunc-onnx/resolve/main/<文件>`

建造件（**全网没有公开匿名下载入口**，由包内脚本从上面的下载件造出来）：

| 名字 | 产物（相对 `--models`） | 字节 | sha256 | 怎么造 |
|---|---|---|---|---|
| `punc_f32` | `fireredpunc-onnx\punc.f32.onnx` | 406,954,097 | `71c54314cb8129e4ca491169f4a770061041e8127e53b1b0edda423a8a103d95` | `python dequant_punc.py punc.q8w.onnx punc.f32.onnx` |
| `aed_encoder_f32` | `sherpa-…-2026-02-26\encoder.f32.onnx` | 956,747 | `1fa3b6c8503143a21d32fbf3a4dadae6ad73dfb4ebc6b3ac2d35c69a037e0b42` | `python dequant_aed.py convert encoder.int8.onnx encoder.f32.onnx` |
| ↳ 外置权重 | `…\encoder.f32.onnx.data` | 3,103,167,616 | `66c3b4f0293c476681b7c01ca7fac356e718878adc6bfbf0a7aefeed40b25f1a` | 同上，一次写出 |
| `aed_decoder_f32` | `…\decoder.f32.onnx` | 1,099,726 | `52bd0efacb0f536c3f4cc050a09d6094f3451d7938af52ca0f54c96fd03575a6` | `python dequant_aed.py convert decoder.int8.onnx decoder.f32.onnx` |
| ↳ 外置权重 | `…\decoder.f32.onnx.data` | 1,550,396,160 | `7728fe4ac558e36b0d0c07b619dde8e87aad8d7d2bf44a65432bf46b0406d654` | 同上 |

两张 `.data` 的哈希是 2026-10-06 在校机上算出来的，`fetch_assets.py` 现在**逐字节**核它们，
不再只核尺寸。同一轮里还查出一个必须知道的坑：**`onnx` 写外置权重对已存在的 `.data` 是
追加而不是覆盖** —— 在同一个目录里重跑 `dequant_aed.py`，第一次还好好的（956,747 B 图 +
3,103,167,616 B 权重），第二次就变成 957,649 B 图 + **6,206,335,232 B**（正好两倍）权重，
图里的 offset 跟着错位。包内 `dequant_aed.py` 已在保存前删掉上一轮的产物，所以
`--force` 重建、或者在已有 `models` 目录里重建，现在都能得到与上表一致的文件；
这条修复让包内那份与产线那份**不再逐字节相同**（见 12.1）。

`--graph` 决定建哪几件（和驱动侧 `--graph` 同名同义）：

| `--graph` | 要建 | 常驻 | 峰值（解包那一会儿） |
|---|---|---|---|
| `int8` | 只建 `punc_f32` | 1.68 GB | 2.46 GB |
| `mixed`（默认，产线在用） | `punc_f32` + `aed_encoder_f32` | 4.57 GB | 5.35 GB |
| `f32` | 再加 `aed_decoder_f32` | 6.02 GB | 6.80 GB |

`LICENSE` / `README.md` 两件（约 15 KB）不计进判据，但照样会下。

### 4.2 为什么必须"建造"，不能直接用下载件

两件事互相独立，别混：

1. **`punc.q8w.onnx` 在 onnxruntime 1.20 及更早加载不了。** 它把权重存成
   `com.microsoft.MatMulNBits`（weight-only 8-bit，`bits=8, block_size=32`，73 个节点），
   而 ≤1.20 的 contrib 只实现了 4-bit，直接报
   `matmul_nbits.cc:93 ... Only 4b quantization is supported`。
   `dequant_punc.py` 把 73 个节点全部逆量化成普通 `MatMul(+Add bias)`。
   公式是**核过**的不是猜的：`w = (q_uint8 - 128) * scale`（分块对称量化、无 zero_points），
   单独把一个 MatMulNBits 成图交给能跑 8-bit 的 ORT 当参照逐选手算，只有这一形式对得上
   （最大绝对差 9.5e-7，其它形式差 12 个数量级）。
   *顺带一句：如果你环境里的 ORT ≥1.21 且能正常 import，可以跳过这一步，
   直接 `--punc-file fireredpunc-onnx\punc.q8w.onnx` 用下载件；但实测 f32 反而**快 2.9 倍**
   （4 线程、101 个位置：183 ms vs 534 ms），因为运行时反量化比直接 SGEMM 贵，所以默认还是走 f32。*
2. **AED 的 f32 图是为了速度而不是为了能加载。** int8 图能直接跑；但实测（425.09 s 音频、
   VAD 20 s 分段、18 线程、同一台机器）f32 编码器把编码环节从 82.9 s 压到 28.9 s，
   所以产线用 `mixed` = f32 编码器 + int8 解码器。`dequant_aed.py` 把 sherpa 的
   动态量化图（`QLinearMatMul`/`DynamicQuantizeLinear` 那一套）重写成纯 f32 权重，
   权重超 2 GB 所以走 ONNX 外置数据格式（同名 `.onnx.data`）。

### 4.3 线程数：默认 0.5×逻辑 CPU，VAD 另给上限

`--threads` 的默认从 `逻辑CPU*0.75` 改成 `逻辑CPU*0.5`（那台 24 vCPU 上就是 18 → 12），
依据是 2026-10-06 在**一条 603.6 s / 248 段的真实音频**上逐档实测（机器空载，每档各跑一次）：

| `--threads` | 墙钟 | 累计 CPU 秒 | 实吃核数 | 相对 18 线程 |
|---|---|---|---|---|
| 18 | 96.8 s | 1,824 | 18.9 | — |
| 12 | 94.5 s | 1,216 | 12.9 | 同速，CPU **−33%** |
| 8 | 103.4 s | 882 | 8.5 | 时间 +6.8%，CPU −52% |
| 4 | 129.8 s | 541 | 4.2 | 时间 +34%，CPU −70% |

五档转写文本 md5 完全一致（`3ed5a9462d14…`，2,669 字符）——**降线程不改一个字的输出**。
12 比 18 还略快说明 18 是超配：多出来的线程在自旋和同步屏障上空转。想追极限吞吐就显式加
`--threads 18`（或更高），想省 CPU 跟别的程序同台就 8。

`--vad-threads`（默认 `min(8, 逻辑CPU)`，填 0 = 不设上限）管的是断句环：`models\fireredvad-onnx\infer_onnx.py`
建 session 时只设了 `graph_optimization_level`，没设 `intra_op_num_threads`，ORT 的默认 0 =
吃满所有物理核；而 VAD 是每个文件开头对**整条**音频一次性提特征，于是会瞬时点满全部核心。
驱动在建 VAD 期间临时替换 `onnxruntime.InferenceSession`，只给"没显式设过线程数"的 session
注入上限（ASR 由 `aed_ort` 显式设 `--threads`、标点显式设 `--punc-threads`，都不受影响），
建完立即还原工厂、不改 vendor 文件。这一档**收益不在账上**：产线侧那次的读数是省总 CPU 约 16 秒、
VAD 环节 1.0 s → 0.9 s，消掉的是那个尖峰（这两个数我没有独立复现，见 §9）。
默认不写死 8 而是取 `min(8, 逻辑CPU)`：核数不到 8 的机器上写死 8 会变成超订，而那台 24 vCPU 上算出来仍是 8。

---

## 5. 第 2 步：跑

Windows：

```bat
:: 前台
.venv\Scripts\python.exe xhs-chain-cpu.py --input-dir .\audio --data-root .\data

:: 后台（其余参数原样带进后台实例）
.venv\Scripts\python.exe xhs-chain-cpu.py --data-root .\data --start

:: 看状态（只读，不改任何东西）/ 下一个文件边界优雅停止
.venv\Scripts\python.exe xhs-chain-cpu.py --data-root .\data --status
.venv\Scripts\python.exe xhs-chain-cpu.py --data-root .\data --stop
```

Linux / macOS 换成 `.venv/bin/python` 和 POSIX 路径，命令形状一模一样：

```sh
.venv/bin/python xhs-chain-cpu.py --input-dir /srv/audio --data-root /srv/asr
.venv/bin/python xhs-chain-cpu.py --data-root /srv/asr --start     # setsid 脱离终端
.venv/bin/python xhs-chain-cpu.py --data-root /srv/asr --status
.venv/bin/python xhs-chain-cpu.py --data-root /srv/asr --stop
```

默认目录布局（全部从 `--data-root` 派生，想分开放就各自再给一个参数；下面按 Windows 写法，
Linux 上分隔符是 `/`）：

```
<data-root>\p            输入（待转音频）
<data-root>\txt          输出 <组名>.txt、rules.txt、tmplist.txt、.lock、STOP
<data-root>\txt\log      日志 chain_YYYYMMDD_HHMMSS.log
<data-root>\f            失败隔离区（按相对路径保目录形状）
<脚本目录>\models        模型（fetch_assets 的默认落点）
```

**停止只用 `--stop` 或 Ctrl+C，别关掉启动它的那个窗口** —— 在 Windows Terminal 下杀掉
启动链所在的窗口会连坐杀死后台 worker（实测 4/4），批次会静默死掉。Linux/macOS 上
`--start` 走的是 setsid，子进程已经没有控制终端，关终端不会带走它（读代码得到的机制，没实跑）。

退出码（`--help` 末尾也写着）：

| 码 | 含义 |
|---|---|
| 0 | 队列转完且没有失败 |
| 1 | 有文件失败（已隔离进 `--fail-dir`） |
| 2 | 引擎加载失败（模型缺 / ORT 起不来） |
| 3 | 连续失败熔断（被隔离的文件已搬回输入目录） |
| 4 | 已有实例在跑（锁没放） |
| 5 | 输入根读不到 —— **这不是"转完了"** |
| 6 | 参数或环境不对（目录建不出来、不认识的参数等） |

第 5 条和第 3 条是这套驱动被专门修过的地方：早先的版本会把"没扫到文件 / 全批失败"
报成成功结算，现在绝不报假成功。

---

## 6. 参数与优先级

每一项可指定项都 obey 同一条优先级：

```
命令行参数  >  环境变量 FIREDASR_<参数名大写>  >  默认值（相对脚本目录，不含任何机器专属路径）
```

批量驱动的开关（`--help` 全量）：

| 组 | 参数 |
|---|---|
| 路径 | `--data-root` `--input-dir` `--root-group` `--out-dir` `--fail-dir` `--log-dir` `--rules` `--tmplist` `--lock-file` `--stop-file` `--code-dir` `--models-dir` `--asr-dir` |
| 引擎 | `--threads`（默认 `max(1, 逻辑CPU*0.5)`，见 4.3）`--graph{mixed,int8,f32}` `--asr-mode{greedy,beam}` `--cache{auto,max,N}` `--punc-threads` `--vad-threads`（默认 `min(8, 逻辑CPU)`，0=不设限）`--no-punc` `--vad-dir` `--punc-dir` `--punc-file` `--min-seg-sec` |
| 队列 | `--extensions` `--on-done{trash,delete,keep}` `--notify-every` `--abort-consec-fail` `--repeat-filter-max-line` `--ntfy-url` `--log-level` |
| 动作 | `--start` `--stop` `--status` |

单文件脚本 `asr_chain.py` 另有 `--models --asr-dir --vad-dir --punc-dir --punc-file --model{aed,ctc}
--threads --asr-mode --asr-cache --asr-graph --no-vad --no-punc --punc-threads --json --max-secs
--lid --lid-backend --lid-mode --lid-win --lid-hop`。
注意它的 `--asr-graph` 默认是 `int8`（保守，对拍背书就在这一档），批量驱动默认是 `mixed`（产线档）。
`--models` 是给"整套模型都放在另一个根目录"用的：它会一次把 ASR、VAD、Punc、LID 四个派生目录
都指过去（`<models>/sherpa-onnx-fire-red-asr2-*`、`<models>/fireredvad-onnx`、
`<models>/fireredpunc-onnx`、`<models>/FireRedLID*`）；`--vad-dir`/`--punc-dir`/`--asr-dir`
这些单项参数在它之后生效，所以要覆盖单项就把单项参数和 `--models` 一起给。

环境变量清单（同一批名字，不加参数时用这些）：`FIREDASR_DATA_ROOT` `FIREDASR_INPUT_DIR`
`FIREDASR_OUT_DIR` `FIREDASR_MODELS`（=`FIREDASR_MODELS_DIR` 的链路侧别名）`FIREDASR_ASR_DIR`
`FIREDASR_VAD_DIR` `FIREDASR_PUNC_DIR` `FIREDASR_PUNC_FILE` `FIREDASR_LID_ONNX_DIR`
`FIREDASR_LID_DICT_DIR` 等 —— 完整一套就是上表参数名大写，`grep FIREDASR_ *.py` 能一次看全。

---

## 7. 版本为什么钉死

实测环境：CPython 3.12.15 + `onnxruntime 1.20.1` + `numpy 2.5.3`（`requirements.txt` 就按这个钉）。

两条独立的事实：

1. `onnxruntime<=1.20` 的 contrib 只有 4-bit `MatMulNBits` → 公开的 `punc.q8w.onnx` 加载不了，
   解法是第 4.2 节那个逆量化建造，**不是**升 ORT。
2. `onnxruntime>=1.21` 在一部分 Windows 机器上 `import` 就崩（0xC0000142 /
   "DLL 初始化例程失败"）。根因是 **VC 运行库**不是 ORT：产线那台机器 `System32` 的
   `vcruntime140.dll` 是 14.23.27820.0（2019 年版），而且**根本没有 `vcruntime140_threads.dll`**，
   新 ORT 的二进制按更新版 CRT 构建就起不来。补上新版运行库（或按绝对路径预加载）就能用；
   实测 1.24.2 / 1.24.4 与 1.20.1 **逐格数值一致**、编码还快 13%。
   三条踩过的死路：把包内 `capi/onnxruntime.dll` 复制到 `python.exe` 同目录（无效）、
   把新版 CRT 丢进 `python.exe` 目录或 PATH（**整个解释器直接被打死**，0xC0000142）、
   先 `os.add_dll_directory(capi)` 再 import（无效）。

所以：`1.20.1` 是"只装包就能起"的唯一版本，分发包钉它。
你的机器如果 CRT 是新的、想让 ORT 版本高一点，把 `requirements.txt` 里那行换掉就行，
但**请先跑一遍对拍**（`asr_chain.py --json` 存逐环节输出再比），别直接换产线配置。

---

## 8. 可选：语种判定（LID）

**默认关**，FireRedASR2 不吃语言标记，LID 只是事后标注；实测给 63.7 s 音频加约 10 s
（单文件、帧级 slice 模式），所以只在真要按语种分流时才开。

要跑得凑齐三样，缺一不可，都不在 `fetch_assets.py` 的清单里：

1. 官方权重 `FireRedLID`：`https://huggingface.co/FireRedTeam/FireRedLID/resolve/main/model.pth.tar`
   （匿名可取）。产线那份实测 3,550,103,418 B、sha256
   `7dee2a280e9b11d5241a0e3d4fa60ee1520a036a2e8385f17960371cfea10093`；同目录配套件
   `cmvn.ark` 1,311 B `6efba6105429d1630c05d818d956bfe4edfad37a04b3b27bb5a029b9adb37945`、
   `dict.txt` 779 B `23ef00941731a24c0b547ab3db69387be2a9d203e4f84a61008a6ab20df1f78b`、
   `config.yaml` 0 B `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`（空文件）。
   **全网没有 FireRedLID 的现成 ONNX**，所以第 2 步必须自己做。
2. 导出：`optional-lid/lid_export.py`（整图编码器）+ `lid_export_dec.py`（解码器单步），
   需要 torch 和官方 `fireredlid` 源码包（`pip install` 装不到，得 clone `FireRedTeam/FireRedASR2S`，
   模块在 `fireredasr2s/fireredlid`）。产线现值（2026-10-06 在校机上算的，供你导出后比对）：

   | 产物 | 字节 | sha256 |
   |---|---:|---|
   | `lid_encoder.onnx` | 2,451,530 | `2c4b12e9166f61ab60de0d60844f25049122b479b7bfd8d123722d5e0ec95327` |
   | `lid_encoder.onnx.data` | 2,893,425,664 | `f93f3dedaf557a861d5e62f37004f1731e3058368dfa4fe1c7d2c28cc21d221f` |
   | `lid_decoder_step.onnx` | 549,729 | `76a40e36d3b09022697869a3d560e606e521a8ed225cdeb8fb00458f9a9e3b84` |
   | `lid_decoder_step.onnx.data` | 655,949,824 | `170e028863620069e97a00e399b4d3b714ec172d112a14a3515aebb7516ed8a1` |

   **这两个导出脚本本轮没在校机上重跑过**（要占 torch 环境、几分钟到几十分钟），所以
   "重跑能对上这四张哈希"还是 **D 级**——上面那行是产线读数，不是你产物的一致性证明。
   另外：这两个脚本走的是 `torch.onnx.export(..., external_data=True)`，也是外置权重。
   torch 那条写入路径是否同样对已存在的 `.data` 追加，本包**没实测**，所以重导出前
   手动删掉旧的 `lid_*.onnx` 与 `lid_*.onnx.data` 更稳（4.1 那个已证实的追加坑出自 `onnx` 的
   `save_model`）。
3. 运行：`optional-lid/lid_onnx.py`，落在 `<models-dir>\FireRedLID-onnx`
   （`--lid-onnx-dir` / `FIREDASR_LID_ONNX_DIR` 可改），单文件用 `asr_chain.py --lid`。

一个容易踩的量纲坑，脚本里已经处理但值得知道：LID 前端吃的是 **int16 幅度**
（官方 `kaldiio.load_mat` 的那个尺度），而 `soundfile` 给的是 [-1,1] float32；
量纲错了会让 log-mel 整体掉到下限，实测同一句话从 `zh mandarin (0.9994)` 变成 `nn (0.3287)`。
`asr_chain.to_lid_scale()` 就是那个乘回 32768 的转换。

---

## 9. 已核 / 未核（照着这张表用，别把 B 当 A）

分级：**A** = 本包内可复算或已逐字节实测；**B** = 产线实测过但不在收信人机器上；
**C** = 只核到局部/间接证据；**D** = 没核。

| 断言 | 级 | 依据 |
|---|---|---|
| `punc.f32.onnx` 由包内 `dequant_punc.py` **逐字节**复现（406,954,097 B，sha `71c54314cb81…`） | A | 与产线文件同哈希，脚本重跑一次即得 |
| 逆量化公式 `(uint8-128)*scale` | A | 单个 MatMulNBits 成图对 8-bit-capable ORT 逐选手算，唯一对上（9.5e-7） |
| f32 产物与 q8w 参照**功能等价** | A | 同一句 101 个子词，logits 最大绝对差 1.5e-5，65/65 类别一致；加完标点逐字相同 |
| `punc.q8w.onnx` 本身与上游 fp32 模型的类别一致率 | **C** | 这是**转传仓自己**在 README 里报的基准（153/153 行、8,798 token 全对），本包**没**复算过；我们只核了"f32 = q8w 的等价逆量化"这一条（上一行，A）。也就是说：q8w 若本身有偏，f32 会一样有偏 |
| 英文整句被小写、只还原 `i`/`i'm`/`i'd`/`i've`/`i'll` 与句首大写 | A | 官方 `RuleBaedTxtFix` 逐字搬运，`asr_chain.py:182-213` 直接读得出来 |
| f32 产物在 `onnxruntime 1.20.1` 下能加载并跑出文本 | A | 产线全链对拍逐字通过 |
| 四环纯 ORT 端到端与参照实现**逐字相同** | A | VAD/ASR/Punc 三段对拍（ASR 与 sherpa 逐字相同，test_wavs 4/4） |
| AED `encoder.f32.onnx` **图本体**与产线同哈希（`1fa3b6c8…`） | A | 已比对 |
| AED f32 那 3.1 GB 外置权重 `.data` 逐字节可复现 | **A**（2026-10-06 09:22 起） | 上一行本条写过 **B**、并给了补法——补法已执行：校机上从 `encoder.int8.onnx` 重跑 `dequant_aed.py`，`.data` 得 `66c3b4f0293c…` 3,103,167,616 B = 产线同哈希；`decoder.f32.onnx.data` 同样得 `7728fe4ac558…` = 产线同哈希。两条哈希现已写进 4.1 表和 `fetch_assets.py`，`--check` 从此**逐字节**核它们，不再只核尺寸。同一轮还查出"重复建造会追加写出两倍大 `.data`"的坑（见 12.4） |
| 整套资源清单 = 产线现场（下载 12 件 + 建造 5 件，共 17 项逐 sha256） | A | 2026-10-06 校机 `--check --graph f32` → 全部就绪、退 0（12.4） |
| **用包内脚本从零造出的那批模型跑端到端，转写与产线模型逐字相同** | A | 2026-10-06 校机：`asr_chain.py --models` 指向 `mt\`（punc.f32 / encoder.f32 / decoder.f32 全是当轮重跑的产物）跑 `test_wavs/0.wav`，退 0；与产线模型的 JSON 比 15 个字段，除墙钟类 `rtf`/`timings` 外 13 个逐字相同，5 段的 `text`/`ids`/边界全对（12.4） |
| 包内 `requirements.txt` 的 7 个钉版本 = 产线在装的版本 | A | 产线 `env\Lib\site-packages\*.dist-info` 逐个对：`numpy 2.5.3 / onnxruntime 1.20.1 / soundfile 0.14.0 / scipy 1.18.1 / tokenizers 0.23.2 / kaldi-native-fbank 1.22.3 / kaldiio 2.18.1` 全一致；解释器 3.12.15 |
| `Send2Trash` 的确切版本 | **C** | 产线那份是 `site-packages\send2trash\` 裸目录、无 `.dist-info`（手工拷入），核不到版本号；包里钉 `2.1.0` 是按 PyPI 现值，不是按产线 |
| 包内 `infer_onnx.py` 与 VAD 卡片 = HF `tardigrade-doc/FireRedVAD_onnx` 上的原件 | A | 2026-10-06 从该仓直接取回算哈希：`dda73c30f190956e` 17,442 / `cb1c61f627021d26` 270，两份都与产线相同 |
| 旧文档写的 VAD 仓 `jiangzhuo9357/fireredvad-onnx` 取不到文件 | B | 2026-10-06 经校机代理请求 `…/resolve/main/infer_onnx.py` 与 `…/README.md` 都 **401**；这是当天该台机器的网络读数，不是"该仓不存在"的永久断言（12.5 的更正据此） |
| LID 路线（`--lid`）造出的两个图与产线那四件逐字节相同 | **D** | 包内有 `lid_export.py` / `lid_export_dec.py`，产线有权重与导出件（第 8 节给了哈希），但**本轮没重跑导出**（要 torch + 官方源码 clone），所以可复现性未核。默认链路不开 LID，不影响上面任何一条结论 |
| `mixed` 档 ASR 层 RTF 0.361（int8 0.491、f32 0.424、整条不分段 1.411） | B | 24 vCPU 那台机器、425.09 s 音频、18 线程实测；**你的机器数字一定不同** |
| 12 线程与 18 线程同速且 CPU −33%；五档线程数转写文本 md5 一致 | B | 同一台机器、**一条** 603.6 s / 248 段音频、每档跑一次、空载（见 4.3 表）；单条音频 ⇒ 不保证别的音频也同速 |
| VAD 封顶 8 线程只省总 CPU 约 16 秒、VAD 环节 1.0 s → 0.9 s | B | 同上那台机器；这一条的收益在同机共存，不在吞吐 |
| f32 标点比 q8w 快 2.9 倍 | B | 同一台机器，4 线程、101 位置 |
| 三个模型文件匿名可下载 | A | 2026-10-06 逐条 HEAD 200 + 包内脚本实下载验收过 |
| ORT≥1.21 崩的根因是 VC 运行库 | A | 那台机器 System32 三个读数 + 预加载后 1.21–1.24 全正常的矩阵 |
| 熔断/退出码/`--status` 行为 | A | 本地沙箱实测（假成功修的那一轮） |
| 退出码 6 的三条守卫（不认识的参数 / 非法枚举值 / 缺参数值）都退 6 且**不**撞 2 | A | 2026-10-06 本地实测；改前 argparse 默认退 2 = "引擎加载失败" |
| `--input-dir` 打错时退 6、**不**替你建空目录（改前会建出来→扫到 0 个→退 0 像"转完了"） | A | 2026-10-06 本地实测；`--status` 同一路径仍 0 并标 `(不存在)` |
| `asr_chain.py --models` 会把 VAD/Punc 派生目录一起搬走 | A | 2026-10-06 校机实测：改前只搬 ASR/LID，断句与标点仍读旧目录 → 旧目录无 `infer_onnx.py` 时直接 `ModuleNotFoundError`；改后冒烟退 0（12.4.1） |
| 整包在**没装过这套链**的机器上从建环境到出文本全跑通 | A（**旧版 `setup.bat`**） | 2026-10-06：建 venv → `fetch_assets --graph int8`（1.68 GB 常驻，模型根实测 1,809,128,533 B）→ 6 个 wav 的脏队列（成功 4 / 空转写 1 / 失败 1，退 1，坏文件按相对路径进隔离区）→ 再跑 5 个干净文件（退 0）。**注意范围**：那一轮跑的是逻辑写在 bat 里的旧版；`setup.py` 是把同一套步骤搬进 python 的新件，**没有**再跑过一次（下一行） |
| `setup.py` / 新版 `setup.bat` 跑通 | **D** | 只做了静态检查（`ast.parse` 过、help 串里无裸 `%`、bat 纯 ASCII + CRLF）。想核的人一条命令：`python setup.py --skip-check`（它只建 venv 和装依赖，不下模型） |
| 钉版本在 Linux x86_64 / aarch64 有轮子、`requires_python` 与下载字节数 | A（2026-10-06 取 PyPI 官方 JSON `pypi.org/pypi/<包>/<版本>/json`） | 8 件运行侧 + `onnx==1.23.1` 逐个查：都有 `manylinux_*_x86_64` 与 `manylinux_*aarch64`（`kaldiio`/`Send2Trash` 是 `py3-none-any`）；numpy/scipy 标 `>=3.12`，ORT 1.20.1 最高 cp313；下载量 win 72.7 / x86_64 79.3 / aarch64 75.4 MB |
| 驱动那四处 OS 分支的**写法**（锁、后台、按键、回收站） | A | 逐行读得出来：`xhs-chain-cpu.py:295/299/320`、`:365-370`、`:880/883`、`:542-547`。这是"代码里有这条分支"级别，不等于下一行 |
| **整条链在 Linux/macOS 上跑通** | **D** | 没有 Linux 机器可跑：产线和本机都是 Windows。上面两行只证明"轮子存在 + 分支写了"。第一次在 Linux 上跑请按 3.3 的表逐条看，`--check` 和 `setup.py` 都不碰 OS 特殊路径，出问题先报这两步的原文 |
| beam 解码能追上 greedy 的质量 | **D** | 产线**没调通**：beam(B=3) 慢 6 倍以上且截断+输出退化，留档不推荐 |

`fetch_assets.py` 的每一次运行都会把这张表里"文件"那部分变成你机器上的现场读数（`--check`），
其余档位性能请自己复测。

---

## 10. 许可与隐私

* **三个模型家族各自的出处，以及本包核到了哪一步**（查不动的就写"未核"，不猜）：

  | 环 | 本包实际取的地方 | 它自己声明的上游 | 核到哪一步 |
  |---|---|---|---|
  | VAD | HF `tardigrade-doc/FireRedVAD_onnx`（4 件二进制 + `infer_onnx.py` + 模型卡片，共 6 件；前 4 件走 `fetch_assets.py`，后 2 件随包） | 卡片 frontmatter：`base_model: FireRedTeam/FireRedVAD`、`license: apache-2.0` | 六件都能匿名取到，随包的 `infer_onnx.py`（`dda73c30f190956e`）和卡片（270 B / `cb1c61f627021d26`）2026-10-06 与仓库逐字节对过（第 9、12.5 节）。**转传仓是否官方运营没核** |
  | Punc | HF `jiangzhuo9357/fireredpunc-onnx`（`punc.q8w.onnx` 等 5 件） | 仓里 `README.md` 逐条写了：权重上游 `FireRedTeam/FireRedPunc`（Apache-2.0）、代码上游 `FireRedTeam/FireRedASR2S` @ commit `4e7d9aaf` 的 `fireredasr2s/fireredpunc`、底模 `chinese-lert-base`、这份 q8w 是作者用 `MatMulNBitsQuantizer`（8 bit / block 32 / 对称）从 fp32 重导的，**不是** `42ailab/FireRedPunc-ONNX` 那个动态 int8 文件 | `LICENSE` 逐行读过 = Apache-2.0 标准正文（11,357 B / 201 行）；那份 `README.md`（4,227 B）也随 `fetch_assets.py` 落盘，你机器上直接能看到。**上游 HF 页面本身没去对**（转传仓自述，属 C 级） |
  | AED | GitHub `k2-fsa/sherpa-onnx` 的 release 资产 `sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26.tar.bz2` | sherpa 侧只说是 FireRedASR2 的 int8 转换；**底模（小红书 FireRedTeam FireRedASR2A）的许可原文本包没有核实** | 未核。商用前自己查 `FireRedTeam/FireRedASR2S` 的模型卡与仓库 LICENSE |

  顺带一条能自证的：Punc 输出里英文会被整体小写，只有 `i`/`i'm`/`i'd`/`i've`/`i'll`（行首与句中）
  和句首字母被还原成大写——这是**官方** `RuleBaedTxtFix` 的行为，本包逐字搬的就是它，
  对着 `asr_chain.py:182-213` 一眼能核（转传仓 README 也独立提到同一现象）。
* **不随包发的东西**：
  * 产线的 `rules.txt`（里面是具体错词表）。包内只有格式示例 `rules.example.txt`。
  * ntfy 推送地址：`--ntfy-url` 默认空串 = 不推。那是个人订阅端点，属凭据，
    不给就不推，代码里也没有硬编码。
  * 任何机器专属路径 / 账号 / 远端端口 / SSH 信息。出厂副本里这类东西一律不留：
    路径全部走环境变量或命令行参数，默认值相对本包目录。拿到包后自己复扫一遍
    （PowerShell 两条；第二条里的 `<账号名>` 换成你那台机器的登录名）：

    ```powershell
    Get-ChildItem -Recurse -Include *.py,*.md,*.txt,*.bat,*.json |
      Select-String -Pattern '[A-Za-z]:[\\/]'          # 任何盘符开头的绝对路径
    Get-ChildItem -Recurse -Include *.py,*.md,*.txt,*.bat,*.json |
      Select-String -Pattern 'serverchan|ntfy\.'        # 推送端点
    Get-ChildItem -Recurse -Include *.py,*.md,*.txt,*.bat,*.json |
      Select-String -Pattern '\b(10|17[2-9]|19[0-2])\.\d'   # 内网段
    Get-ChildItem -Recurse -Include *.py,*.md,*.txt,*.bat,*.json |
      Select-String -SimpleMatch -Pattern "<账号名>"   # 换成你自己那台的登录名
    ```

    2026-10-07 出厂副本的读数（本包全部 `.py`/`.md`/`.txt`/`.bat` 加 `.gitignore`、
    `.gitattributes`，逐行核对；本包没有 `assets.json`，资源清单写在 `fetch_assets.py` 里。
    命中数**不含**本节这段扫描说明自己的行——那几条命令的文字本身会把模式再打一遍，属自指）：

    | 模式 | 包内命中 | 是什么 |
    |---|---|---|
    | 盘符开头的绝对路径 | **0** | 一个都没有；原先那种盘符示例占位也全换成 `.\audio`、`.\m` 这种相对写法 |
    | 内网段、`serverchan`、推送端点、账号名、私钥文件名 | **0** | 一个都没有；这些字串只出现在上面那几条示例命令的模式文字里 |
    | `127.0.0.1:7890` | 5 | `--proxy` 的**写法示例**：§4 那行、`fetch_assets.py:18/205/388`、`setup.py:165`。回环地址，任何 mihomo/clash 的默认 mixed-port 都长这样，不是某台机器的真实入口；本包不含任何代理地址、订阅或凭据 |

    命中行里的中文在 chcp 936 的控制台可能显示成乱码，是显示问题，行号可信。

    **一处公开更正**：这一节原先写"真实盘符目录、登录账号名…全 0 命中"，但 §12.4 那轮对拍当时
    带着产线机器上"账号名 + 临时目录"的真路径（某盘符下 `\…\tmp\…` 那种形态，5 处）——那是补对拍记录
    时漏掉的自相矛盾。10-06 下午已把那五处换成 `<验证根>` / `<产线模型根>` 占位，真实路径只留在
    本地记录里，所以第一行那格现在确实是 0。

---

## 11. 排障

| 现象 | 先查这里 |
|---|---|
| `Only 4b quantization is supported` | ORT ≤1.20 读 q8w。跑 `fetch_assets.py` 造 `punc.f32.onnx`，或 `--punc-file` 指到 q8w 并换 ORT≥1.21 |
| `import onnxruntime` 就崩 / 0xC0000142 | VC 运行库太旧（第 7 节）。要么钉回 1.20.1，要么补新 CRT |
| 启动就退出码 4 | 锁还留着：`--status` 看在跑的是谁；确认没活的实例后删 `<out-dir>\.lock`（**别**在实例活着时删） |
| 退出码 5 | 输入根不存在 / 扫不到任何匹配 `--extensions` 的文件。这不是"转完了" |
| 一批全失败、退出码 3 | 环境故障（模型缺、盘满、目录权限）。看 `--fail-dir` 里第一批文件和日志最后 20 行 |
| 长文件转出来截断 | `--cache` 上限：AED 解码 cache 会截断过长段，试 `--cache max` |
| 复读刷屏的坏案 | 已内置复读折叠；再补 `--rules` 词表（格式见 `rules.example.txt`） |
| 日志里中文变乱码 | 后台启动时 stdout 是文件，脚本已强制 UTF-8；若你用 `type` 看仍乱码，是控制台代码页，`chcp 65001` |
| Linux 上 `python3 -m venv` 报 `ensurepip is not available` | 发行版把 venv 拆成单独的包：`sudo apt update && sudo apt install -y python3-venv python3-pip`（`setup.py` 失败时打的也是这句） |
| Linux 上 `import onnxruntime` 报 `GLIBC_2.28 not found` | 这台机器的 glibc 低于钉版本轮子的标签（第 3.2 节）。换系统或自己编 ORT，本包不负责 |
| Linux 上 Ctrl+N 没反应 | 那是 `msvcrt` 实现的，只有 Windows 有；停止用 `--stop` 或 Ctrl+C（3.3 表） |
| 想确认资源齐不齐 | `python fetch_assets.py --check`（只读，逐件报 就绪/缺/坏） |

---

## 12. 这份包是怎么来的

从产线目录整份复制正在跑的脚本，然后：删掉所有机器专属路径 → 每一项改成
"CLI > `FIREDASR_*` 环境变量 > 相对脚本目录的默认值" → 大文件全部改成
"下载 or 建造"并逐件写进 `fetch_assets.py` 的清单（哈希、字节数都在产线目录实测）→
`setup.py` 做环境（Windows 上 `setup.bat` 只是转发它）、`fetch_assets.py` 做资源、
`xhs-chain-cpu.py` 做转写。

行为与产线唯一的差别就是"没有默认凭据/没有默认词表"（第 10 节）；其余判据、
退出码、结算规则一致。建造脚本（`dequant_*.py`）本来就在产线上，这里只是把它们的位置
从"某台机器的 tools 目录"挪进了包内。

### 12.1 与产线逐件对账（2026-10-06，sha256 前 16 位 + 字节数）

| 文件 | 产线 | 包内 | 判定 |
|---|---|---|---|
| `aed_ort.py` | `2d6018194602eb06` 13,241 | `2d6018194602eb06` 13,241 | **逐字节相同** |
| `models/fireredvad-onnx/infer_onnx.py` | `dda73c30f190956e` 17,442 | `dda73c30f190956e` 17,442 | **逐字节相同**（也是上游原文件） |
| `dequant_aed.py` | `edfbed1c0d42a801` 8,148 | `b0637acb0979dd6e` 8,720 | 差异 **+9 行**：保存前删掉上一轮的 `out` 与 `out.data`（12.4 查出的追加 bug）。产线那份没有这段——它只在全新目录里建过一次，没碰到这个坑；若要在产线原地重建，先手动删掉同名 `.data` |
| `asr_chain.py` | `851da03fb8c70ee3` 18,477 | `5ec36d4e60bcd443` 22,141 | 差异 **+78 / −15 行**：全部是把写死的路径/线程/档位换成参数 + 单文件 CLI 加了 `--lid`，判据与后处理（复读折叠、`RuleBaedTxtFix` 原样）未动。其中 12.4 查出的 `--models` 只搬 ASR、不搬 VAD/Punc 那个 bug 占 +7 行 |
| `xhs-chain-cpu.py` | `428735e499b9fc20` 31,695 | `c55c8d2d7948b52f` 43,517 | 差异 **+361 / −168 行**：参数化 + 退出码诚实化（熔断回搬、结算、`--status/--stop/--start`）+ 第 10 节说的"凭据与路径清空"。两侧现在都有那两处退出码修正（产线 13:05 合入，`_Parser` 那条在产线落成"拼错参数退 6"的一行改动），也都有 4.3 的线程结论，但形态不同：产线是写死的 `ASR_THREADS = 12` / `VAD_THREADS = 8`，包内是默认值 `0.5×逻辑CPU` / `min(8, 逻辑CPU)` + `--threads` / `--vad-threads`（后者填 0 关闭封顶）。产线那次的实测表在 4.3，包内多出来的封顶逻辑另经 4 例无模型单元核对（`_orig_ss` 工厂在建完必还原、已显式设过线程数的 session 不被覆盖）。**10-06 下午 Linux 那一轮**只动了文本和打印：文件头说明改成"Windows / Linux / macOS"、13 条 help 里的示例路径由 `\` 改 `/`、两处 `§4.2` 引用改成 `§4.3`、`--start` 完成后那两行提示按平台分开打（Windows 保留"别关启动它的窗口"，POSIX 改说"已 setsid、关终端不带走"）——**判据与主循环逻辑一行未动，产线上不需要跟着换** |
| `dequant_punc.py` | `1fba75c34f435c61` 7,893 | `631241ce963cfb21` 8,408 | 包内是重写版（docstring、边界处理、可被 `fetch_assets.py` 当模块调用）。**产物逐字节相同**：干净机上两次（含 `--only punc_f32 --force`）都造出 `71c54314cb8129e4…` = 产线 `punc.f32.onnx` |

产线上还有两件**没**进包：一版更早的标点转换器（`tools/punc_convert/q8_to_f32.py`，
`dc1cf5a6790db918` 4,389 B）——功能被 `dequant_punc.py` 覆盖；以及产线的 `rules.txt` 词表。

### 12.2 这个配方验过什么（2026-10-06，在一台**没跑过这套链**的开发机上一次性做的）

这一节只证明"照着 §3–§5 三步能在干净机器上把资源补全、把链跑起来、退出码按文档兑现"。
它**不是**性能基线（那是第 4.1 表那台 24 vCPU 机器的数字），也不是你那边必须重复的动作——
`fetch_assets.py --check` 一条就够。

1. `setup.bat` → 建 `.venv`、装依赖、导入自检，全过（当时这些逻辑还写在 bat 里，且那版 bat
   是 CRLF + 纯 ASCII；10-06 下午整体搬进了 `setup.py`，新壳没跑过，见第 9 节那两行 D）。
2. `fetch_assets.py --graph int8` → 实下载 5 件 + 解包 AED tar + 建造 `punc.f32.onnx`，
   逐件 sha256 对上了；模型根实测常驻 1,809,128,533 B = 1.68 GB（和 4.1 末表一致）。
3. 建造确定性：`--only punc_f32 --force` 重造，输出哈希与产线相同。
4. 全链转写：6 个 wav 的队列 → 成功 4 / 空转写 1 / 失败 1 → **退 1**，坏文件按相对路径
   进了隔离区；再来 5 个干净文件 → 退 0；`--graph mixed` 而 f32 编码器缺件时 →
   `[CRITICAL] 引擎加载失败` + **退 2**。日志里每条 RTF 都在 0.5–0.75（8 线程、
   16 逻辑核的机器，别和 4.1 表里 24 vCPU 的数字混）。
5. 下载中断复原：故意让 `punc.q8w.onnx` 截断（162,126,326 / 162,771,205 B）→
   旧版会把半截文件留在 `.part` 上直接失败；现在按 `Range` 续下并逐件验收，重跑即成。
6. 退出码守卫（这一轮补的，见第 5 节的 6）：不认识的参数 / 非法枚举值 / 缺参数值 → 6；
   `--input-dir` 指向不存在的目录 → 6 **且不建那个目录**；同一路径下 `--status` 仍 0 并标
   `(不存在)`，`--stop` 正常落 STOP 标志并 0。
7. 编码：`[致命]` / `[参数不对]` 这类早于 logger 的输出，现在和日志同一条码（UTF-8），
   重定向到文件不再被系统代码页（这台是 cp936）改成乱码。

### 12.3 刻意保留的两处"不统一"

* `dequant_aed.py` 的中文输出在某些重定向场景下要靠调用侧
  （`fetch_assets.py` 给它 `PYTHONIOENCODING=utf-8`）来兜，脚本自己不做 reconfigure。
  它原本与产线逐字节相同，本轮为修 12.4 那个追加 bug 加了 9 行，其余一字未动。
* 收信人环境里 ORT≥1.21 且能 import 时，`--punc-file` 指回 `punc.q8w.onnx` 可以不造 f32
  （第 4.2 节），但默认路线仍造，因为 f32 快 2.9 倍。

### 12.4 与产线的逐字节对拍（2026-10-06 09:16–09:52，在校机上做的）

这一节回答的是"照着这个包能不能造出一台和我一样的机器"。做法：把包内当时那些文本文件
（现在的全量清单见第 2 节）推到校机上另建的一个临时目录，下面记作 `<验证根>`（真实路径只留在
本地记录里，按第 10 节"不含账号与机器专属路径"的口径不进包），推上去的副本与本地逐个 sha256
相同；用**产线那个资源清单**跑 `--check`，把三件建造件在隔离目录 `<验证根>\mt\` 里从 int8/q8w
原料重跑一遍，最后拿 `mt\` 当模型根跑一条音频，与产线模型跑出来的同一条结果逐字段比。

| 动作 | 结果 |
|---|---|
| `fetch_assets.py --models <产线模型根> --graph f32 --check` | 17 项**全部就绪**、逐条 sha256 对上（含两张 `.data`）；退出 0；耗时 5.5 s |
| 环境依赖：`env` 里装的版本 vs `requirements.txt` 的钉法 | `numpy 2.5.3 / onnxruntime 1.20.1 / soundfile 0.14.0 / scipy 1.18.1 / tokenizers 0.23.2 / kaldi-native-fbank 1.22.3 / kaldiio 2.18.1` —— **7/7 一致**；解释器 3.12.15 |
| `Send2Trash` | 产线是 `site-packages\send2trash\` 一个裸目录、**没有 `.dist-info`**（手工拷的），所以钉不回确切版本；包里钉 `2.1.0` |
| 重跑 `dequant_punc.py`（q8w → f32） | 388.1 MB，sha `71c54314cb81…` = 产线 ✓（用时 4–7 s） |
| 重跑 `dequant_aed.py`（decoder int8 → f32） | 图 `52bd0efacb0f…` ✓、`.data` `7728fe4ac558…` ✓（用时 14 s） |
| 重跑 `dequant_aed.py`（encoder int8 → f32） | 第一次：图 `1fa3b6c85031…` ✓、`.data` `66c3b4f0293c…` ✓（用时 33 s）；**第二次在同一目录重跑 → 图 957,649 B、`.data` 6,206,335,232 B，与产线不符**（4.1 讲的追加 bug），`--force` 的门禁当场把它判死；加上修复后在同一（已污染的）目录第三次重跑 → 图 956,747 B / `.data` 3,103,167,616 B，两个哈希都回到产线值 ✓（用时 94 s） |
| 端到端冒烟：`asr_chain.py --models <验证根>\mt`（全套模型都是本次从零造出来的那批）跑 `test_wavs/0.wav`，档位 `mixed` | 退出 0，10.1 s 音频；与同一文件用**产线模型**跑出来的 JSON 比对：**15 个字段里除 `rtf`、`timings` 两个墙钟字段外 13 个逐字相同**（含 `text_raw`、`text_punc`、`audio_dur`、`speech_dur` 和 5 段的全部 `start_s/end_s/text/ids`，逐段 `ids` 6/4/3/7/4 一致）|

结论：**下载件的哈希来自产线实测，建造件的逐字节可复现也在校机上实测过，而且用这批从零造出来的
模型跑端到端能得到与产线相同的转写**——两边是同一套文件、同一条链。`--graph mixed`（默认）需要的
件全在这张表里。差异只剩两处非文件项：解释器小版本（产线 3.12.15，干净机 3.13.11 也跑通，见第 3 节）
和 `Send2Trash` 的来源方式。

校机上留了约 5 GB 测试产物在那个 `<验证根>`（`mt\` 里是本次重跑的三份 f32），没动过产线目录；
要不要清由你定，一条命令 `rmdir /s /q <验证根>`。

### 12.4.1 这一轮从对拍里查出来的两个 bug（都已修在包内）

* **`.data` 追加**：`onnx.save_model(..., save_as_external_data=True)` 对**已存在**的外置权重文件是
  追加不是覆盖，所以在同一目录原地重建会得到一份"图里 offset 指向第一轮、第二轮数据糊在后面"的
  文件（实测 6.2 GB 那份）。修法是保存前删掉上一轮的 `out` 与 `out.data`（见 4.1）。
* **`--models` 只搬了 ASR**：`asr_chain.py` 里 `--models` 原来只重算 `MODELS` 和两个 LID 目录，
  `VAD_DIR`/`PUNC_DIR` 还停在导入时算好的"脚本旁 models"。表现是断句和标点去读旧目录，旧目录没有
  那份推理脚本就直接 `ModuleNotFoundError: No module named 'infer_onnx'`（校机实测）。上面的冒烟
  就是加了 `--models` 的 VAD/Punc 派生之后才跑起来的。批量驱动 `xhs-chain-cpu.py` 不受影响——它是
  import 之后再逐个给全局赋值，本来就覆盖到了；环境变量那条路（`FIREDASR_MODELS`）也不受影响，
  因为派生发生在导入时。

### 12.5 一处公开更正

上一版文档（校机侧的 `school-asr-onnx\README.md` 第 71 行）把 VAD 的来源写成
HF `jiangzhuo9357/fireredvad-onnx`。**那个地址在校机上请求返回 401，取不到任何文件**，
而 `tardigrade-doc/FireRedVAD_onnx` 上 `model.onnx`/`cmvn.bin`/`cmvn.json`/`config.json`/
`infer_onnx.py`/`README.md` 六件都能匿名取到。本轮直接取回并算哈希的是后两件
（`infer_onnx.py` = `dda73c30f190956e` 17,442 B、卡片 = `cb1c61f627021d26` 270 B），
与产线那两份逐字节相同；前四件走的是同一条链：仓库取回 → 命中 4.1 清单的 sha →
而 4.1 清单的 sha 又等于产线文件（12.4 的 `--check` 逐条核过）。
所以 VAD 的正确出处是 `tardigrade-doc/FireRedVAD_onnx`，旧文档那行已改掉。
