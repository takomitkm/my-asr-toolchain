r"""
FireRed ASR2 · 纯 ONNX 全链批量转写驱动（Windows / Linux / macOS · CPU）

引擎链：FireRedVAD（判段）-> FireRedASR2-AED（纯 onnxruntime，可选 int8/f32/mixed 图，贪心）
        -> FireRedPunc（逐段加标点）。
不需要 torch，不需要 sherpa-onnx，不需要 GPU。

数据面契约（跑完一个文件才动那个文件，全程可断可续）：
  · 输入   --input-dir 递归扫音频（--extensions 决定后缀），按第一级子目录分组，
    组内按 mtime 升序读取（时间顺序）；根目录散文件归入组名 "p"
  · 输出   每组追加写 <out-dir>/<组名>.txt；一条记录 = "title:<相对输入根的路径>"
    + 正文 + 空行（含子目录时分隔符取本机 os.sep：Windows 是反斜杠，Linux/macOS 是正斜杠）
  · 结算   成功/空转写 -> 按 --on-done 处理源文件（trash=回收站 / delete=直接删 /
    keep=不动）；失败 -> 按相对路径隔离到 --fail-dir
  · 连续 --abort-consec-fail 个文件一个都没成功 = 判为环境故障，当场熔断：本段被隔离的
    文件全部搬回输入目录，剩余文件原样留着，退出码 3（绝不报"转完了"）
  · 单实例锁 <out-dir>/.lock；STOP 文件 / Ctrl+C（一次=当前文件跑完停，两次=立即停）
  · --rules 指向"原词=替换词"规则文件（缺文件=不清洗，不报错）；幻觉过滤 =
    复读折叠 + 相邻重复行去重；队列快照写 --tmplist；日志写 <log-dir>/*.log（UTF-8）
  · --ntfy-url 给了才推进度/结算（默认不推，见 README 的隐私说明）；Ctrl+N 临时静音（仅 Windows）

引擎档位（2026-10-05 在一台 24 vCPU 机器上实测）：
  · 最快 = VAD 分段（FireRedVAD 自带 20 s 段上限）+ mixed 图（f32 编码器 + int8 解码器）
    + 贪心 + 18 线程：ASR 层 RTF 0.361；纯 int8 0.491、纯 f32 0.424、整条不分段 1.411
  · 标点环按官方口径逐段跑（每个 VAD 段单独进 FireRedPunc，不做分块；BERT 512 token
    上限内不会炸），段文本拼成整条写进记录
  · 不跑语种判定：FireRedASR2 不吃语言标记，LID 只是事后标注，单文件用法见 asr_chain.py --lid
  · 模型进程内只加载一次、跨文件复用；无外部子进程

所有路径与档位都能从外部指定：命令行参数优先，其次环境变量 FIREDASR_<同名>，最后是默认值
（默认值全部落在脚本目录内，不含任何机器专属路径）。--help 看全部参数。

用法示例：
  python Oc-fired.py --help
  python Oc-fired.py --data-root .\asr --models-dir .\asr\models            前台跑
  python Oc-fired.py --data-root .\asr --start                              后台跑（参数原样带过去）
  python Oc-fired.py --data-root .\asr --stop                               下一个文件边界优雅停止
"""

import argparse
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# 后台启动（--start）时 stdout / stderr 是文件不是控制台：Python 会改用系统代码页
# （中文 Windows 是 cp936）编码，中文提示写进 UTF-8 日志就成乱码，文件名里 GBK 表示不了
# 的字符还会让 logging 抛 UnicodeEncodeError。放在最前面，[致命] / [参数不对] 这些
# 早于 logger 的输出也走同一条码。真控制台（isatty）不动，那条路径 Python 走
# WriteConsoleW 本来就正确。
for _s in (sys.stdout, sys.stderr):
    try:
        if _s is not None and not _s.isatty():
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ==================== 外部参数：CLI > 环境变量 FIREDASR_* > 默认 ====================
#
# 默认值一律相对本脚本所在目录，所以整包放哪儿都能跑；--data-root 是一个总根，
# 下面这些子目录默认从它派生，想分开放就各自再给一个参数。

SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    ap = _Parser(
        prog="Oc-fired.py",
        formatter_class=_Formatter,
        description="FireRed ASR2 纯 ONNX 全链批量转写驱动（VAD -> AED -> Punc）",
        epilog="退出码：0 队列转完 / 1 有文件失败 / 2 引擎加载失败 / 3 连续失败熔断 / "
               "4 已有实例在跑 / 5 输入根读不到（不是转完了）/ 6 参数或环境不对",
    )
    # ---- 路径 ----
    g = ap.add_argument_group("路径（不给就用 FIREDASR_* 环境变量，再不给就用括号里的默认）")
    g.add_argument("--data-root", metavar="DIR",
                   help="数据总根，默认 <脚本目录>/data")
    g.add_argument("--input-dir", metavar="DIR",
                   help="待转音频根目录，默认 <data-root>/p")
    g.add_argument("--root-group", metavar="NAME",
                   help="输入根下散文件（不在任何子目录里）归到哪个分组，默认取输入目录名")
    g.add_argument("--out-dir", metavar="DIR", help="转写 txt 目录，默认 <data-root>/txt")
    g.add_argument("--fail-dir", metavar="DIR", help="失败隔离区，默认 <data-root>/f")
    g.add_argument("--log-dir", metavar="DIR", help="日志目录，默认 <out-dir>/log")
    g.add_argument("--rules", metavar="FILE", help="规则文件（原词=替换词），默认 <out-dir>/rules.txt")
    g.add_argument("--tmplist", metavar="FILE", help="队列快照文件，默认 <out-dir>/tmplist.txt")
    g.add_argument("--lock-file", metavar="FILE", help="单实例锁，默认 <out-dir>/.lock")
    g.add_argument("--stop-file", metavar="FILE", help="停止标志文件，默认 <out-dir>/STOP")
    g.add_argument("--code-dir", metavar="DIR",
                   help="asr_chain.py / aed_ort.py 所在目录，默认脚本所在目录")
    g.add_argument("--models-dir", metavar="DIR",
                   help="模型目录（里面有 fireredvad-onnx / fireredpunc-onnx / sherpa-onnx-fire-red-asr2*），"
                        "默认 <脚本目录>\\models")
    g.add_argument("--asr-dir", metavar="DIR",
                   help="AED 模型目录，默认在 <models-dir> 下按 sherpa-onnx-fire-red-asr2* 自动找")
    # ---- 引擎档位 ----
    g = ap.add_argument_group("引擎档位")
    g.add_argument("--threads", type=int, help="ASR 推理线程数，默认 max(1, 逻辑CPU*0.5)"
                                              "（产线那台 24 vCPU 的读数：12 与 18 同速、CPU 省 33%%，见 §4.3）")
    g.add_argument("--graph", choices=("mixed", "int8", "f32"),
                   help="ASR 计算图：mixed=f32编码器+int8解码器（最快，需 encoder.f32.onnx）；"
                        "int8=官方 int8 图（省内存）；f32=全 f32（最吃内存）")
    g.add_argument("--asr-mode", choices=("greedy", "beam"), help="AED 解码方式")
    g.add_argument("--cache", metavar="auto|max|N",
                   help="AED 解码器 cache 长度（段太长会被截断）")
    g.add_argument("--punc-threads", type=int, help="标点环节线程数，默认 4")
    g.add_argument("--vad-threads", type=int,
                   help="断句（VAD）环节的推理线程上限，默认 min(8, 逻辑CPU)；填 0 = 不设上限。"
                        "VAD 建 session 时本来没设 intra_op_num_threads，会瞬时吃满所有物理核，"
                        "在同机共存时体感很明显（省不了多少总时间，见 §4.3）")
    g.add_argument("--no-punc", action="store_true", help="不跑标点环")
    g.add_argument("--vad-dir", metavar="DIR",
                   help="FireRedVAD 目录，默认 <models-dir>/fireredvad-onnx")
    g.add_argument("--punc-dir", metavar="DIR", help="FireRedPunc 目录，默认 <models-dir>/fireredpunc-onnx")
    g.add_argument("--punc-file", metavar="FILE",
                   help="指定标点模型文件（默认优先 punc.f32.onnx，缺则 punc.q8w.onnx）")
    g.add_argument("--min-seg-sec", type=float, help="短于此秒数的段/文件直接跳过，默认 0.1")
    # ---- 队列与结算 ----
    g = ap.add_argument_group("队列与结算")
    g.add_argument("--extensions", metavar=".wav,.mp3,...",
                   help="认作音频的后缀集合，逗号分隔（可带点可不带）")
    g.add_argument("--on-done", choices=("trash", "delete", "keep"),
                   help="转写成功后怎么处理源文件：trash=回收站（需 Send2Trash）/ "
                        "delete=直接删 / keep=不动（重复跑会重复转写）")
    g.add_argument("--notify-every", type=int, help="每处理这么多文件推一条进度，0=不推")
    g.add_argument("--abort-consec-fail", type=int,
                   help="连续失败多少个文件就熔断，0=不熔断（强烈不建议）")
    g.add_argument("--repeat-filter-max-line", type=int,
                   help="超过此长度的行跳过带反向引用的正则，规避回溯爆炸")
    g.add_argument("--ntfy-url", metavar="URL",
                   help="ntfy 推送地址（给了才推，默认不推）。注意：这是个人订阅地址，属凭据")
    g.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="日志级别")
    # ---- 动作 ----
    g = ap.add_argument_group("动作")
    g.add_argument("--start", action="store_true", help="后台启动（其余参数原样传给后台实例）")
    g.add_argument("--stop", action="store_true", help="让在跑的实例在下一个文件边界退出")
    g.add_argument("--status", action="store_true", help="看当前锁与队列规模，不改任何东西")
    return ap


class _Formatter(argparse.ArgumentDefaultsHelpFormatter):
    r"""默认值是派生的（<data-root>\p 之类），help 里已经写清楚了，别再追加 (default: None)。"""
    def _get_help_string(self, action):
        h = argparse.HelpFormatter._get_help_string(self, action)
        if action.default in (None, "", False) or h is None:
            return h
        return h + f"（默认 {action.default}）"


class _Parser(argparse.ArgumentParser):
    r"""argparse 的 error() 自己退 2，而 2 在本脚本里是"引擎加载失败"。
    参数不对就是"参数或环境不对"，退 6，和 README 的退出码表一致。"""
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        sys.exit(6)


def _pick(args, name, default):
    """CLI 给了用 CLI；否则用环境变量 FIREDASR_<NAME>；否则用默认值。"""
    v = getattr(args, name, None)
    if v not in (None, ""):
        return v
    e = os.environ.get("FIREDASR_" + name.upper())
    if e:
        return e
    return default


def _apply_config(args) -> dict:
    data_root = Path(_pick(args, "data_root", str(SCRIPT_DIR / "data")))
    out_dir = Path(_pick(args, "out_dir", str(data_root / "txt")))
    code_dir = Path(_pick(args, "code_dir", str(SCRIPT_DIR)))
    models_dir = Path(_pick(args, "models_dir", str(SCRIPT_DIR / "models")))

    exts = _pick(args, "extensions", ".mp3,.m4a,.mp4,.wav,.oga,.ogg,.opus,.flac,.aac")
    if isinstance(exts, str):
        exts = {("." + e.strip().lstrip(".").lower()) for e in exts.split(",") if e.strip()}

    # 2026-10-06 在 24 vCPU 上拿一条 603.6 s / 248 段的真实音频逐档实测（机器空载，每档跑一次）：
    #   18 线程 wall 96.8 s / 1824 CPU 秒；12 线程 wall 94.5 s / 1216 CPU 秒 —— 同速但 CPU 省 33%，
    #   说明 0.75 那个系数是超配（多出的线程在自旋和同步屏障上空转）；8 线程 +6.8% 时间、CPU 省 52%。
    #   五档转写文本 md5 完全一致 —— 降线程不改变输出。所以默认取 0.5×逻辑 CPU（这台机器上就是 12）。
    thr_default = max(1, int((os.cpu_count() or 4) * 0.5))
    # VAD 环建 session 时没设 intra_op_num_threads，ORT 默认吃满所有物理核，而 VAD 是每个文件
    # 开头对整条音频一次性提特征 —— 会瞬时点满机器。产线那台 24 vCPU 上封顶取 8（对方的读数是
    # vad_wall 1.0s→0.9s、CPU 总量只省 16 秒，即收益在同机共存体感上、不在账上；我没有独立复现）。
    # 核数不足 8 的机器写死 8 反而超订，所以默认取 min(8, 逻辑CPU)——在她那台机器上算出来仍是 8。
    vad_thr_default = min(8, max(1, os.cpu_count() or 4))
    input_dir = Path(_pick(args, "input_dir", str(data_root / "p")))
    # 散文件的分组名：默认跟着输入目录走，输入目录叫 p 就还是 p.txt，叫 audio 就是 audio.txt
    root_group = str(_pick(args, "root_group", input_dir.name or "root"))
    return {
        "PCM_INPUT": input_dir,
        "ROOT_GROUP": root_group,
        "FAILED_DIR": Path(_pick(args, "fail_dir", str(data_root / "f"))),
        "OUT_DIR": out_dir,
        "LOG_DIR": Path(_pick(args, "log_dir", str(out_dir / "log"))),
        "RULES_FILE": Path(_pick(args, "rules", str(out_dir / "rules.txt"))),
        "TMPLIST": Path(_pick(args, "tmplist", str(out_dir / "tmplist.txt"))),
        "LOCK_FILE": Path(_pick(args, "lock_file", str(out_dir / ".lock"))),
        "STOP_FILE": Path(_pick(args, "stop_file", str(out_dir / "STOP"))),
        "ONNX_DIR": code_dir,
        "MODELS_DIR": models_dir,
        "ASR_DIR": str(_pick(args, "asr_dir", "")),
        "VAD_DIR": Path(_pick(args, "vad_dir", str(models_dir / "fireredvad-onnx"))),
        "PUNC_DIR": Path(_pick(args, "punc_dir", str(models_dir / "fireredpunc-onnx"))),
        "PUNC_FILE": _pick(args, "punc_file", "") or "",
        "EXTENSIONS": exts,
        "ASR_THREADS": int(_pick(args, "threads", thr_default)),
        "ASR_GRAPH": _pick(args, "graph", "mixed"),
        "ASR_MODE": _pick(args, "asr_mode", "greedy"),
        "CACHE_LEN": _pick(args, "cache", "auto"),
        "PUNC_THREADS": int(_pick(args, "punc_threads", 4)),
        # 0 = 不设上限（保持 onnxruntime 默认：吃满所有物理核）
        "VAD_THREADS": int(_pick(args, "vad_threads", vad_thr_default)),
        "USE_PUNC": not bool(getattr(args, "no_punc", False)),
        "MIN_SEG_SEC": float(_pick(args, "min_seg_sec", 0.1)),
        "ON_DONE": _pick(args, "on_done", "trash"),
        "NOTIFY_EVERY": int(_pick(args, "notify_every", 100)),
        "ABORT_CONSEC_FAIL": int(_pick(args, "abort_consec_fail", 15)),
        "REPEAT_FILTER_MAX_LINE": int(_pick(args, "repeat_filter_max_line", 4000)),
        # 默认不推：ntfy 地址是个人的，属凭据，不写进分发包
        "NTFY_TOPIC_URL": _pick(args, "ntfy_url", "") or "",
        "LOG_LEVEL": getattr(logging, _pick(args, "log_level", "INFO")),
    }


# 先解析、再落全局：日志、锁、扫描都在模块级用这些值，必须它们先就位。
# 未知参数（可能是 --status 之类的旧写法打错了）在这里就报错退出，绝不静默开跑。
_PASER = build_parser()
ARGS, _EXTRA = _PASER.parse_known_args()
if _EXTRA:
    _PASER.error(f"不认识的参数：{' '.join(_EXTRA)}")
for _k, _v in _apply_config(ARGS).items():
    globals()[_k] = _v
MODELS = MODELS_DIR          # asr_chain 侧同名变量的别名，方便读代码时对齐
DEL_MODE_NOTE = ""

# ==================== INIT ====================

for _p in (OUT_DIR, LOG_DIR, FAILED_DIR):
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except Exception as _e:
        # 建不出来是环境故障（权限/盘/路径）。这里不能只打日志继续跑：那样后面每个文件
        # 都会失败一遍，看上去像"转完了"。直接非零退出。
        print(f"[致命] 目录建不出来 {_p}：{_e!r}", file=sys.stderr)
        sys.exit(6)

# 输入目录反过来：它不存在时**不替用户造**。造了一个空目录，扫描就是 0 个文件、
# 队列"跑完"、退出码 0 —— 路径打错的人会以为整批转完了。--status/--stop 是只读动作，
# 不在这儿拦（它们本来就不需要输入存在）。
if not PCM_INPUT.is_dir() and not (ARGS.status or ARGS.stop):
    print(f"[致命] 输入目录不存在：{PCM_INPUT}\n"
          f"       路径打错就改 --input-dir；想用默认目录就先把它建出来。"
          f"驱动不替你建空输入目录，否则会被当成\"转完了\"。", file=sys.stderr)
    sys.exit(6)

# 告诉 asr_chain.py / lid_onnx.py 模型在哪（它们在没有 CLI 参数的场景下读这个环境变量）
os.environ["FIREDASR_MODELS"] = str(MODELS_DIR)

# ==================== LOGGER ====================

log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"chain_{log_timestamp}.log"
logger = logging.getLogger("firedasr_chain")
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


# ==================== Ctrl+C ====================
#
# 一次 Ctrl+C：当前文件跑完再退（已写出的结果照常、源文件照常回收）。
# 两次 Ctrl+C：立即抛 KeyboardInterrupt 退出；没跑完的文件还在输入目录里，下轮重跑。
# 与 STOP 文件等价，主循环在每个文件边界检查同一个条件。
# 处理器内不调用 logger：可能与主线程争同一把日志锁。

_ctrl_c_count = 0
_stop_requested = False


def _on_ctrl_c(signum, frame):
    global _ctrl_c_count, _stop_requested
    _ctrl_c_count += 1
    if _ctrl_c_count == 1:
        _stop_requested = True
        sys.stderr.write("[Ctrl+C] 当前文件结束后停止；再按一次立即终止\n")
    else:
        sys.stderr.write("[Ctrl+C] 立即终止（未完成的文件留在输入目录，下轮重跑）\n")
        raise KeyboardInterrupt


def install_ctrl_c_handler():
    signal.signal(signal.SIGINT, _on_ctrl_c)


def stop_requested() -> bool:
    return _stop_requested or STOP_FILE.exists()


# ==================== NTFY（urllib，无第三方依赖）====================

# Ctrl+N 开关。单独一个线程轮询按键：主线程跑转写时不在读键，若只在文件边界查键，
# 这次按键要等当前文件结束才生效。无控制台（分离启动）时 kbhit 抛异常，线程自行退出。

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
        logger.warning(f"ntfy 推送失败：{e}")


# ==================== RULES ====================

def load_rules(path: Path):
    """逐条容错：单条非法正则不应导致整份规则失效。"""
    rules_list = []
    if not path.exists():
        logger.info(f"规则文件不存在：{path}")
        return rules_list
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except Exception as e:
        logger.error(f"读取规则文件失败：{e}")
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
            logger.warning(f"规则第 {lineno} 行正则非法，已跳过：{e}")
    logger.info(f"加载规则 {len(rules_list)} 条" + (f"，跳过 {bad} 条" if bad else ""))
    return rules_list


def apply_rules(text: str, rules_list: list) -> str:
    for pattern, repl in rules_list:
        text = pattern.sub(repl, text)
    return text


# ==================== 幻觉过滤（AED 复读类坏案的通用清洗）====================

_REPEAT_SHORT = re.compile(r'(.{1,20}?)\1{5,}')
_REPEAT_LONG  = re.compile(r'(.{10,50}?)\1{3,}')


def remove_repetitions(text: str) -> str:
    """按行处理并限长：带反向引用的正则在超长单行上会回溯爆炸。"""
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


# ==================== 文件扫描与分组（一级子目录分组、组内 mtime 升序）====================

def build_file_groups() -> dict:
    """按输入目录（--input-dir）下第一级子目录分组，组内按 mtime 升序。"""
    groups = {}
    for f in PCM_INPUT.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in EXTENSIONS:
            continue
        parts = f.relative_to(PCM_INPUT).parts
        groups.setdefault(ROOT_GROUP if len(parts) == 1 else parts[0], []).append(f)

    for key in groups:
        groups[key].sort(key=lambda p: p.stat().st_mtime)
    groups = {k: groups[k] for k in sorted(groups)}

    total = sum(len(v) for v in groups.values())
    try:
        with open(TMPLIST, "w", encoding="utf-8") as tf:
            tf.write(f"# 生成时间：{datetime.now()}  总计：{total} 个文件\n\n")
            for key, files in groups.items():
                tf.write(f"# [{key}] → {key}.txt  共 {len(files)} 个\n")
                for f in files:
                    tf.write(f"  {f}\n")
                tf.write("\n")
    except Exception as e:
        logger.warning(f"写入队列文件失败：{e}")

    logger.info(f"本次队列：{len(groups)} 个分组，共 {total} 个文件")
    for key, files in groups.items():
        logger.info(f"  [{key}] {len(files)} 个文件 → {key}.txt")
    return groups


def quarantine(source_file: Path):
    """按相对路径隔离失败文件，保留目录结构以免不同子目录的同名文件互相覆盖。

    返回隔离后的路径（没搬成返回 None）——熔断时要凭这个把文件搬回输入目录，
    环境故障期间被搬走的健康文件不该留在隔离区里再也没人扫。
    """
    try:
        rel = source_file.relative_to(PCM_INPUT)
    except ValueError:
        rel = Path(source_file.name)
    dest = FAILED_DIR / rel
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(source_file), str(dest))
        logger.info(f"已隔离至：{dest}")
        return dest
    except Exception as e:
        logger.error(f"隔离失败 {source_file.name}：{e}")
        return None


def settle_source(source_file: Path):
    """按 --on-done 结算已转写/已判空转写的源文件。

    keep 什么都不做（重复跑会重复转写）；trash 走回收站（要 Send2Trash，缺库就退回
    不动并记下原因，绝不静默永久删）；delete 直接删。
    """
    if ON_DONE == "keep":
        logger.info(f"保留源文件（--on-done keep）：{source_file.name}")
        return
    if ON_DONE == "delete":
        try:
            os.remove(str(source_file))
        except Exception as e:
            logger.error(f"删除失败 {source_file.name}：{e}")
        return
    try:
        from send2trash import send2trash
    except ImportError:
        logger.error(f"没装 Send2Trash（pip install Send2Trash），源文件保留不删：{source_file.name}")
        return
    try:
        send2trash(str(source_file))
    except Exception as e:
        logger.error(f"移入回收站失败 {source_file.name}：{e}")


# ==================== 引擎（进程内只加载一次，跨文件复用）====================

class Engine:
    """纯 ORT 全链：FireRedVAD（判段）-> FireRedASR2-AED（图与解码方式见参数）-> FireRedPunc。

    与链路 asr_chain.py 用同一批实现（load_audio / build_recognizer / Punc 直接复用），
    区别只有两点：① 模型在进程里保活一次；② VAD 不走"临时 wav 往返"，
    链路 run_vad 会 sf.write 到 %TEMP% 再让 vad.detect 读回来，这里按同样口径
    内存直喂（float->int16 用 rint 取整，与 PCM_16 落盘的舍入口径一致）。
    """

    def __init__(self):
        sys.path.insert(0, str(ONNX_DIR))
        import asr_chain as chain
        # 驱动的参数是唯一权威：把链路侧的目录全局改成同一套值，
        # 这样 --models-dir / --vad-dir 等对三个环节同时生效。
        chain.MODELS = MODELS_DIR
        chain.ASR_DIR = ASR_DIR
        chain.VAD_DIR = VAD_DIR
        chain.PUNC_DIR = PUNC_DIR
        chain.PUNC_FILE = PUNC_FILE
        self.chain = chain

        sys.path.insert(0, str(VAD_DIR))
        from infer_onnx import FireRedVadOnnx

        t_all = time.perf_counter()
        t0 = time.perf_counter()
        # VAD 的 session 是 vendor 的 infer_onnx.py 自己建的（第 320-323 行），它只设了
        # graph_optimization_level，没设 intra_op_num_threads —— ORT 的 0 = 吃满所有物理核。
        # VAD 又在每个文件开头对整条音频一次性提特征，所以会瞬时点满全部核心：省不了多少
        # 总时间（实测 VAD 环节 1.0s -> 0.9s），收益在同机共存时不被瞬抢。这里临时替换
        # InferenceSession 工厂，只给"没显式设过线程数"的 session 注入上限；ASR 由 aed_ort
        # 显式设 ASR_THREADS、Punc 显式设 PUNC_THREADS，都不受影响。try/finally 保证建完就
        # 还原工厂，不污染后面的建 session。不改 vendor 文件，它升级时不会冲突。
        import onnxruntime as _ort
        _orig_ss = _ort.InferenceSession

        def _capped_ss(path_or_bytes, sess_options=None, *a, **kw):
            if not VAD_THREADS:
                return _orig_ss(path_or_bytes, sess_options=sess_options, *a, **kw)
            so = sess_options if sess_options is not None else _ort.SessionOptions()
            if not getattr(so, "intra_op_num_threads", 0):
                so.intra_op_num_threads = VAD_THREADS
            return _orig_ss(path_or_bytes, sess_options=so, *a, **kw)

        _ort.InferenceSession = _capped_ss
        try:
            self.vad = FireRedVadOnnx(model_dir=str(VAD_DIR))
        finally:
            _ort.InferenceSession = _orig_ss
        t_vad = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.asr = chain.build_recognizer("aed", ASR_THREADS, mode=ASR_MODE,
                                          cache=CACHE_LEN, graph=ASR_GRAPH)
        t_asr = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.punc = chain.Punc(threads=PUNC_THREADS) if USE_PUNC else None
        t_punc = time.perf_counter() - t0
        logger.info(f"引擎加载：VAD {t_vad:.1f}s + ASR({ASR_GRAPH}/{ASR_MODE}) {t_asr:.1f}s "
                    f"+ Punc {t_punc:.1f}s，合计 {time.perf_counter() - t_all:.1f}s")

    def segments(self, wav, sr):
        import numpy as np
        w16 = np.rint(np.clip(wav, -1.0, 1.0) * 32768.0).clip(-32768, 32767)
        w16 = w16.astype(np.int16)
        feat = self.vad._extract_features(w16)
        probs = self.vad._run_model(feat)
        decisions = self.vad.postprocessor.process(probs.tolist())
        return self.vad.postprocessor.decisions_to_segments(decisions, len(wav) / sr)

    def transcribe(self, path: Path):
        """返回 (最终文本, 统计)。空字符串 = 无语音/无内容的合法空转写。"""
        wav, sr = self.chain.load_audio(path)
        dur = len(wav) / sr
        min_samples = int(MIN_SEG_SEC * sr)
        if wav.size < min_samples:
            return "", {"dur": round(dur, 2), "speech": 0.0, "segs": 0, "sec": 0.0}

        t0 = time.perf_counter()
        segs = self.segments(wav, sr)
        speech = sum(e - s for s, e in segs)
        texts = []
        n_trunc = 0
        for s, e in segs:
            seg = wav[int(s * sr): int(e * sr)]
            if seg.size < min_samples:
                continue
            r = self.asr.transcribe_wav(seg, sr)
            n_trunc += bool(r["truncated"])
            texts.append(r["text"].strip())
        raw = "".join(texts)
        if n_trunc:
            logger.warning(f"有 {n_trunc} 段撞上 cache 上限被截断（文本可能缺尾）")
        if raw and self.punc is not None:
            raw = "".join(self.punc.add(x) for x in texts)
        sec = time.perf_counter() - t0
        return raw, {"dur": round(dur, 2), "speech": round(speech, 2),
                     "segs": len(segs), "sec": round(sec, 2)}


# ==================== 主处理 ====================

class Stats:
    def __init__(self):
        self.ok = 0          # 有文本、已落盘
        self.dropped = 0     # 空转写（无语音），源文件已删、不落记录
        self.failed = []
        self.stopped = False
        self.aborted = False     # 连续失败熔断
        self.abort_streak = 0    # 熔断那一段连续失败了几个（回搬后 consec_fail 归零，靠它写结算）
        self.consec_fail = 0     # 距上一次成功/空转写以来连续失败的文件数
        self.streak = []         # 这段连续失败里被隔离走的 (原路径, 隔离路径)，熔断时搬回去

    def processed(self) -> int:
        return self.ok + self.dropped + len(self.failed)


def restore_streak(stats: Stats) -> int:
    """熔断时把这段连续失败里隔离走的文件搬回输入目录，并撤销它们的失败记账。

    连续全失败说明是环境坏了（读不到/写不进），不是文件本身有问题。这些文件
    留在隔离区里就再也不会被扫到，等于被静默丢弃，所以必须回搬并改回"未处理"。
    """
    moved = 0
    back = set()
    for src, dest in stats.streak:
        try:
            if dest.exists() and not src.exists():
                src.parent.mkdir(parents=True, exist_ok=True)
                os.replace(str(dest), str(src))
                back.add(str(src.relative_to(PCM_INPUT)))
                moved += 1
        except Exception as e:
            logger.error(f"熔断回搬失败 {dest}：{e}")
    stats.streak.clear()
    stats.consec_fail = 0
    if back:
        stats.failed = [(n, err) for n, err in stats.failed if n not in back]
    return moved


def emit(out_txt: Path, source_file: Path, text: str) -> bool:
    """写入一条转写记录（一次写全，尽量缩小被杀时写半截的窗口）。"""
    rec = f"title:{source_file.relative_to(PCM_INPUT)}\n{text}\n\n"
    try:
        with open(out_txt, "a", encoding="utf-8") as f:
            f.write(rec)
        return True
    except Exception as e:
        logger.error(f"写入 {out_txt.name} 失败，保留源文件：{e}")
        return False


def process_file(eng: Engine, out_txt: Path, source_file: Path, rules_list: list,
                 stats: Stats):
    logger.info(f"转写：{source_file.name}")
    try:
        raw, info = eng.transcribe(source_file)
    except Exception as e:
        logger.error(f"FAILED: [{source_file.name}] {e!r}", exc_info=True)
        stats.failed.append((str(source_file.relative_to(PCM_INPUT)), repr(e)))
        stats.consec_fail += 1
        dest = quarantine(source_file)
        if dest is not None:
            stats.streak.append((source_file, dest))
        return

    text = postprocess(raw, rules_list)

    # 空转写（纯静音 / 无语音段）是合法结果：直接结算掉，不写输出、不计失败
    if not text:
        logger.info(f"空输出（无语音）：{source_file.name}")
        settle_source(source_file)
        stats.dropped += 1
        stats.consec_fail = 0
        stats.streak.clear()
        return

    if not emit(out_txt, source_file, text):
        stats.failed.append((str(source_file.relative_to(PCM_INPUT)), "写输出失败"))
        stats.consec_fail += 1
        return

    stats.consec_fail = 0
    stats.streak.clear()

    rtf = info["sec"] / info["dur"] if info["dur"] else 0.0
    logger.info(f"完成：{source_file.name}  音频 {info['dur']}s / 语音 {info['speech']}s "
                f"/ {info['segs']} 段 / 用时 {info['sec']}s（RTF {rtf:.3f}）")
    stats.ok += 1
    settle_source(source_file)


def process_group(eng: Engine, group_key: str, file_list: list, rules_list: list,
                  stats: Stats, total: int):
    out_txt = OUT_DIR / f"{group_key}.txt"
    logger.info(f"分组 [{group_key}]：{len(file_list)} 个文件 → {out_txt.name}")

    for f in file_list:
        if stop_requested():
            logger.info("检测到停止请求（Ctrl+C 或 STOP 文件），停止处理")
            stats.stopped = True
            return
        if not f.exists():
            continue
        process_file(eng, out_txt, f, rules_list, stats)
        if ABORT_CONSEC_FAIL and stats.consec_fail >= ABORT_CONSEC_FAIL:
            streak = stats.consec_fail
            stats.aborted = True
            stats.abort_streak = streak
            rolled = restore_streak(stats)
            logger.critical(
                f"连续 {streak} 个文件一个都没成功（累计 成功 {stats.ok} / 失败 {len(stats.failed)}）"
                f"—— 判定为环境故障（权限/磁盘/路径），熔断停止，剩余文件原样留在输入目录"
            )
            if rolled:
                logger.critical(f"本段连续失败搬走的 {rolled} 个文件已搬回输入目录，下次启动会重新转")
            return
        if NOTIFY_EVERY and stats.processed() % NOTIFY_EVERY == 0:
            notify(f"asr进度：已处理 {stats.processed()}/{total} 个文件"
                   f"（成功 {stats.ok} / 空转写 {stats.dropped} / 失败 {len(stats.failed)}）")


# ==================== MAIN ====================

def count_input_audio():
    """用 os.walk + onerror 独立复扫一遍待转文件。

    Path.rglob 在目录读不了（权限、损坏、抖动）时是**静默返回空**的，不会抛异常。
    所以"0 个待转文件"必须区分"真转完了"和"根本读不到"：后者绝不能报成功，
    否则整批文件会被无声跳过（本项目就栽过一次：目录 ACL 被掏空的那分钟里
    6 千个文件全部 open 失败，队列照跑完、还推了"成功"）。
    """
    errors = []
    n = 0
    for root, dirs, files in os.walk(str(PCM_INPUT), onerror=lambda e: errors.append(e)):
        for name in files:
            if Path(name).suffix.lower() in EXTENSIONS:
                n += 1
    return n, errors


def main_loop() -> int:
    if STOP_FILE.exists():
        try:
            STOP_FILE.unlink()
        except Exception:
            pass

    try:
        eng = Engine()
    except Exception as e:
        logger.critical(f"引擎加载失败：{e!r}", exc_info=True)
        return 2

    rules = load_rules(RULES_FILE)
    groups = build_file_groups()
    total = sum(len(v) for v in groups.values())
    if not groups:
        n_walk, walk_err = count_input_audio()
        if walk_err or n_walk:
            logger.critical(f"分组结果为 0，但复扫仍有 {n_walk} 个待转文件、{len(walk_err)} 个枚举错误"
                            f"（首个：{walk_err[0] if walk_err else '无'}）—— 判为读取异常，不报成功")
            notify(f"asr 异常：分组扫到 0 个，但复扫有 {n_walk} 个待转文件 / {len(walk_err)} 个枚举错误"
                   f"，这是读不到不是转完了，源文件一个没动")
            return 5
        logger.info("队列为空，无文件需要处理")
        notify("asr successful 😀")
        return 0

    stats = Stats()
    for group_key, file_list in groups.items():
        process_group(eng, group_key, file_list, rules, stats, total)
        if stats.stopped or stats.aborted:
            break

    logger.info(f"统计：成功 {stats.ok} / 空转写 {stats.dropped} / 失败 {len(stats.failed)}")
    if stats.failed:
        logger.error(f"失败 {len(stats.failed)} 个：")
        for fname, err in stats.failed:
            logger.error(f"  {fname} | {err}")
    elif stats.aborted:
        logger.error(f"熔断退出：队列共 {total} 个，已处理 {stats.processed()} 个，"
                     f"剩余 {total - stats.processed()} 个原样留在输入目录，未转写")
    else:
        logger.info("所有文件处理成功")

    if stats.aborted:
        notify(f"asr 熔断：连续 {stats.abort_streak} 个文件全失败，已停（成功 {stats.ok}，"
               f"约 {total - stats.processed()} 个未处理）—— 环境故障，源文件都在输入目录，没转完")
        return 3
    if stats.failed:
        notify(f"asr 结束：成功 {stats.ok} / 空转写 {stats.dropped} / 失败 {len(stats.failed)}"
               f" —— 有文件没转成，别当已完成")
        return 1
    if stats.stopped:
        notify("asr stopped 😀")
    else:
        notify("asr successful 😀")
    return 0


# ==================== 后台启动 / 优雅停止 ====================

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


# 动作类参数只对本次命令行生效，不转发给后台实例（否则后台一启动就把自己停了）
_ACTION_FLAGS = ("--start", "--bg", "/start", "--stop", "/stop", "--status")


def child_argv() -> list:
    """本次命令行里的配置参数原样转交给后台实例；动作参数剔除。"""
    return [a for a in sys.argv[1:] if a.lower() not in _ACTION_FLAGS]


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
    here = os.path.abspath(__file__)
    proc = subprocess.Popen([sys.executable, here] + child_argv(), **kwargs)
    print(f"已后台启动：pid={proc.pid}")
    print(f"参数：{' '.join(child_argv()) or '(全用默认值)'}")
    print(f"控制台输出：{err_h.name}")
    print(f"停止：{sys.executable} {here} --stop")
    if os.name == "nt":
        print("注意：别关掉启动它的那个窗口/终端，同一个 job 会连坐把后台 worker 一起杀掉")
    else:
        print("已 setsid 脱离控制终端，关终端不会带走它；Ctrl+N 静音只在 Windows 有，这里用 --stop")
    return 0


def cmd_start() -> int:
    if not acquire_lock():
        print(f"已有实例在运行（锁：{LOCK_FILE}），本次不启动")
        return 4
    release_lock()
    return spawn_background()


def cmd_stop() -> int:
    STOP_FILE.write_bytes(b"")
    print(f"已建立停止标志：{STOP_FILE}")
    print("在跑的实例会转完当前文件后退出并释放锁；下次启动会自动删掉该标志。")
    return 0


def cmd_status() -> int:
    """只读：锁在不在、输入目录有多少待转文件、各分组输出多大。"""
    print(f"输入目录 : {PCM_INPUT}  ({'存在' if PCM_INPUT.is_dir() else '不存在'})")
    n, errs = (0, [])
    if PCM_INPUT.is_dir():
        n, errs = count_input_audio()
    print(f"待转文件 : {n}" + (f"（枚举报错 {len(errs)} 个，首个：{errs[0]}）" if errs else ""))
    print(f"输出目录 : {OUT_DIR}")
    if OUT_DIR.is_dir():
        for p in sorted(OUT_DIR.glob("*.txt")):
            if p == TMPLIST:
                continue
            print(f"  {p.name:24s} {p.stat().st_size:>12,} B")
    print(f"隔离区   : {FAILED_DIR}（失败文件按相对路径留在这儿，不会自动重扫）")
    # 锁探测：能拿到 = 没有在跑的实例；拿不到 = 有，且本次不写它
    print(f"运行状态 : {'有实例在跑（锁 ' + str(LOCK_FILE) + '）' if not acquire_lock() else '空闲'}")
    if _lock_fh:
        release_lock()
    return 0


# ==================== ENTRY ====================

if __name__ == "__main__":
    # 参数合法性在模块级已经过了 argparse（不认识的参数当场 error 退出），这里只分流。
    if ARGS.stop:
        sys.exit(cmd_stop())
    if ARGS.start:
        sys.exit(cmd_start())
    if ARGS.status:
        sys.exit(cmd_status())

    code = 3
    if not acquire_lock():
        logger.critical(f"已有实例在运行（锁：{LOCK_FILE}），本次退出")
        print(f"已有实例在运行（锁：{LOCK_FILE}），本次退出。"
              f"要让它在下一个文件边界停下就加 --stop，别去杀进程/关窗口。", file=sys.stderr)
        sys.exit(4)
    install_ctrl_c_handler()
    start_key_watcher()
    try:
        code = main_loop()
    except KeyboardInterrupt:
        logger.info("用户中断")
        code = 130
    except Exception as e:
        logger.critical(f"未捕获异常：{e}", exc_info=True)
    finally:
        release_lock()
        logging.shutdown()
    sys.exit(code)
