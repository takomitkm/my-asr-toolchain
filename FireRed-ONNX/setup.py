#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""setup.py —— 建虚拟环境 + 装依赖 + 自检 + 列出还缺什么资源。

Windows 和 Linux/macOS 都能跑（Windows 上可以双击 setup.bat，它只是转过来调用本文件）。
这一步不下载任何模型；模型是第 2 步 fetch_assets.py 的事。

用法：
    python setup.py                 运行侧 + 建造侧依赖（默认）
    python setup.py nobuilder       只装运行侧（用 --graph int8 转写，不造 f32）
    python setup.py noruntime       只装建造侧
    python setup.py --python /path/to/python3.12     用指定解释器建 venv
    python setup.py --skip-check    不跑最后那条资源核对
    python setup.py --force         跳过解释器版本窗口检查

退出码：0 环境就绪（哪怕资源还缺，那是下一步的事）/ 6 环境不对。
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"

# numpy 2.5.3 / scipy 1.18.1 的 requires_python 都是 >=3.12，
# onnxruntime 1.20.1 只发到 cp313（3.14 没轮子），所以窗口是 3.12–3.13。
PY_MIN = (3, 12)
PY_MAX = (3, 13)

RUNTIME_IMPORTS = ["onnxruntime", "numpy", "soundfile", "scipy",
                   "tokenizers", "kaldi_native_fbank", "kaldiio"]


def say(msg=""):
    print(msg, flush=True)


def fail(msg, hint=None):
    say(f"\n[失败] {msg}")
    if hint:
        for line in hint.splitlines():
            say(f"       {line}")
    sys.exit(6)


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(argv, cwd=None, capture=False):
    """跑一个子命令；capture=True 时返回 CompletedProcess，否则失败就把输出抖出来。"""
    proc = subprocess.run(argv, cwd=cwd, text=True, errors="replace",
                          capture_output=True) if capture else \
        subprocess.run(argv, cwd=cwd)
    return proc


def step0_python(base: str, force: bool):
    say("=== 0. 解释器 ==============================================")
    probe = run([base, "-c",
                 "import sys;print('%d.%d.%d'%sys.version_info[:3]);"
                 "print(sys.executable)"], capture=True)
    if probe.returncode != 0:
        fail("这个解释器跑不起来", f"{base}\n{(probe.stderr or '').strip()}")
    lines = (probe.stdout or "").strip().splitlines()
    ver = tuple(int(x) for x in lines[0].split("."))
    say(f"     用 {base} -> CPython {lines[0]}（{lines[-1]}）")
    if not force and (ver < PY_MIN or ver > PY_MAX):
        fail(f"解释器 {lines[0]} 不在钉版本的窗口里（{'.'.join(map(str, PY_MIN))}–"
             f"{'.'.join(map(str, PY_MAX))}）",
             "requirements.txt 钉的 numpy/scipy 要求 >=3.12，onnxruntime 1.20.1 只到 3.13。\n"
             "换一个解释器：python setup.py --python /path/to/python3.12\n"
             "或者你确定要试：python setup.py --force（pip 解析失败别当本包的 bug 报）")
    return ver, lines[0]


def step1_venv(base: str):
    say("=== 1. 建 .venv ============================================")
    vp = venv_python(VENV)
    if vp.exists():
        say(f"     .venv 已经在了，复用：{vp}")
        return vp
    proc = run([base, "-m", "venv", str(VENV)], capture=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        hint = "Windows：勾上 launcher；或者 python setup.py --python py（用 py 启动器）"
        if os.name != "nt":
            hint = ("Debian/Ubuntu 的 python3 常把 venv 拆成单独的包：\n"
                    "    sudo apt update && sudo apt install -y python3-venv python3-pip\n"
                    "（conda / pyenv 装的解释器一般自带 venv，不用 apt）")
        fail(".venv 建不起来", f"{err}\n{hint}")
    if not vp.exists():
        fail("venv 建完了但里面没有解释器", f"预期路径：{vp}\n删掉 {VENV} 再跑一次")
    say(f"     建好：{vp}")
    return vp


def step2_pip(vp: Path, mode: str):
    say("=== 2. 装依赖 ==============================================")
    run([str(vp), "-m", "pip", "install", "--upgrade", "pip"], capture=True)
    jobs = []
    if mode != "noruntime":
        jobs.append(("requirements.txt", "转写运行侧（8 件钉版本，下载约 65–70 MB）"))
    if mode != "nobuilder":
        jobs.append(("requirements-build.txt", "建造侧（onnx + numpy，再加约 8–9 MB）"))
    for name, label in jobs:
        req = HERE / name
        if not req.exists():
            fail(f"缺 {name}", "本包文件不完整，重新取一份")
        say(f"     {name} —— {label}")
        proc = run([str(vp), "-m", "pip", "install", "-r", str(req)])
        if proc.returncode != 0:
            extra = ""
            if name == "requirements.txt":
                extra = ("多半是钉死的版本在你的平台没有轮子（尤其 onnxruntime==1.20.1）。\n"
                         "先确认解释器在 3.12–3.13；Linux 上 glibc 要 >=2.28（manylinux_2_28）。\n"
                         "只要转写、不打算造 f32：python setup.py nobuilder，然后用 --graph int8。")
            fail(f"pip install -r {name} 失败", extra)


def step3_import(vp: Path, mode: str):
    say("=== 3. 导入自检 ============================================")
    if mode == "noruntime":
        code = "import onnx, numpy; print('builder imports ok, onnx', onnx.__version__)"
    else:
        code = ("import onnxruntime as ort, numpy, soundfile, scipy, tokenizers, "
                "kaldi_native_fbank, kaldiio;"
                "print('runtime imports ok, onnxruntime', ort.__version__, "
                "'providers', ort.get_available_providers())")
    proc = run([str(vp), "-c", code], capture=True)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        hint = ""
        if "DLL" in err or "126" in err:
            hint = ("Windows 上 import onnxruntime 崩（0xC0000142 / DLL 初始化例程失败）"
                    "多是 System32 的 VC 运行库太旧，装一份最新的 vc_redist.x64 再试。")
        elif "GLIBC" in err or "version `GLIBC" in err:
            hint = "这台机器的 glibc 太老，onnxruntime 1.20.1 的轮子要 >=2.28。"
        fail("导入自检没过 —— 别往下走", f"{err}\n{hint}")
    say("     " + (proc.stdout or "").strip())


def step4_assets(vp: Path, skip: bool):
    say("=== 4. 资源现状 ============================================")
    if skip:
        say("     跳过（--skip-check）")
        return
    proc = run([str(vp), str(HERE / "fetch_assets.py"), "--check"])
    if proc.returncode != 0:
        say("     ↑ 上面列出的缺项是正常的：本步只管环境，资源在第 2 步下载/建造")


def step5_next(vp: Path, pyver: str):
    say("")
    say(f"环境就绪（{pyver}）。第 2 步把模型补齐，三档挑一条：")
    say(f"    {vp} fetch_assets.py --list            先看全部 URL / 字节数 / sha256")
    say(f"    {vp} fetch_assets.py --graph int8      最小档：占 1.68 GB，峰值 2.46 GB，不用 onnx")
    say(f"    {vp} fetch_assets.py --graph mixed     产线默认：占 4.57 GB，峰值 5.35 GB，要造 f32 编码器")
    say(f"    {vp} fetch_assets.py --graph f32       全解耦：占 6.02 GB，峰值 6.80 GB")
    say("    要代理就加  --proxy http://127.0.0.1:7890")
    say("第 3 步转写：")
    say(f"    {vp} xhs-chain-cpu.py --input-dir <音频目录> --data-root <数据根>")
    say("    Windows 上直接前台跑就行；要后台加 --start，但别关掉启动它的那个窗口（同 job 会连坐）")
    say("    Linux/macOS 上 --start 走 setsid，脱离控制终端，关终端不会带走它；停止一律 --stop")
    say("尺寸与哈希清单见 README.md 第 4.1 节。")


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            if s is not None and not s.isatty():
                s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="建 venv + 装依赖 + 自检（Windows / Linux / macOS 通用）",
        epilog="退出码：0 环境就绪 / 6 环境不对",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=("noruntime", "nobuilder"),
                    help="noruntime=只装建造侧；nobuilder=只装运行侧；不给=两侧都装")
    ap.add_argument("--python", dest="base_python", default=sys.executable,
                    help="用它来建 venv，默认当前解释器")
    ap.add_argument("--force", action="store_true",
                    help="跳过解释器版本窗口检查")
    ap.add_argument("--skip-check", action="store_true",
                    help="不跑 fetch_assets.py --check")
    args = ap.parse_args()

    os.chdir(HERE)
    say(f"包目录：{HERE}")

    _ver, pyver = step0_python(args.base_python, args.force)
    vp = step1_venv(args.base_python)
    step2_pip(vp, args.mode)
    step3_import(vp, args.mode)
    step4_assets(vp, args.skip_check)
    step5_next(vp, pyver)
    sys.exit(0)


if __name__ == "__main__":
    main()
