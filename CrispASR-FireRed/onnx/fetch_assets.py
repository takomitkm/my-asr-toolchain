#!/usr/bin/env python3
"""一键补全本包需要的全部资源：公开下载 + 本地建造，逐件按 sha256 验收。

为什么有两类来源：
  · 下载 —— AED 的 int8 图在 sherpa-onnx 的 GitHub release 里，VAD / Punc 的导出在
    HuggingFace 上，都不需要 token、不需要登录（2026-10-06 逐条 HEAD 实测 HTTP 200）。
  · 建造 —— encoder.f32.onnx 和 punc.f32.onnx 全网没有公开匿名下载件。前者是
    dequant_aed.py 把 ORT 动态量化的 int8 图重写成纯 f32 图；后者是 dequant_punc.py
    把 weight-only 8-bit 的 MatMulNBits 逆量化成普通 MatMul（onnxruntime<=1.20 的
    contrib 只实现了 4-bit，不逆量化就加载不了 punc.q8w.onnx）。两个脚本都在本包里，
    造出来的东西逐字节可复现：2026-10-06 在校机上从 int8/q8w 原料重跑，punc.f32、
    encoder.f32(+.data)、decoder.f32(+.data) 五件全部与产线文件同 sha256。

用法：
  python fetch_assets.py --models ./models --graph mixed
  python fetch_assets.py --models ./models --check          # 只核对，不下不建
  python fetch_assets.py --models ./models --list           # 打印完整资源清单
需要代理时：--proxy http://127.0.0.1:7890（或直接用环境变量 https_proxy）
建造需要的 onnx 包默认用当前 python；若转写环境里没有 onnx，用
  --builder-python <装了 onnx 的 python.exe> 指一个装了 onnx 的解释器。

断点续传：半截文件写成 <目标>.part，按 Range 接着下；连接中途断掉或被代理截短
（实测会少几十万字节）同样按 .part 续，最多 3 次。只有逐件 sha256 对上才算完成，
对不上就删掉重下一次，绝不把坏文件留在盘上让链路去猜。
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request

# Windows 控制台/重定向默认走 cp936，中文文件名一多就 UnicodeEncodeError，先钉成 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

B = 2**20

# ---------------------------------------------------------------- 资源清单
# 每条：dest（相对 --models）/ url / bytes / sha256（全部在产线目录实测）
GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
HFV = "https://huggingface.co/tardigrade-doc/FireRedVAD_onnx/resolve/main"
HFP = "https://huggingface.co/jiangzhuo9357/fireredpunc-onnx/resolve/main"

AED_DIR = "sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26"

DOWNLOADS = [
    # AED：一个 tar.bz2 解开就是 encoder.int8 / decoder.int8 / tokens.txt
    {"name": "aed_int8_tar", "dest": "fire-red-asr2-aed-int8.tar.bz2",
     "url": f"{GH}/sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26.tar.bz2",
     "bytes": 838589068, "sha256": "43015b3f1643a5688b4821e8ed323473d38b798c4ec291471fe00df1bcfc4f1c",
     "extract": {"into": AED_DIR, "prefix": AED_DIR + "/",
                 "members": [
                     {"path": "encoder.int8.onnx", "bytes": 817286833,
                      "sha256": "54048d66b6e8f3c80ea7ce95efe794587b0fd81d7271651d0decd3803852ae82"},
                     {"path": "decoder.int8.onnx", "bytes": 417291928,
                      "sha256": "b840ce7196ae4a14d05ae84bbf56082b6b61ccec5610fda907dddbcea37354ff"},
                     {"path": "tokens.txt", "bytes": 79172,
                      "sha256": "1bc613de2112d257e61a349c3e72d1b1a9cf19c33d3ca954197ad2171e5ea07b"}]}},
    # VAD：4 个二进制/配置文件；推理脚本 infer_onnx.py 随本包（与上游同哈希，哈希见 README.md）
    {"name": "vad_model", "dest": "fireredvad-onnx/model.onnx",
     "url": f"{HFV}/model.onnx", "bytes": 2461278,
     "sha256": "517e9c6207618407da41fc274b1e3f09e8cde531db42f039a52be93b29a49151"},
    {"name": "vad_cmvn_bin", "dest": "fireredvad-onnx/cmvn.bin",
     "url": f"{HFV}/cmvn.bin", "bytes": 644,
     "sha256": "b020eb6a57b01993c7aa032fbb0e33d257359ef1bdcb4b66e3dc360f11b42d4e"},
    {"name": "vad_cmvn_json", "dest": "fireredvad-onnx/cmvn.json",
     "url": f"{HFV}/cmvn.json", "bytes": 3293,
     "sha256": "662d07cfe6e111ef386ce5b932def7e6c8218eb3fdd73db7588f2ab66851e825"},
    {"name": "vad_config", "dest": "fireredvad-onnx/config.json",
     "url": f"{HFV}/config.json", "bytes": 2,
     "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"},
    # Punc：q8w 本体 + 分词器 + 标点类别表（f32 由本包 dequant_punc.py 造）
    {"name": "punc_q8w", "dest": "fireredpunc-onnx/punc.q8w.onnx",
     "url": f"{HFP}/punc.q8w.onnx", "bytes": 162771205,
     "sha256": "5b7cfdd8a8b7228c56b4d2123b4b09a4af34d70cd43f613d1fcccb35bd2ece8f"},
    {"name": "punc_tokenizer", "dest": "fireredpunc-onnx/tokenizer.json",
     "url": f"{HFP}/tokenizer.json", "bytes": 268961,
     "sha256": "53ff61207898738bbdc000f38abebef01041c8d23b6270c11855fc692d0a3ad6"},
    {"name": "punc_out_dict", "dest": "fireredpunc-onnx/out_dict",
     "url": f"{HFP}/out_dict", "bytes": 33,
     "sha256": "6f0f7e0004881d617bc6e1d7b5b39972da80dcb49576bca489b1603ee55e20bb"},
    {"name": "punc_license", "dest": "fireredpunc-onnx/LICENSE",
     "url": f"{HFP}/LICENSE", "bytes": 11357,
     "sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"},
    {"name": "punc_readme", "dest": "fireredpunc-onnx/README.md",
     "url": f"{HFP}/README.md", "bytes": 4227,
     "sha256": "2ed2edffe0e405a0441e7e9e3c4efae11b10eb6e74131461754d3745a0c9c495"},
]

# 建造件：产物哈希 = 产线上正在用的那份（本机同名脚本重跑一次即可复现）
BUILDS = {
    "punc_f32": {
        "dest": "fireredpunc-onnx/punc.f32.onnx",
        "bytes": 406954097,
        "sha256": "71c54314cb8129e4ca491169f4a770061041e8127e53b1b0edda423a8a103d95",
        "needs_src": "fireredpunc-onnx/punc.q8w.onnx",
        "argv": ["{pkg}/dequant_punc.py", "{src}", "{dst}"],
        "label": "Punc 逆量化（8-bit MatMulNBits -> f32）",
    },
    "aed_encoder_f32": {
        "dest": f"{AED_DIR}/encoder.f32.onnx",
        "bytes": 956747,                      # 图本体；权重在同名 .data（3,103,167,616 B）
        "sha256": "1fa3b6c8503143a21d32fbf3a4dadae6ad73dfb4ebc6b3ac2d35c69a037e0b42",
        "needs_src": f"{AED_DIR}/encoder.int8.onnx",
        "argv": ["{pkg}/dequant_aed.py", "convert", "{src}", "{dst}"],
        "label": "AED 编码器反量化（int8 动态量化图 -> 纯 f32）",
        "extra_dest": f"{AED_DIR}/encoder.f32.onnx.data",
        "extra_bytes": 3103167616,
        "extra_sha256": "66c3b4f0293c476681b7c01ca7fac356e718878adc6bfbf0a7aefeed40b25f1a",
    },
    "aed_decoder_f32": {
        "dest": f"{AED_DIR}/decoder.f32.onnx",
        "bytes": 1099726,
        "sha256": "52bd0efacb0f536c3f4cc050a09d6094f3451d7938af52ca0f54c96fd03575a6",
        "needs_src": f"{AED_DIR}/decoder.int8.onnx",
        "argv": ["{pkg}/dequant_aed.py", "convert", "{src}", "{dst}"],
        "label": "AED 解码器反量化（只在 --graph f32 时需要）",
        "extra_dest": f"{AED_DIR}/decoder.f32.onnx.data",
        "extra_bytes": 1550396160,
        "extra_sha256": "7728fe4ac558e36b0d0c07b619dde8e87aad8d7d2bf44a65432bf46b0406d654",
    },
}

# --graph 与要建的东西一一对应（和 xhs-chain-cpu.py 的 --asr-graph 同名词）
GRAPH_BUILDS = {"int8": ["punc_f32"],
               "mixed": ["punc_f32", "aed_encoder_f32"],
               "f32": ["punc_f32", "aed_encoder_f32", "aed_decoder_f32"]}


def sha256_of(path, block=8 * B):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def human(n):
    for unit, div in (("GB", 1024**3), ("MB", B), ("KB", 1024)):
        if n >= div:
            return f"{n/div:.1f} {unit}"
    return f"{n} B"


def opener(proxy):
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener()


def download(item, models, OP, force):
    dest = os.path.join(models, item["dest"].replace("/", os.sep))
    if os.path.isfile(dest) and not force:
        ok, msg = verify(dest, item["bytes"], item["sha256"])
        if ok:
            print(f"[已有] {item['dest']}  {msg}")
            return True
        print(f"[重下] {item['dest']}  {msg}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    url = item["url"]
    for attempt in (1, 2, 3):
        # 每次重进都按磁盘上的实际 .part 大小续，别信上一轮的旧读数
        have = os.path.getsize(part) if os.path.isfile(part) else 0
        if have > item["bytes"]:
            os.remove(part)
            have = 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "firedasr-onnx-win-aigc/fetch_assets"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            print(f"[下载] {item['dest']}  从 {have}/{item['bytes']} B 起 "
                  f"({attempt} 次)")
            with OP.open(req, timeout=180) as r, open(part, "ab" if have else "wb") as f:
                code = getattr(r, "status", 200)
                if have and code != 206:          # 服务器不认 Range，推倒重来
                    have = 0
                    f.truncate(0)
                got = have
                last = got // (200 * B)
                while True:
                    chunk = r.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if got // (200 * B) != last:   # 每 200 MB 报一次，不刷屏
                        last = got // (200 * B)
                        print(f"       … {human(got)} / {human(item['bytes'])}", flush=True)
        except (urllib.error.URLError, OSError) as e:
            if attempt == 3:
                raise SystemExit(f"[失败] 下载 {url} 出错：{e!r}\n"
                                 f"       需要代理就加 --proxy http://127.0.0.1:7890")
            print(f"[网络出错] {e!r}，保留 .part 重试", flush=True)
            continue
        if got != item["bytes"]:
            # 连接提前结束（代理抖动常见）：这不是坏文件，接着下就行，别推倒重来
            if attempt == 3:
                raise SystemExit(f"[失败] {item['dest']} 重试 3 次仍只收到 {got} B，"
                                 f"清单写 {item['bytes']} B")
            print(f"[断了] {item['dest']} 收到 {got}/{item['bytes']} B，"
                  f"从断点接着下（{attempt} 次）", flush=True)
            continue
        ok, msg = verify(part, item["bytes"], item["sha256"])
        if not ok:
            os.remove(part)
            if attempt == 3:
                raise SystemExit(f"[失败] {item['dest']} 重下后哈希仍不对：{msg}")
            print(f"[哈希不符] {item['dest']}：{msg}，删掉重下一次", flush=True)
            continue
        os.replace(part, dest)
        print(f"[完成] {item['dest']}  sha256 对上了")
        return True
    return False


def verify(path, want_bytes, want_sha):
    if not os.path.isfile(path):
        return False, "文件不存在"
    n = os.path.getsize(path)
    if n != want_bytes:
        return False, f"大小 {n} != 清单 {want_bytes}"
    got = sha256_of(path)
    if got != want_sha:
        return False, f"sha256 {got[:12]}… != {want_sha[:12]}…"
    return True, f"{human(n)} sha256 {want_sha[:12]}… ✓"


def verify_extra(spec, models):
    """外置权重 .data 的验收：清单给了 extra_sha256 就逐字节核，没给只能核尺寸。

    两件 f32 的 .data 都有产线实测哈希，而且 2026-10-06 在校机上从 int8 原料重跑
    dequant_aed.py 造出来的那两份与产线逐字节相同，所以这里是核哈希、不是只核尺寸。
    """
    if not spec.get("extra_dest"):
        return True, ""
    p = os.path.join(models, spec["extra_dest"].replace("/", os.sep))
    if not spec.get("extra_sha256"):
        n = os.path.getsize(p) if os.path.isfile(p) else -1
        return (n == spec["extra_bytes"],
                "文件不存在" if n < 0
                else f"{n} B（清单 {spec['extra_bytes']} B）")
    return verify(p, spec["extra_bytes"], spec["extra_sha256"])


def extract(item, models, keep_tar):
    tar = os.path.join(models, item["dest"].replace("/", os.sep))
    into = os.path.join(models, AED_DIR.replace("/", os.sep))
    ready = True
    for m in item["extract"]["members"]:
        p = os.path.join(into, m["path"].replace("/", os.sep))
        if not (os.path.isfile(p) and os.path.getsize(p) == m["bytes"]
                and sha256_of(p) == m["sha256"]):
            ready = False
    if ready:
        print(f"[已有] {AED_DIR}/ 三个成员齐全且哈希对，跳过解包")
        return
    print(f"[解包] {tar} -> {into}")
    prefix = item["extract"]["prefix"]
    with tarfile.open(tar, "r:bz2") as tf:
        for member in tf:
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            rel = member.name[len(prefix):]
            if not rel or rel.startswith("..") or os.path.isabs(rel):
                raise SystemExit(f"[失败] 包里有不安全的路径成员 {member.name}")
            dst = os.path.join(into, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            src = tf.extractfile(member)
            with open(dst, "wb") as out:
                shutil.copyfileobj(src, out, 8 * B)
    for m in item["extract"]["members"]:
        p = os.path.join(into, m["path"].replace("/", os.sep))
        ok, msg = verify(p, m["bytes"], m["sha256"])
        print(("  [ok]  " if ok else "  [FAIL]") + f" {m['path']}  {msg}")
        if not ok:
            raise SystemExit(f"[失败] 解包后 {m['path']} 验收没过")
    if not keep_tar and os.path.isfile(tar):
        os.remove(tar)
        print(f"[清理] 删掉 {item['dest']}（要保留加 --keep-tar）")


def need_builder_python(bp):
    try:
        r = subprocess.run([bp, "-c", "import onnx, numpy; print(onnx.__version__)"],
                           capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"[失败] 建造用的解释器跑不起来：{bp}  {e!r}")
    if r.returncode != 0:
        raise SystemExit(
            f"[失败] 建造 f32 需要装了 onnx + numpy 的 python，{bp} 不行：\n"
            f"       {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '无输出'}\n"
            f"       换 --builder-python 指一个（例如 pip install onnx 的虚拟环境），"
            f"或先用 --graph int8 免掉 AED 的反量化。")
    print(f"[建造] 解释器 {bp}  onnx {r.stdout.strip()}")


def build(key, models, pkg, bp, force, dry):
    spec = BUILDS[key]
    dest = os.path.join(models, spec["dest"].replace("/", os.sep))
    src = os.path.join(models, spec["needs_src"].replace("/", os.sep))
    if os.path.isfile(dest) and not force:
        ok, msg = verify(dest, spec["bytes"], spec["sha256"])
        extra_ok, _ = verify_extra(spec, models)
        if ok and extra_ok:
            print(f"[已有] {spec['dest']}  {msg}")
            return
        print(f"[重建] {spec['dest']}  {msg}")
    if not os.path.isfile(src):
        raise SystemExit(f"[失败] 建造 {spec['dest']} 的原料不在：{src}（先跑下载）")
    argv = [bp] + [a.format(pkg=pkg.replace(os.sep, "/"), src=src, dst=dest)
                   for a in spec["argv"]]
    print(f"[建造] {spec['label']}\n       {' '.join(argv)}")
    if dry:
        return
    # 建造脚本的中文输出走 UTF-8：不指定则子进程按系统代码页（cp936）编码，
    # 重定向到文件时那些 … / → 会被写成乱码甚至抛 UnicodeEncodeError
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = __import__("time").time()
    r = subprocess.run(argv, env=env)
    if r.returncode != 0:
        raise SystemExit(f"[失败] 建造进程返回 {r.returncode}，见上面它的输出")
    print(f"       用时 {__import__('time').time()-t0:.0f} s")
    ok, msg = verify(dest, spec["bytes"], spec["sha256"])
    print(("  [ok]  " if ok else "  [FAIL]") + f" {spec['dest']}  {msg}")
    if not ok:
        raise SystemExit("[失败] 造出来的文件与产线不一致——别用它跑链路，把上面两行哈希发回来")
    extra = spec.get("extra_dest")
    if extra:
        e_ok, e_msg = verify_extra(spec, models)
        print(f"  {'[ok]  ' if e_ok else '[FAIL]'} {extra}  {e_msg}")
        if not e_ok:
            raise SystemExit("[失败] 外置权重 .data 与产线不一致——别用它跑链路")


def report(models, keys, items=None):
    """把每个资源的实际状态列成一张表，返回缺/坏的条数。"""
    rows, bad = [], 0
    for item in (DOWNLOADS if items is None else items):
        if item["name"] == "aed_int8_tar":
            for m in item["extract"]["members"]:
                p = os.path.join(models, AED_DIR, m["path"])
                ok, msg = verify(p, m["bytes"], m["sha256"])
                rows.append((ok, f"{AED_DIR}/{m['path']}", msg))
                bad += (not ok)
        else:
            p = os.path.join(models, item["dest"].replace("/", os.sep))
            ok, msg = verify(p, item["bytes"], item["sha256"])
            rows.append((ok, item["dest"], msg))
            bad += (not ok)
    for key in keys:
        spec = BUILDS[key]
        p = os.path.join(models, spec["dest"].replace("/", os.sep))
        ok, msg = verify(p, spec["bytes"], spec["sha256"])
        rows.append((ok, f"[建] {spec['dest']}", msg))
        bad += (not ok)
        if spec.get("extra_dest"):
            e_ok, e_msg = verify_extra(spec, models)
            rows.append((e_ok, f"[建] {spec['extra_dest']}", e_msg))
            bad += (not e_ok)
    for ok, name, msg in rows:
        print(("  就绪  " if ok else "  缺/坏  ") + f"{name:58s} {msg}")
    return bad


def main():
    ap = argparse.ArgumentParser(description="下载 + 建造本包的全部必需资源")
    ap.add_argument("--models", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "models"), help="模型根目录，默认包旁的 models")
    ap.add_argument("--graph", choices=sorted(GRAPH_BUILDS), default="mixed",
                    help="按转写侧 --asr-graph 决定要建哪些 f32（int8=只建 Punc；"
                         "mixed=再加 AED 编码器；f32=再加解码器）")
    ap.add_argument("--check", action="store_true", help="只核对现状，不下不建")
    ap.add_argument("--list", action="store_true", help="打印资源清单（下载地址与哈希）")
    ap.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or "",
                    help="代理，如 http://127.0.0.1:7890；默认取环境变量")
    ap.add_argument("--builder-python", default=sys.executable,
                    help="跑 dequant 脚本的解释器（要装了 onnx+numpy），默认当前 python")
    ap.add_argument("--force", action="store_true", help="已就绪的也重下/重建")
    ap.add_argument("--keep-tar", action="store_true", help="解包后保留 tar.bz2（默认删）")
    ap.add_argument("--dry-run", action="store_true", help="只打印要执行的建造命令")
    ap.add_argument("--only", metavar="NAME[,NAME]", help="只处理这些条目（名字见 --list）")
    A = ap.parse_args()

    models = os.path.abspath(A.models)
    pkg = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(models, exist_ok=True)

    if A.list:
        print("下载件（全部匿名可取，无需 token）：")
        for it in DOWNLOADS:
            print(f"  {it['name']:16s} {it['dest']:52s} {human(it['bytes']):>10s}  {it['sha256'][:16]}…")
            print(f"  {'':16s} {it['url']}")
            if it.get("extract"):
                for m in it["extract"]["members"]:
                    print(f"  {'':16s}   解出 {AED_DIR}/{m['path']:20s} {human(m['bytes']):>10s}  {m['sha256'][:16]}…")
        print("建造件（无公开下载入口，本包脚本可复现）：")
        for k, s in BUILDS.items():
            print(f"  {k:16s} {s['dest']:52s} {human(s['bytes']):>10s}  {s['sha256'][:16]}…")
            print(f"  {'':16s} 由 {' '.join(s['argv'])} 生成，原料 {s['needs_src']}")
        print("\n--graph 对应要建的东西：")
        for g, ks in GRAPH_BUILDS.items():
            print(f"  {g:6s} -> {', '.join(ks)}")
        return 0

    only = set(A.only.split(",")) if A.only else None
    items = [it for it in DOWNLOADS if only is None or it["name"] in only]
    keys = [k for k in GRAPH_BUILDS[A.graph] if only is None or k in only]

    OP = opener(A.proxy)
    if A.check:
        bad = report(models, GRAPH_BUILDS[A.graph])
        print(f"\n[结论] {'全部就绪' if not bad else f'{bad} 项缺或坏 —— 跑 fetch_assets.py（不带 --check）补齐'}")
        return 0 if not bad else 1

    if only is None:
        # 分两个数：tar.bz2 解包成功后会删掉，所以"解出来留在盘上的成员"才是常驻成本，
        # 压缩包本体只是解包那一会儿的瞬时成本。只按下载件的字节数算是低估（这份包
        # 解出来 1.23 GB > 压缩包 0.84 GB）。
        resident = transient = 0
        for it in DOWNLOADS:
            if it["name"] in ("punc_license", "punc_readme"):
                continue                      # 只有法务价值，不计进判据（照样会下）
            if it.get("extract"):
                transient += it["bytes"]       # 压缩包：解完即删
                resident += sum(m["bytes"] for m in it["extract"]["members"])
            else:
                resident += it["bytes"]
        for k in keys:
            resident += BUILDS[k]["bytes"] + BUILDS[k].get("extra_bytes", 0)
        need = resident + transient
        free = shutil.disk_usage(models).free
        print(f"[空间] 常驻约 {human(resident)}，解包那一会儿再多 {human(transient)}"
              f"（合计峰值约 {human(need)}）；{models} 所在盘现有 {human(free)}")
        if free < need:
            raise SystemExit("[失败] 空间不够，先腾地方（或 --graph int8 少建 3 GB 的编码器 f32）")
    if keys and not A.dry_run:
        need_builder_python(A.builder_python)

    for it in items:
        download(it, models, OP, A.force)
        if it.get("extract"):
            extract(it, models, A.keep_tar)
    for key in keys:
        build(key, models, pkg, A.builder_python, A.force, A.dry_run)

    bad = report(models, keys, items)
    scope = "" if only is None else "（--only 范围内）"
    print(f"\n[结算] {'全部就绪' if not bad else f'{bad} 项没过关'}{scope}")
    if bad:
        return 1
    if only is None and A.graph != "int8":
        print("下一步：python xhs-chain-cpu.py --input-dir <音频目录> --data-root <工作目录>")
    elif only is None:
        print("下一步：跑转写时记得用 --asr-graph int8，否则链路会去找没建的 encoder.f32.onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
