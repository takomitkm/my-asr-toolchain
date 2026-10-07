#!/usr/bin/env python3
r"""按同目录 assets.json 一键补全本引擎需要的全部组件，逐件按 sha256 验收。

清单是机器可读的：每一条只有 url / bytes / sha256 三个硬字段，加上落到哪个根目录
（roots）和文件名（path）。程序 zip 还带 members（解压出来的每个文件各自的 sha256），
所以"下了 zip"不等于"组件齐了"——展开后逐个成员还要再对一次哈希才算数。

用法：
  python fetch_assets.py                     # 下载 + 校验，缺什么补什么
  python fetch_assets.py --check             # 只核对现状，不联网
  python fetch_assets.py --list              # 打印清单（URL、字节数、sha256、出处等级）
  python fetch_assets.py --only a,b          # 只处理指定条目（名字见 --list）
  python fetch_assets.py --force             # 已经对上的也重来
  python fetch_assets.py --proxy http://127.0.0.1:7897
  python fetch_assets.py --set-root model=/path/to/别的目录

落点默认相对本目录：<本目录>/crispasr、<本目录>/model（assets.json 的 roots.default）。
想跟清单里的默认值不同，用 --set-root NAME=PATH 或环境变量（roots.env 写的那个名字）。
补完之后把驱动 CONFIG 里的 CRISPASR_BIN_DIR / MODEL_DIR 指到这两个目录就能跑，
本脚本不改驱动、也不假设驱动的路径。

断点续传：半截文件写成 <目标>.part，按 Range 接着下，最多 3 次；只有逐件 sha256
对上才算完成，对不上就删掉重下，绝不把坏文件留在盘上让链路去猜。

出处等级（每件都标，别把 B 当 A 用）：
  A = 这个 sha256 同时是发布方给的（GitHub release digest 或 HuggingFace 的 LFS oid），
      而且和作者生产机上那份逐字节相同 —— 链接下来的就是她在跑的那个文件。
  B = sha256 取自作者生产机那份，链接内容没有独立凭据可对照（清单里的 bytes 与上游一致）。
  C = 没有公开匿名下载件，需要人工装或从别处拷（kind=manual，--check 只给 WARN 不算失败）。
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

B = 2 ** 20
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256_of(path, block=8 * B):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def human(n):
    for unit, div in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n} B"


def verify(path, want_bytes, want_sha):
    if not os.path.isfile(path):
        return False, "文件不存在"
    size = os.path.getsize(path)
    if want_bytes and size != want_bytes:
        return False, f"字节数 {size} != 清单 {want_bytes}"
    if want_sha:
        got = sha256_of(path)
        if got != want_sha:
            return False, f"sha256 {got[:16]}… != 清单 {want_sha[:16]}…"
    return True, f"{human(size)} sha256 对上"


def download(item, dest, OP, force, ua):
    if os.path.isfile(dest) and not force:
        ok, msg = verify(dest, item.get("bytes"), item.get("sha256"))
        if ok:
            print(f"[已有] {os.path.basename(dest)}  {msg}")
            return True
        print(f"[重下] {os.path.basename(dest)}  {msg}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    want = item["bytes"]
    for attempt in (1, 2, 3):
        have = os.path.getsize(part) if os.path.isfile(part) else 0
        if have > want:
            os.remove(part)
            have = 0
        try:
            req = urllib.request.Request(item["url"], headers={"User-Agent": ua})
            if have:
                req.add_header("Range", f"bytes={have}-")
            print(f"[下载] {item['name']}  从 {have}/{want} B 起（第 {attempt} 次）")
            with OP.open(req, timeout=300) as r, open(part, "ab" if have else "wb") as f:
                code = getattr(r, "status", 200)
                if have and code != 206:
                    have = 0
                    f.truncate(0)
                got = have
                last = got // (100 * B)
                while True:
                    chunk = r.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if got // (100 * B) != last:
                        last = got // (100 * B)
                        print(f"       … {human(got)} / {human(want)}", flush=True)
        except (urllib.error.URLError, OSError) as e:
            if attempt == 3:
                raise SystemExit(f"[失败] 下载 {item['url']} 出错：{e!r}\n"
                                 f"       需要代理就加 --proxy http://127.0.0.1:7897")
            print(f"[网络出错] {e!r}，保留 .part 重试", flush=True)
            continue
        if got != want:
            if attempt == 3:
                raise SystemExit(f"[失败] {item['name']} 重试 3 次仍只收到 {got} B，"
                                 f"清单写 {want} B")
            print(f"[断了] {item['name']} 收到 {got}/{want} B，从断点接着下", flush=True)
            continue
        ok, msg = verify(part, want, item.get("sha256"))
        if not ok:
            print(f"[哈希不对] {item['name']} {msg} —— 删掉重下")
            os.remove(part)
            continue
        os.replace(part, dest)
        print(f"[完成] {item['name']}  {msg}")
        return True
    return False


def extract_zip(item, root, keep_zip):
    """把 zip 里的 members 逐个展开到 root（去掉 zip_prefix），每个成员单独对哈希。"""
    src = os.path.join(root, item["path"].replace("/", os.sep))
    prefix = item.get("zip_prefix", "")
    problems = []
    with zipfile.ZipFile(src) as z:
        names = set(z.namelist())
        for m in item["members"]:
            target = os.path.join(root, m["path"].replace("/", os.sep))
            inside = prefix + m["path"]
            if m.get("sha256") and os.path.isfile(target):
                ok, msg = verify(target, m["bytes"], m["sha256"])
                if ok:
                    print(f"[已就位] {m['path']}  {msg}")
                    continue
            if inside not in names:
                problems.append(f"zip 里没有 {inside}")
                continue
            os.makedirs(os.path.dirname(target) or root, exist_ok=True)
            with z.open(inside) as f:
                data = f.read()
            with open(target, "wb") as w:
                w.write(data)
            ok, msg = verify(target, m["bytes"], m.get("sha256"))
            if ok:
                print(f"[展开] {m['path']}  {msg}")
            else:
                problems.append(f"{m['path']}：{msg}")
    if not keep_zip:
        os.remove(src)
    if problems:
        for p in problems:
            print(f"[失败] {p}")
        return False
    return True


def check_zip(item, root):
    """--check：zip 本身删掉了也能看成员是否已就位；缺成员就报缺。"""
    prefix = item.get("zip_prefix", "")
    missing, bad = [], []
    for m in item["members"]:
        target = os.path.join(root, m["path"].replace("/", os.sep))
        if not os.path.isfile(target):
            (missing if m.get("sha256") is not None else bad).append(m["path"])
            continue
        ok, msg = verify(target, m["bytes"], m.get("sha256"))
        if not ok:
            bad.append(f"{m['path']} {msg}")
    if missing:
        return False, "缺 " + ", ".join(missing) + "（跑 python fetch_assets.py 补）"
    if bad:
        return False, "; ".join(bad)
    return True, f"{len(item['members'])} 个成员全部就位且 sha256 对上"


def check_file(item, root):
    target = os.path.join(root, item["path"].replace("/", os.sep))
    return verify(target, item.get("bytes"), item.get("sha256"))


def check_manual(item, root):
    """kind=manual：没有公开匿名下载件。成员在位就算过，不在位只 WARN。"""
    members = item.get("members") or []
    if not members:
        return None, item.get("note", "需要人工处理").splitlines()[0]
    ok = 0
    for m in members:
        if verify(os.path.join(root, m["path"].replace("/", os.sep)),
                  m.get("bytes"), m.get("sha256"))[0]:
            ok += 1
    if ok == len(members):
        return True, f"{ok}/{len(members)} 件在位（清单记的是作者机上那几件）"
    return None, (f"{ok}/{len(members)} 件在位；" + item.get("note_short", "按 note 人工装"))


def resolve_roots(manifest, overrides):
    roots = {}
    for key, spec in manifest["roots"].items():
        env = spec.get("env")
        val = overrides.get(key) or (os.environ.get(env) if env else "") or spec["default"]
        # 清单允许写 ~/… 或 %USERPROFILE%\… 这类家目录形式；相对路径按本脚本所在目录算，与驱动同源
        val = os.path.expanduser(os.path.expandvars(val))
        roots[key] = val if os.path.isabs(val) else os.path.join(HERE, val)
    return roots


def walk(items):
    for it in items:
        yield it


def do_list(manifest):
    print(f"引擎 {manifest['engine']}   驱动 {manifest['driver']}")
    print("根目录（--set-root NAME=PATH 可改，也可用列出的环境变量）：")
    for key, spec in manifest["roots"].items():
        print(f"  {key:<8} default={spec['default']:<20} env={spec.get('env') or '-'}"
              f"   {spec.get('label','')}")
    print()
    for it in manifest["items"]:
        print(f"· {it['name']}  [{it['kind']} → {it.get('root','')}/]"
              f"  出处 {it.get('grade','-')}")
        print(f"    {it.get('url','（无下载链接）')}")
        if it["kind"] != "manual":
            print(f"    {it.get('path')}  {it.get('bytes')} B  sha256 {it.get('sha256')}")
        else:
            print("    需要人工装/拷，详见下面 note")
        for line in (it.get("note") or "").strip().splitlines():
            print(f"    | {line.strip()}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="按 assets.json 补全并校验本引擎的全部组件")
    ap.add_argument("--assets", default=os.path.join(HERE, "assets.json"))
    ap.add_argument("--check", action="store_true", help="只核对，不联网")
    ap.add_argument("--list", action="store_true", help="打印清单")
    ap.add_argument("--only", metavar="NAME[,NAME]", help="只处理这些条目")
    ap.add_argument("--force", action="store_true", help="已对上的也重来")
    ap.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or
                    os.environ.get("https_proxy") or "", help="例 http://127.0.0.1:7897")
    ap.add_argument("--set-root", action="append", default=[], metavar="NAME=PATH",
                    help="覆盖清单里某个根目录，可重复")
    ap.add_argument("--keep-zip", action="store_true", help="程序 zip 解完保留（默认删）")
    args = ap.parse_args()

    with open(args.assets, encoding="utf-8") as f:
        manifest = json.load(f)
    if args.list:
        return do_list(manifest)

    overrides = {}
    for pair in args.set_root:
        if "=" not in pair:
            raise SystemExit(f"[参数不对] --set-root 要写成 NAME=PATH，收到 {pair!r}")
        k, v = pair.split("=", 1)
        overrides[k] = v
    roots = resolve_roots(manifest, overrides)
    only = set(args.only.split(",")) if args.only else None
    ua = f"my-asr-toolchain-{manifest['engine']}/fetch_assets"
    OP = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy})
        if args.proxy else urllib.request.ProxyHandler())

    hard_fail, warns, done = [], [], 0
    for it in manifest["items"]:
        if only and it["name"] not in only:
            continue
        root = roots.get(it.get("root", ""))
        if root is None and it["kind"] != "manual":
            raise SystemExit(f"[清单错] {it['name']} 的 root={it.get('root')!r} 没在 roots 里")
        if root:
            os.makedirs(root, exist_ok=True)
        print(f"—— {it['name']}  {it.get('bytes') and human(it['bytes']) or '（多件）'}"
              f"  → {root or '人工'}")

        if args.check:
            if it["kind"] == "file":
                ok, msg = check_file(it, root)
            elif it["kind"] == "zip":
                ok, msg = check_zip(it, root)
            else:
                ok, msg = check_manual(it, root)
        else:
            if it["kind"] == "manual":
                ok, msg = check_manual(it, root)
            elif it["kind"] == "file":
                dest = os.path.join(root, it["path"].replace("/", os.sep))
                ok = download(it, dest, OP, args.force, ua)
                msg = "已下载并校验" if ok else "失败"
            else:  # zip：先把 zip 下到手（对 zip 自己的 sha256），再展开逐个成员
                dest = os.path.join(root, it["path"].replace("/", os.sep))
                if os.path.isfile(dest) and not args.force:
                    ok, _ = verify(dest, it["bytes"], it["sha256"])
                    if ok:
                        print(f"[已有] {it['path']}  zip 哈希已对上")
                    else:
                        os.remove(dest)
                if not os.path.isfile(dest):
                    ok = download(it, dest, OP, args.force, ua)
                if ok:
                    ok = extract_zip(it, root, args.keep_zip)
                msg = "已下载、展开并逐成员校验" if ok else "失败"
        if ok is None:
            warns.append(f"{it['name']}：{msg}")
            print(f"[WARN] {msg}")
        elif ok:
            done += 1
            print(f"[{'检查通过' if args.check else 'OK'}] {msg}")
        else:
            hard_fail.append(f"{it['name']}：{msg}")
            print(f"[不通过] {msg}")

    print()
    print(f"合计 通过 {done} 项，警告 {len(warns)} 项，不通过 {len(hard_fail)} 项")
    if warns:
        print("警告（不算失败，但要人看一眼）：")
        for w in warns:
            print("  - " + w)
    if hard_fail:
        print("不通过：")
        for h in hard_fail:
            print("  - " + h)
        return 3
    print("全部组件就位。把驱动 CONFIG 里的 CRISPASR_BIN_DIR / MODEL_DIR 指到上面的根目录即可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
