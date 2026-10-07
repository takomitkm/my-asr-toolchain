"""把 FireRedPunc 的 weight-only 8-bit 图（MatMulNBits）逆量化成纯 f32 图。

为什么需要它：公开可匿名下载的只有 punc.q8w.onnx（162 MB），它用的是
com.microsoft::MatMulNBits（bits=8、block_size=32），而 onnxruntime<=1.20 的 contrib
只实现了 4-bit，加载 8-bit 会报 "Only 4b quantization is supported"。f32 版（407 MB）
没有公开匿名下载入口，所以这里给出制作方法。

逆量化口径（与 ORT 自己的 MatMulNBits CPU kernel 逐元素对齐过，见下）：
    W_q    : uint8  (N, K/block_size, block_size)   展平存放，元素数 = N*K
    scales : f32    (N, K/block_size)               每行每块一个 scale
    无 zero_points 输入时按"无符号存储、有符号解释"处理，即恒定减 128：
        W_f32[n, k] = (W_q[n, k] - 128) * scales[n, k // block_size]
    带 zero_points 时（8-bit 同样要先把 zp 减 128）：
        W_f32[n, k] = (W_q[n, k] - zp[n, k // block_size]) * scales[n, k // block_size]
    MatMul 要 (K, N)，所以最后转置。

这套口径的来源：ORT quantization 包里的量化器把 scale/zero_point 直接当 initializer 塞进
图（不走 C2 常量折叠），所以能从 punc.q8w.onnx 里把两者一起读出来手算。10-06 实测三条：
  1) 造出来的图与产线上正在用的 punc.f32.onnx 逐字节相同（sha256 71c54314cb8129e4…，
     406,954,097 B，见 README 的资源表）；
  2) 同一批随机输入，参照 punc.q8w.onnx（ORT 1.24.2 才跑得动 8-bit MatMulNBits）与
     本图 logits 最大绝对差 1.1e-05 ~ 4.0e-01（序列 64/200/400，logit 值域约 ±12），
     5 类标点 argmax 65/65 全一致；
  3) 本图在链路钉住的 ORT 1.20.1 下能正常加载并出结果。
差值没到 float32 舍入级，原因未查（8-bit kernel 的累加路径与转成 f32 后走标准 GEMM
不完全等价），但判据是类别一致率而不是 logits 逐位相同。

节点/权重命名与产线上那份 punc.f32.onnx 完全一致（_dq{序号}_mm / _dq{序号}_w），
所以这边造出来的图和正在跑的图是同一张图。

用法（需要一个装了 onnx 的环境；转写运行环境本身不装 onnx）：
  python dequant_punc.py <in.q8w.onnx> <out.f32.onnx>
  python dequant_punc.py --survey <in.q8w.onnx>     # 只看量化模板是否匹配，不写文件
"""
import argparse
import os
import sys
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper


def dequant(B, S, Z, bits, K, N):
    """(packed uint8, scales[, zero_point]) -> (N, K) float32。口径见模块 docstring。"""
    nb = S.shape[-1]
    if K % nb:
        raise SystemExit(f"K={K} 不是 scale 组数 {nb} 的整数倍，块对齐假设不成立")
    w = B.astype(np.float32).reshape(N, K)
    rep_scale = np.repeat(S.astype(np.float32).reshape(N, nb), K // nb, axis=1)
    if Z is None:
        if bits != 8:
            raise SystemExit(f"bits={bits} 且无 zero_points：只处理 8-bit 对称量化（-128）")
        w = w - 128.0
    else:
        z = Z.astype(np.float32).reshape(N, nb)
        if bits == 8:
            z = z - 128.0
        w = w - np.repeat(z, K // nb, axis=1)
    out = w * rep_scale
    if not np.isfinite(out).all():
        raise SystemExit("逆量化结果里有 inf/nan")
    return out


def survey(src):
    m = onnx.load(src, load_external_data=False)
    old = [n for n in m.graph.node if n.op_type == "MatMulNBits"]
    print(f"MatMulNBits 节点 {len(old)} 个")
    init = {i.name: i for i in m.graph.initializer}
    tpl = {}
    for n in old:
        a = {x.name: x.i for x in n.attribute}
        shapes = []
        for nm in n.input[1:]:
            i = init.get(nm)
            shapes.append(() if i is None else tuple(i.dims))
        key = (a.get("bits"), a.get("block_size"), len(n.input), tuple(shapes))
        tpl[key] = tpl.get(key, 0) + 1
    for k, c in sorted(tpl.items(), key=lambda kv: -kv[1]):
        print(f"  {c:>4} x  bits={k[0]} block={k[1]} 输入数={k[2]} 常量形状={k[3]}")
    missing = [n.name for n in old
               for nm in n.input[1:3] if nm not in init]
    if missing:
        print(f"[warn] {len(missing)} 个节点的权重不是 initializer（无法静态逆量化）："
              f"{missing[:3]}")
    return len(old)


def convert(src, dst):
    m = onnx.load(src)
    init = {i.name: i for i in m.graph.initializer}
    old = [n for n in m.graph.node if n.op_type == "MatMulNBits"]
    print(f"MatMulNBits 节点 {len(old)} 个")
    if not old:
        raise SystemExit("图里一个 MatMulNBits 都没有：这已经不是量化图，不改动")

    sub = {}
    extra_init = []
    retired = set()
    total = 0
    t0 = time.perf_counter()
    for idx, n in enumerate(old):
        a = n.input[0]
        B = numpy_helper.to_array(init[n.input[1]])
        S = numpy_helper.to_array(init[n.input[2]])
        Z, bias = None, None
        for nm in n.input[3:]:
            if nm in ("", None) or nm not in init:
                continue
            arr = numpy_helper.to_array(init[nm])
            if arr.dtype == np.uint8 and arr.shape == S.shape:
                Z = arr
            elif arr.dtype == np.float32 and arr.ndim == 1:
                bias = (nm, arr)
            else:
                raise SystemExit(f"节点 {n.name} 的输入 {nm} 形状 {arr.dtype}{arr.shape} 认不出来")
        attr = {x.name: x.i for x in n.attribute}
        for req in ("K", "N", "bits"):
            if req not in attr:
                raise SystemExit(f"节点 {n.name} 缺属性 {req}，不敢猜")
        K, N, bits = attr["K"], attr["N"], attr["bits"]
        w = dequant(B, S, Z, bits, K, N)
        wt = np.ascontiguousarray(w.T, dtype=np.float32)
        total += wt.size
        tag = f"_dq{idx}"
        wn = f"{tag}_w"
        extra_init.append(numpy_helper.from_array(wt, wn))
        out = n.output[0]
        rep = []
        if bias is not None:
            bn = f"{tag}_b"
            extra_init.append(numpy_helper.from_array(bias[1], bn))
            rep.append(helper.make_node("MatMul", [a, wn], [out + "_dqmm"], name=f"{tag}_mm"))
            rep.append(helper.make_node("Add", [out + "_dqmm", bn], [out], name=f"{tag}_add"))
            retired.add(bias[0])
        else:
            rep.append(helper.make_node("MatMul", [a, wn], [out], name=f"{tag}_mm"))
        # 用旧节点的输出张量名登记替换关系：节点名可能重复，输出名在图里必须唯一
        sub[out] = rep
        retired.update([n.input[1], n.input[2]] + list(n.input[3:]))

    ordered = []
    for n in m.graph.node:
        if n.op_type == "MatMulNBits":
            ordered.extend(sub[n.output[0]])
        else:
            ordered.append(n)
    del m.graph.node[:]
    m.graph.node.extend(ordered)

    keep = [x for x in m.graph.initializer if x.name not in retired]
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep + extra_init)

    contrib = [x for x in m.graph.node if x.domain == "com.microsoft"]
    opset = m.opset_import[0].version if m.opset_import else 17
    del m.opset_import[:]
    m.opset_import.extend([helper.make_opsetid("", opset)])
    if contrib:
        m.opset_import.extend([helper.make_opsetid("com.microsoft", 1)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"逆量化 {total/1e6:.1f}M 参数，用时 {time.perf_counter()-t0:.1f} s -> {dst}")
    print(f"剩余 MatMulNBits: {sum(1 for x in m.graph.node if x.op_type=='MatMulNBits')}，"
          f"剩余 com.microsoft: {len(contrib)}")
    print(f"输出大小 {os.path.getsize(dst)/2**20:,.1f} MiB")


def main():
    ap = argparse.ArgumentParser(
        description="FireRedPunc 的 MatMulNBits(weight-only 8-bit) -> 纯 f32 MatMul")
    ap.add_argument("src", nargs="?", help="输入的量化 onnx（如 models/fireredpunc-onnx/punc.q8w.onnx）")
    ap.add_argument("dst", nargs="?", help="输出路径（如 models/fireredpunc-onnx/punc.f32.onnx）")
    ap.add_argument("--survey", metavar="SRC", help="只打印量化模板，不转换")
    A = ap.parse_args()
    if A.survey:
        survey(A.survey)
        return 0
    if not A.src or not A.dst:
        ap.error("需要 src 和 dst 两个位置参数（或用 --survey 只看模板）")
    convert(A.src, A.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
