"""反量化：把 ORT 动态量化的 int8/u8 图重写成纯 f32 图，实测"反量化"这条路线。

图的量化模板（实测，decoder 用 i8、encoder 用 u8，MatMul/Conv 同构）：
    A_f32 --DynamicQuantizeLinear--> (A_q, a_scale, a_zp)
    W_q(初始化常量, scale, zp)
    MatMulInteger/ConvInteger(A_q, W, a_zp, w_zp) -> i32 --Cast--> f32 --Mul(Mul(a_scale,w_scale))--> out

重写方式：把最后那个 scale Mul 原地改成 f32 的 MatMul/Conv(A_f32, W_f32)，W_f32=(W-zp)*w_scale
作为新常量，删掉 *Integer 与 Cast，再做死节点清扫（DQL、scale 乘失去消费者后消失）。
注意这等于连激活量化误差一起去掉：数值和 int8 图不会逐位相同，文本以对拍为准。
任何一处模板不匹配就整体报错退出——绝不静默跳过。

用法（用 env_export 的 python 跑，它带 onnx）：
  python dequant_aed.py survey  <in.onnx>
  python dequant_aed.py convert <in.onnx> <out.onnx>
"""
import os
import sys
from collections import Counter

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

I8, U8, F32 = TensorProto.INT8, TensorProto.UINT8, TensorProto.FLOAT


def index(g):
    init = {i.name: i for i in g.initializer}
    producer = {}
    for n in g.node:
        for o in n.output:
            if o:
                producer[o] = n
    consumers = {}
    for n in g.node:
        for i in n.input:
            if i:
                consumers.setdefault(i, []).append(n)
    return init, producer, consumers


def classify(n, init, producer, consumers):
    r = {"node": n, "name": n.name, "ok": False, "why": "", "kind": n.op_type}
    A, B = n.input[0], n.input[1]
    zp_b = n.input[3] if len(n.input) > 3 and n.input[3] else None

    dql = producer.get(A)
    if dql is None or dql.op_type != "DynamicQuantizeLinear":
        r["why"] = "A 不是 DQL 输出: " + (dql.op_type if dql else "?")
        return r
    if B not in init or init[B].data_type not in (I8, U8):
        r["why"] = "B 不是 i8/u8 常量"
        return r
    cons1 = consumers.get(n.output[0], [])
    if len(cons1) != 1 or cons1[0].op_type != "Cast":
        r["why"] = "输出不是唯一 Cast: " + ",".join(c.op_type for c in cons1)
        return r
    cast = cons1[0]
    Yf = cast.output[0]
    cons2 = consumers.get(Yf, [])
    if len(cons2) != 1 or cons2[0].op_type != "Mul":
        r["why"] = "Cast 输出不是唯一 Mul: " + ",".join(c.op_type for c in cons2)
        return r
    mul = cons2[0]
    s_name = [i for i in mul.input if i != Yf]
    if len(s_name) != 1:
        r["why"] = "Mul 形状异常"
        return r
    sp = producer.get(s_name[0])
    if sp is None or sp.op_type != "Mul":
        r["why"] = "scale 不是 Mul 出来的: " + (sp.op_type if sp else "常量?")
        return r
    w_scale = None
    for i in sp.input:
        if i in init and init[i].data_type == F32:
            w_scale = i
    if w_scale is None:
        r["why"] = "scale 组合里找不到 f32 常量权重 scale"
        return r

    r.update(ok=True, A_f32=dql.input[0], cast=cast, cast_out=Yf, mul=mul,
             B=B, zp_b=zp_b, w_scale=w_scale)
    if zp_b is not None:
        r["zp_b_zero"] = bool(np.all(numpy_helper.to_array(init[zp_b]) == 0))
    else:
        r["zp_b_zero"] = True
    r["w_scale_shape"] = list(init[w_scale].dims)
    r["w_shape"] = list(init[B].dims)
    return r


def collect(g):
    init, producer, consumers = index(g)
    R = [classify(n, init, producer, consumers)
         for n in g.node if n.op_type in ("MatMulInteger", "ConvInteger")]
    bad = [r for r in R if not r["ok"]]
    print(f"量化节点 {len(R)} 个（MatMulInteger+ConvInteger），模板匹配 {len(R)-len(bad)} 个，"
          f"不匹配 {len(bad)} 个")
    for r in bad[:15]:
        print("  不匹配:", r["name"], "|", r["why"])
    if bad:
        raise SystemExit("模板不匹配，拒绝动刀")
    return R, init, producer


def survey(path):
    m = onnx.load(path, load_external_data=False)
    R, init, producer = collect(m.graph)
    print("按算子:", dict(Counter(r["kind"] for r in R)))
    print("zp_b 全零:", sum(r["zp_b_zero"] for r in R), "/", len(R))
    print("w_scale 形状:", dict(Counter(tuple(r["w_scale_shape"]) for r in R)))
    print("W 形状:", dict(Counter(tuple(r["w_shape"]) for r in R).most_common(12)))
    print("A_f32 的来源:", dict(Counter(
        (producer.get(r["A_f32"]).op_type if producer.get(r["A_f32"]) else "graph_input")
        for r in R)))
    tot_i8 = sum(int(np.prod(r["w_shape"])) for r in R)
    print(f"这些权重量化后合计 {tot_i8/2**20:.0f} MiB，f32 后 {tot_i8*4/2**20:.0f} MiB")
    # 顺带看一眼剩下的 DQL / DequantizeLinear 是什么
    for n in m.graph.node:
        if n.op_type == "DequantizeLinear":
            x = n.input[0]
            k = "常量" if x in init else ("DQL?" if producer.get(x) is not None else "图输入")
            print("DequantizeLinear:", n.name, "| x:", x, "|", k)
    dql_n = sum(1 for n in m.graph.node if n.op_type == "DynamicQuantizeLinear")
    print("DynamicQuantizeLinear:", dql_n)


def convert(in_path, out_path):
    m = onnx.load(in_path, load_external_data=False)
    g = m.graph
    R, init, _ = collect(g)

    drop_nodes = []
    new_inits = []
    for r in R:
        W = numpy_helper.to_array(init[r["B"]]).astype(np.float32)
        zp = (numpy_helper.to_array(init[r["zp_b"]]).astype(np.float32)
              if r["zp_b"] else np.float32(0))
        ws = numpy_helper.to_array(init[r["w_scale"]]).astype(np.float32)
        Wf = (W - zp) * ws
        name = r["B"] + "_f32"
        new_inits.append(numpy_helper.from_array(np.ascontiguousarray(Wf), name))

        mul = r["mul"]
        mul.op_type = "MatMul" if r["kind"] == "MatMulInteger" else "Conv"
        del mul.input[:]
        mul.input.extend([r["A_f32"], name])
        if r["kind"] == "ConvInteger":
            del mul.attribute[:]
            mul.attribute.extend(r["node"].attribute)  # kernel_shape/pads/strides/group 全盘照抄
        drop_nodes += [r["node"], r["cast"]]

    drop_ids = {id(x) for x in drop_nodes}
    keep = [n for n in g.node if id(n) not in drop_ids]
    del g.node[:]
    g.node.extend(keep)
    g.initializer.extend(new_inits)

    # 死节点清扫：没有消费者、也不是图输出的，连同其专用常量一起删（DQL、scale 乘会在这里消失）
    gout = {o.name for o in g.output}
    while True:
        used = Counter()
        for n in g.node:
            for i in n.input:
                if i:
                    used[i] += 1
        keep = [n for n in g.node
                if any((o in gout) or used[o] > 0 for o in n.output)]
        if len(keep) == len(g.node):
            break
        del g.node[:]
        g.node.extend(keep)

    used = set()
    for n in g.node:
        used.update(i for i in n.input if i)
    keep_init = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(keep_init)

    ops = Counter(n.op_type for n in g.node)
    print("重写后 ops:", ", ".join(f"{k}:{v}" for k, v in ops.most_common(12)))
    left = {k: ops[k] for k in ("MatMulInteger", "ConvInteger", "DynamicQuantizeLinear",
                                "DequantizeLinear") if ops[k]}
    print("残留量化算子:", left or "无")

    try:
        onnx.checker.check_model(m, full_check=False)
        print("checker ok")
    except Exception as e:  # 大模型 checker 可能因 2GB 序列化限制先炸，最终以 ORT 能载入为准
        print("checker 警告:", type(e).__name__, str(e)[:200])
    # onnx 写外置权重对**已存在**的 .data 是追加而不是覆盖：同一目录重跑一次，.data 会变成
    # 两倍大、图里的 offset 也跟着变（2026-10-06 校机实测：第二次建出 957,649 B 图 +
    # 6,206,335,232 B 权重，产线是 956,747 / 3,103,167,616）。先删掉上一轮的产物，
    # 保证"在同一目录里重建"和"在全新目录里建"得到同一份文件。
    for stale in (out_path, out_path + ".data"):
        try:
            os.remove(stale)
        except FileNotFoundError:
            pass
    onnx.save_model(m, out_path, save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=out_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] + ".data",
                    size_threshold=1024)
    print("saved", out_path)


def main():
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "survey":
        survey(args[0])
    elif mode == "convert":
        convert(args[0], args[1])
    else:
        raise SystemExit("mode: survey|convert")


if __name__ == "__main__":
    main()
