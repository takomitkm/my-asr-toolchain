#!/usr/bin/env python3
"""只重导解码器单步图，并把 dynamic_shapes 改成"按位置"的元组写法（字典写法在这一版 torch
上把 ys 的第 0 维绑错了，报 ConstraintViolationError）。是 lid_export.py 的 decoder-only 快捷版。"""
import sys, os, time, argparse
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PKG = Path(__file__).resolve().parent.parent      # 分发包根目录

ap = argparse.ArgumentParser(description="只重导 FireRedLID 解码器单步 ONNX（可选环）")
ap.add_argument("--src", default=os.environ.get("FIREDASR_FR2SRC", ""), metavar="DIR",
                help="FireRedASR2S 仓库根（里面有 fireredlid 包）")
ap.add_argument("--models", default=os.environ.get("FIREDASR_MODELS", str(_PKG / "models")),
                metavar="DIR", help="模型根目录")
ap.add_argument("--pt", default="", metavar="DIR",
                help="FireRedLID PyTorch 权重目录，默认 <models>/FireRedLID")
ap.add_argument("--out", default="", metavar="DIR",
                help="导出目录，默认 <models>/FireRedLID-onnx")
ap.add_argument("--ref", default=os.environ.get("FIREDASR_LID_REF",
                                                str(Path(__file__).resolve().parent / "lidref")),
                metavar="DIR", help="对拍参考 .npy 目录（lid_export.py --wav 会生成）")
ap.add_argument("--opset", type=int, default=18)
ap.add_argument("--threads", type=int, default=0, help="torch 线程数，0=torch 自己定")
A = ap.parse_args()

if A.src:
    sys.path.insert(0, str(Path(A.src).resolve()))
sys.path.insert(0, str(_PKG))          # 与链路同源的实现（load_audio 等）

import numpy as np
import torch
import torch.nn as nn
try:
    from fireredlid.lid import load_fireredlid_model
    from fireredlid.models.module.conformer_encoder import ConformerEncoder
except ImportError as e:
    sys.exit(f"导入 fireredlid 失败：{e!r}\n先 git clone FireRedASR2S，再用 --src 指到仓库根")

MD = Path(A.pt or (Path(A.models) / "FireRedLID"))
OUT = Path(A.out or (Path(A.models) / "FireRedLID-onnx"))
REF = Path(A.ref)
OUT.mkdir(parents=True, exist_ok=True)

ConformerEncoder.padding_position_is_0 = staticmethod(lambda padded, lengths: (
    torch.arange(padded.size(1), device=padded.device).unsqueeze(0)
    < lengths.unsqueeze(1)).unsqueeze(1).to(torch.int64))

if A.threads:
    torch.set_num_threads(A.threads)
model = load_fireredlid_model(str(MD / "model.pth.tar")).eval()
dec = model.lid_decoder

class DecoderStep(nn.Module):
    """解码器单步：整个前缀重算（不接 KV cache），只返回最后一个位置的 120 类 logits。
    与官方 batch_beam_search 循环体逐步等价（详见 lid_export.py 同名类的注释）。"""
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, ys, enc_output, src_mask):
        dec = self.dec
        S = ys.size(1)
        # 官方是 (token!=pad).unsqueeze(1) & tril(S,S) 的 uint8 位与，结果 (NB,S,S)；
        # 这里换成 0/1 乘法 + arange 因果比较，避开 BitwiseAnd 和 ORT 没注册的 Equal(uint8/float)。
        ar = torch.arange(S, device=ys.device)
        tril = (ar.view(1, S) <= ar.view(S, 1)).to(torch.int64)
        valid = (ys != dec.pad_id).to(torch.int64).unsqueeze(1)
        tgt_mask = valid * tril
        x = dec.tgt_word_emb(ys) * dec.scale + dec.positional_encoding(ys)
        for layer in dec.layer_stack:
            x = layer.forward(x, enc_output, tgt_mask, src_mask, cache=None)
        x = dec.layer_norm_out(x)
        return dec.tgt_word_prj(x[:, -1])


NB_EX, S_EX = 3, 2
# 先证明 mask 改写与官方逐元素相同（含 pad 行），再拿去导出
for _S in (1, 2, 3):
    _ys = torch.full((3, _S), dec.sos_id, dtype=torch.long)
    if _S > 1:
        _ys[1, 1:] = dec.pad_id
    _off = dec.ignored_target_position_is_0(_ys, dec.pad_id).to(torch.int64)
    _ar = torch.arange(_S)
    _tril = (_ar.view(1, _S) <= _ar.view(_S, 1)).to(torch.int64)
    _mine = (_ys != dec.pad_id).to(torch.int64).unsqueeze(1) * _tril
    assert torch.equal(_off, _mine), (_S, tuple(_off.shape), tuple(_mine.shape))
print("[mask] 因果+pad mask 改写与官方 ignored_target_position_is_0 逐元素相同 (NB=3,S=1/2/3,含 pad 行)")

ys_ex = torch.full((NB_EX, S_EX), dec.sos_id, dtype=torch.long)
eo_ex = torch.from_numpy(np.load(REF / "short_v2_encout.npy")).repeat(NB_EX, 1, 1)
sm_ex = torch.from_numpy(np.load(REF / "short_v2_encmask.npy")).to(torch.int64).repeat(NB_EX, 1, 1)

NB = torch.export.Dim("NB", min=1, max=256)
Ti = torch.export.Dim("Ti", min=1, max=8000)
Sd = torch.export.Dim("S", min=1, max=8)

dp = str(OUT / "lid_decoder_step.onnx")
t0 = time.time()
torch.onnx.export(DecoderStep(dec), (ys_ex, eo_ex, sm_ex), dp,
                  input_names=["ys", "enc_output", "src_mask"],
                  output_names=["logits_last"],
                  dynamo=True, opset_version=A.opset, external_data=True,
                  dynamic_shapes=({0: NB, 1: Sd}, {0: NB, 1: Ti}, {0: NB, 2: Ti}),
                  verify=False, report=False)
print("[export] decoder %.1f s -> %.1f KB" % (time.time() - t0, os.path.getsize(dp) / 1024))

import onnx
g = onnx.load(dp).graph
for v in list(g.input) + list(g.output):
    t = v.type.tensor_type
    print(" ", v.name, "elem", t.elem_type,
          [str(d.dim_value) if d.HasField("dim_value") else "dyn:" + d.dim_param for d in t.shape.dim])
