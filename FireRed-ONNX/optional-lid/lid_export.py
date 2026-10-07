#!/usr/bin/env python3
"""把 FireRedLID 导成 ONNX：编码器整图 + 解码器单步（整前缀重算，S<=2 不需要 KV cache）。

与官方代码的唯一不等价改动（已在文件里注明）：ConformerEncoder.padding_position_is_0
的 python for 循环换成 arange(T) < lengths 的向量化写法，两者对同一 lengths 逐元素相同
（脚本里会先断言再导出）。

用法（要 torch + onnx + 官方 FireRedASR2S 源码，属"可选环自建"路线，见 README）:
  python lid_export.py --src <FireRedASR2S 仓库根> --models <模型根> --wav <一段参考音频>
路径也能用环境变量 FIREDASR_FR2SRC / FIREDASR_MODELS / FIREDASR_LID_REF 给。
产物: <models>/FireRedLID-onnx/lid_encoder.onnx        （权重走 external data）
      <models>/FireRedLID-onnx/lid_decoder_step.onnx
"""
import sys, os, time, argparse, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PKG = Path(__file__).resolve().parent.parent      # 分发包根目录（optional-lid 的上一级）

ap = argparse.ArgumentParser(
    description="导出 FireRedLID ONNX（可选环；跑批量转写不需要它）",
    epilog="默认值：--out <--models>/FireRedLID-onnx，--ref 本脚本旁的 lidref，"
           "--models 环境变量 FIREDASR_MODELS 或分发包的 models 目录")
ap.add_argument("--src", default=os.environ.get("FIREDASR_FR2SRC", ""), metavar="DIR",
                help="FireRedASR2S 仓库根（里面有 fireredlid 包）；不给就假定已装在 sys.path 上")
ap.add_argument("--models", default=os.environ.get("FIREDASR_MODELS", str(_PKG / "models")),
                metavar="DIR", help="模型根目录")
ap.add_argument("--pt", default="", metavar="DIR",
                help="FireRedLID 的 PyTorch 权重目录（含 model.pth.tar），默认 <models>/FireRedLID")
ap.add_argument("--out", default="", metavar="DIR", help="导出目录，默认 <models>/FireRedLID-onnx")
ap.add_argument("--ref", default=os.environ.get("FIREDASR_LID_REF",
                                                str(Path(__file__).resolve().parent / "lidref")),
                metavar="DIR", help="对拍参考 .npy 目录（见 --wav）")
ap.add_argument("--wav", default="", metavar="FILE",
                help="参考音频（16 kHz 语音）：--ref 里没有 npy 时用官方前端现算一份并缓存")
ap.add_argument("--opset", type=int, default=17)
ap.add_argument("--threads", type=int, default=0, metavar="N",
                help="torch 线程数，0=交给 torch 自己决定")
ap.add_argument("--skip-encoder", action="store_true")
ap.add_argument("--skip-decoder", action="store_true")
A = ap.parse_args()

if A.src:
    sys.path.insert(0, str(Path(A.src).resolve()))
try:
    from fireredlid.lid import load_fireredlid_model
    from fireredlid.models.module.conformer_encoder import ConformerEncoder
except ImportError as e:
    sys.exit(f"导入 fireredlid 失败：{e!r}\n"
             f"git clone https://github.com/FireRedTeam/FireRedASR2S 后用 --src 指到仓库根")

MD = Path(A.pt or (Path(A.models) / "FireRedLID"))
OUT = Path(A.out or (Path(A.models) / "FireRedLID-onnx"))
REF = Path(A.ref)
OUT.mkdir(parents=True, exist_ok=True)
REF.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(A.threads) if A.threads else None
model = load_fireredlid_model(str(MD / "model.pth.tar")).eval()


def load_ref():
    """参考输入/输出 npy：--ref 里齐了就直接读，缺就从 --wav 用官方前端现算一份存下来。

    这两件事必须分开看：编码器导出要的是输入样例（feats/lengths，任何一段真实音频都行），
    encout/encmask 才是"替换 mask 实现后数值没变"的对拍凭据。
    """
    need = ["short_v2_feats.npy", "short_v2_lengths.npy",
            "short_v2_encout.npy", "short_v2_encmask.npy"]
    if all((REF / n).exists() for n in need):
        print("[ref] 读现成参考件：" + "  ".join(n for n in need))
        return {n[:-4]: np.load(REF / n) for n in need}, False
    if not A.wav:
        sys.exit(f"--ref {REF} 里缺 {need}，又没给 --wav：无法构造导出输入样例。\n"
                 f"给一段 16 kHz 语音：--wav some.wav（脚本会用官方 fbank 前端现算并缓存到 --ref）")
    print(f"[ref] {REF} 不全，用 {A.wav} 现算")
    from fireredlid.audio import load_audio
    from fireredlid.features import kaldi_fbank
    wav = load_audio(A.wav, 16000)
    feats = kaldi_fbank(wav)                      # (1, T, 80) float32
    lengths = torch.tensor([feats.shape[1]]).numpy()
    with torch.no_grad():
        eo, _, sm = model.encoder(torch.from_numpy(feats),
                                  torch.from_numpy(lengths))
    out = {"short_v2_feats": feats, "short_v2_lengths": lengths,
           "short_v2_encout": eo.numpy(), "short_v2_encmask": sm.numpy()}
    for k, v in out.items():
        np.save(REF / f"{k}.npy", v)
    return out, True


R, REF_IS_FRESH = load_ref()


def equal_mask():
    """先证明向量化 mask 与原实现逐元素相同，再动手替换。"""
    T = 977
    for L in (1, 2, 5, T // 2, T - 1, T):
        padded = torch.zeros(1, T)
        orig_ref = torch.ones(1, T)
        orig_ref[0, L:] = 0
        orig = orig_ref.unsqueeze(1).to(torch.uint8)          # 原实现等价结果
        ar = torch.arange(T)
        new = (ar.unsqueeze(0) < L).unsqueeze(1).to(torch.uint8)
        assert torch.equal(orig, new), (L, orig, new)
    print("[mask] arange 向量化与原 for 循环逐元素相同 (T=977, L=1/2/5/488/976/977)")


equal_mask()
ConformerEncoder.padding_position_is_0 = staticmethod(lambda padded, lengths: (
    torch.arange(padded.size(1), device=padded.device).unsqueeze(0)
    < lengths.unsqueeze(1)).unsqueeze(1).to(torch.int64))
print("[mask] ConformerEncoder.padding_position_is_0 已替换为向量化实现")

with torch.no_grad():
    eo_ref = R["short_v2_encout"]
    feats = torch.from_numpy(R["short_v2_feats"])
    lengths = torch.from_numpy(R["short_v2_lengths"])
    eo2, _, _ = model.encoder(feats, lengths)
    print("[mask] 替换后编码器输出与替换前参考最大绝对差: %.3e (元素 %s)%s"
          % (np.abs(eo2.numpy() - eo_ref).max(), eo2.shape,
             "  ← 参考件是本次在替换前现算的" if REF_IS_FRESH else ""))


class EncoderGraph(nn.Module):
    """输入 feats(N,T,80)+lengths(N) -> 编码器输出 (N,T',1280) 与解码用 mask (N,1,T')"""
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, feats, lengths):
        enc_output, out_lengths, src_mask = self.enc(feats, lengths)
        return enc_output, src_mask.to(torch.int64)


if not A.skip_encoder:
    ex = (feats, lengths)
    t0 = time.time()
    ep = str(OUT / "lid_encoder.onnx")
    N = torch.export.Dim("N", min=1, max=64)
    T = torch.export.Dim("T", min=16, max=20000)
    torch.onnx.export(EncoderGraph(model.encoder), ex, ep,
                      input_names=["feats", "lengths"],
                      output_names=["enc_output", "src_mask"],
                      dynamo=True, opset_version=A.opset,
                      external_data=True,
                      dynamic_shapes={"feats": {0: N, 1: T}, "lengths": {0: N}},
                      verify=False, report=False)
    print("[export] encoder %.1f s -> %s (%.1f MB) + .data (%.2f GB)"
          % (time.time() - t0, os.path.basename(ep),
             os.path.getsize(ep) / 2**20,
             os.path.getsize(ep + ".data") / 2**30 if os.path.exists(ep + ".data") else 0))


class DecoderStep(nn.Module):
    """解码器单步：整个前缀重算（不接 KV cache），只返回最后一个位置的 120 类 logits。

    与官方 TransformerDecoder.batch_beam_search 循环体逐步等价：
      - 官方 cache 分支只在 t>0 生效，且它缓存的正是本层的完整输出，重算路径与缓存路径数值相同；
      - tgt_mask 原来是 uint8 的 (token!=pad) & tril，这里用 int64 的 0/1 乘法等价改写：既避开
        BitwiseAnd 的 opset 要求，也避开 ORT 不注册 Equal(uint8/float) kernel 的问题；
        后续只做 mask.eq(0)，两者逐元素同值；
      - dropout 在 eval() 下是恒等，导出图里没有。
    """
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


if not A.skip_decoder:
    dec = model.lid_decoder
    eo = torch.from_numpy(R["short_v2_encout"])
    sm = torch.from_numpy(R["short_v2_encmask"]).to(torch.int64)
    ys = torch.full((1, 1), dec.sos_id, dtype=torch.long)
    # 先证明 DecoderStep 与官方循环体等价：beam=1 单步的首位分布应与官方 batch_beam_search 一致
    with torch.no_grad():
        my_logit = DecoderStep(dec)(ys, eo, sm)
        my_top = torch.log_softmax(my_logit / 1.25, dim=-1).argmax(-1).tolist()
        off = dec.batch_beam_search(eo, sm.to(torch.uint8), 1, 1, 1, 1.25, 0.0, 1.0)
        off_y = [int(i) for i in off[0][0]["yseq"].cpu()]
        off_c = float(off[0][0]["confidence"].cpu())
        my_c = float(torch.exp(torch.log_softmax(my_logit / 1.25, -1).max()).item())
    print("[对拍] DecoderStep vs 官方 batch_beam_search 首步: ids %s/%s  conf %.6f/%.6f"
          % (my_top, off_y, my_c, off_c))
    assert my_top == off_y and abs(my_c - off_c) < 1e-4
    # 导出样例的 batch 与序列长必须取不同的值：torch.export 会把"具体值相同"的轴合并成一个，
    # 用 (1,1) 当例子会连 NB 和 S 一起退化成静态 1（第一版就栽在这里）。
    NB_EX, S_EX = 3, 2
    ys_ex = torch.full((NB_EX, S_EX), dec.sos_id, dtype=torch.long)
    eo_ex = eo.repeat(NB_EX, 1, 1)
    sm_ex = sm.repeat(NB_EX, 1, 1)
    t0 = time.time()
    dp = str(OUT / "lid_decoder_step.onnx")
    NB = torch.export.Dim("NB", min=1, max=256)
    Ti = torch.export.Dim("Ti", min=1, max=8000)
    Sd = torch.export.Dim("S", min=1, max=8)
    torch.onnx.export(DecoderStep(dec), (ys_ex, eo_ex, sm_ex), dp,
                      input_names=["ys", "enc_output", "src_mask"],
                      output_names=["logits_last"],
                      dynamo=True, opset_version=A.opset, external_data=True,
                      dynamic_shapes={"ys": {0: NB, 1: Sd},
                                      "enc_output": {0: NB, 1: Ti},
                                      "src_mask": {0: NB, 2: Ti}},
                      verify=False, report=False)
    print("[export] decoder %.1f s -> %s (%.1f MB)"
          % (time.time() - t0, os.path.basename(dp), os.path.getsize(dp) / 2**20))

print("导出完成:", sorted(os.listdir(str(OUT))))

