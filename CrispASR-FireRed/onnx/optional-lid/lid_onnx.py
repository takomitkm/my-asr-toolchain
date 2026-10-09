#!/usr/bin/env python3
"""FireRedLID 的纯 ONNXRuntime 推理：不依赖 sherpa-onnx，也不依赖 torch 运行时。

结构：lid_encoder.onnx（编码器整图）+ lid_decoder_step.onnx（解码器单步，整前缀重算）。
beam 记账是官方 TransformerDecoder.batch_beam_search 的逐行移植（transformer_decoder.py:40-158），
只把 torch 换成 numpy（没装 torch 的运行环境也能跑），差别两处且都不改数值：
  1) 每步解码的前向交给 ONNX 图（官方用 cache，这里整前缀重算；S<=decode_max_len=2 时等价）；
  2) 编码器的 for 循环 padding mask、解码器的 uint8 位与 mask 换成 0/1 乘法（导出时已断言与
     官方逐元素相同；ORT 没注册 Equal 的 uint8/float kernel，只认 int64）。
对拍（lid_check.log）：编码器输出最大绝对差 3.8e-06，>1e-4 的格子 0，mask 全同，
beam 的 ids 与 confidence 与官方 torch 到小数点后 6 位一致。

三种用法：
  LidOnnx(...).process(uttids, [(sr, int16wav), ...])   整段级，官方协议
  LidOnnx(...).process_windows(wav, sr, win_s, hop_s)   帧块级：逐窗解码，带窗时间戳
  LidOnnx(...).segment_vote(windows, t0, t1)            把窗序列汇成 ASR 的段级标签

音频量纲必须是 int16 幅度（官方 kaldiio.load_mat 读出来的那个尺度）。喂 [-1,1] 的 float 会让
log-mel 能量整体掉到下限，实测同一句话从 "zh mandarin"(0.9994) 变成 "nn"(0.3287)。
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

INF = 1e10


class FeatNumpy:
    """官方 fireredlid.data.feat.FeatExtractor 的去 torch 复刻：同一个 kaldi_native_fbank、
    同一组 FbankOptions（dither=0 / snip_edges=True / 80 mel）、同一个 cmvn.ark。
    官方类顶部 import torch，而运行环境（env）里没有 torch，所以自己写；
    与官方输出的逐元素对拍见 lid_check.py 的 [feat] 行。"""

    def __init__(self, cmvn_path):
        import kaldi_native_fbank as knf
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0.0
        opts.mel_opts.num_bins = 80
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.knf, self.opts = knf, opts
        self.means, self.istd = self._read_cmvn(cmvn_path)

    @staticmethod
    def _read_cmvn(path):
        import kaldiio
        stats = np.asarray(kaldiio.load_mat(path), dtype=np.float64)
        assert stats.shape[0] == 2, "cmvn.ark 应该是 2 行统计量"
        dim = stats.shape[1] - 1
        count = stats[0, dim]
        assert count >= 1
        means = stats[0, :dim] / count
        var = np.maximum(stats[1, :dim] / count - means * means, 1e-20)
        return means, 1.0 / np.sqrt(var)

    def fbank(self, sr, x):
        f = self.knf.OnlineFbank(self.opts)
        f.accept_waveform(sr, np.asarray(x).tolist())
        n = f.num_frames_ready
        if n < 1:
            return None
        return np.vstack([f.get_frame(i) for i in range(n)])

    def __call__(self, batch_wav, uttids=None):
        rows, lengths, durs = [], [], []
        for (sr, x), u in zip(batch_wav, uttids or [None] * len(batch_wav)):
            v = self.fbank(sr, x)
            if v is None or v.shape[0] < 1:
                continue
            rows.append(((v - self.means) * self.istd).astype(np.float32))
            lengths.append(rows[-1].shape[0])
            durs.append(np.asarray(x).shape[0] / sr)
        if not rows:
            return None, None, [], [], []
        pad = np.zeros((len(rows), max(lengths), 80), dtype=np.float32)
        for i, r in enumerate(rows):
            pad[i, :lengths[i]] = r
        return pad, np.asarray(lengths, dtype=np.int64), durs, list(batch_wav), list(uttids or [])


class LidOnnx:
    def __init__(self, model_dir, lid_dir=None, ort_threads=18, smoothing=1.25,
                 beam_size=3, nbest=1, decode_max_len=2, length_penalty=0.6,
                 eos_penalty=1.0):
        import onnxruntime as ort
        self.ort = ort
        lid_dir = (lid_dir or os.environ.get("FIREDASR_LID_DICT")
                   or str(Path(__file__).resolve().parent.parent / "models" / "FireRedLID"))
        self.tokenizer = self._load_tokenizer(os.path.join(lid_dir, "dict.txt"))
        self.feat = self._load_feat(lid_dir)
        so = ort.SessionOptions()
        so.intra_op_num_threads = ort_threads
        so.inter_op_num_threads = 1
        self.enc = ort.InferenceSession(os.path.join(model_dir, "lid_encoder.onnx"),
                                        sess_options=so,
                                        providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(os.path.join(model_dir, "lid_decoder_step.onnx"),
                                        sess_options=so,
                                        providers=["CPUExecutionProvider"])
        self.sos_id = self.tokenizer["<sos>"]
        self.eos_id = self.tokenizer["<eos>"]
        self.pad_id = self.tokenizer["<pad>"]
        self.cfg = dict(beam_size=beam_size, nbest=nbest, decode_max_len=decode_max_len,
                        softmax_smoothing=smoothing, length_penalty=length_penalty,
                        eos_penalty=eos_penalty)
        self.times = {"feat": 0.0, "enc": 0.0, "dec": 0.0}

    # ---------- 前端 ----------
    @staticmethod
    def _load_tokenizer(dict_path):
        id2w, w2id = [], {}
        with open(dict_path, encoding="utf8") as f:
            for i, line in enumerate(f):
                t = line.strip().split()
                w, idx = (t[0], int(t[1])) if len(t) >= 2 else (t[0], i)
                w2id[w] = idx
                id2w.append(w)
        return {"id2word": id2w, "word2id": w2id, **w2id}

    @staticmethod
    def _load_feat(lid_dir):
        return FeatNumpy(os.path.join(lid_dir, "cmvn.ark"))

    def feats(self, batch_wav):
        t0 = time.perf_counter()
        feats, lengths = self.feat(batch_wav, ["u"] * len(batch_wav))[:2]
        self.times["feat"] += time.perf_counter() - t0
        return np.asarray(feats, dtype=np.float32), np.asarray(lengths, dtype=np.int64)

    # ---------- 两张图 ----------
    def encode(self, feats, lengths):
        t0 = time.perf_counter()
        eo, sm = self.enc.run(None, {"feats": np.ascontiguousarray(feats),
                                     "lengths": np.ascontiguousarray(lengths)})
        self.times["enc"] += time.perf_counter() - t0
        return eo, sm

    def dec_step(self, ys, enc_out, src_mask):
        t0 = time.perf_counter()
        lg = self.dec.run(None, {"ys": np.ascontiguousarray(ys, dtype=np.int64),
                                 "enc_output": np.ascontiguousarray(enc_out),
                                 "src_mask": np.ascontiguousarray(src_mask)})[0]
        self.times["dec"] += time.perf_counter() - t0
        return lg

    # ---------- 官方 beam 协议（numpy 逐行移植） ----------
    @staticmethod
    def _topk(x, k):
        idx = np.argsort(-x, axis=-1, kind="stable")[..., :k]
        return np.take_along_axis(x, idx, axis=-1), idx

    @staticmethod
    def _log_softmax(x):
        m = x.max(axis=-1, keepdims=True)
        return (x - m) - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))

    def _finished_score(self, scores, is_finished):
        NB, B = scores.shape
        mask = np.array([0.0] + [-INF] * (B - 1), dtype=np.float32).reshape(1, B).repeat(NB, 0)
        return scores * (1 - is_finished) + mask * is_finished

    def _finished_y(self, ys, is_finished):
        fin = is_finished.astype(ys.dtype)
        return ys * (1 - fin) + self.eos_id * fin

    def batch_beam_search(self, encoder_outputs, src_masks, beam_size=3, nbest=1,
                          decode_max_len=2, softmax_smoothing=1.25,
                          length_penalty=0.6, eos_penalty=1.0):
        B = int(beam_size)
        N, Ti, H = encoder_outputs.shape
        maxlen = int(decode_max_len) if decode_max_len > 0 else Ti
        assert eos_penalty > 0.0

        enc = np.repeat(encoder_outputs[:, None], B, axis=1).reshape(N * B, Ti, H)
        src_mask = np.repeat(src_masks[:, None], B, axis=1).reshape(N * B, -1, Ti)
        ys = np.full((N * B, 1), self.sos_id, dtype=np.int64)
        confidences = np.zeros((N * B, 1), dtype=np.float32)
        scores = np.tile(np.array([0.0] + [-INF] * (B - 1), dtype=np.float32),
                         N).reshape(N * B, 1)
        is_finished = np.zeros((N * B, 1), dtype=np.float32)

        for t in range(maxlen):
            t_logit = self.dec_step(ys, enc, src_mask)
            t_scores = self._log_softmax(t_logit / softmax_smoothing)
            if eos_penalty != 1.0:
                t_scores[:, self.eos_id] *= eos_penalty

            t_topB_scores, t_topB_ys = self._topk(t_scores, B)
            t_topB_scores = self._finished_score(t_topB_scores, is_finished)
            t_topB_ys = self._finished_y(t_topB_ys, is_finished)

            scores = scores + t_topB_scores
            scores, topB_score_ids = self._topk(scores.reshape(N, B * B), B)
            scores = scores.reshape(-1, 1)

            row_in_B = (topB_score_ids // B).reshape(N * B)
            stride = B * np.arange(N).reshape(N, 1).repeat(B).reshape(N * B)
            row = (row_in_B + stride).astype(np.int64)

            t_ys = np.take_along_axis(t_topB_ys.reshape(N, B * B),
                                       topB_score_ids, axis=1).reshape(N * B, 1)
            ys = np.concatenate([ys[row], t_ys], axis=1)

            t_conf = np.take_along_axis(t_topB_scores.reshape(N, B * B),
                                        topB_score_ids, axis=1).reshape(N * B, 1)
            confidences = np.concatenate([confidences[row], np.exp(t_conf)], axis=1)

            is_finished = (ys[:, -1] == self.eos_id).astype(np.float32).reshape(-1, 1)
            if is_finished.sum() == N * B:
                break

        scores = scores.reshape(N, B)
        ys = ys.reshape(N, B, -1)
        ys_lengths = np.sum(ys != self.eos_id, axis=-1).astype(np.int32)
        if length_penalty > 0.0:
            scores = scores / np.power((5 + ys_lengths.astype(np.float32)) / 6.0, length_penalty)
        nbest_scores, nbest_ids = self._topk(scores, int(nbest))
        nbest_scores = -1.0 * nbest_scores
        index = nbest_ids + B * np.arange(N).reshape(N, 1)
        nbest_ys = ys.reshape(N * B, -1)[index.reshape(-1)].reshape(N, nbest_ids.shape[1], -1)
        nbest_len = ys_lengths.reshape(N * B)[index.reshape(-1)].reshape(N, -1)
        nbest_conf = confidences.reshape(N * B, -1)[index.reshape(-1)].reshape(
            N, nbest_ids.shape[1], -1)

        out = []
        for n in range(N):
            hs = []
            for i in range(nbest_scores.shape[1]):
                L = int(nbest_len[n, i])
                c = nbest_conf[n, i, 1:L]
                hs.append({"yseq": nbest_ys[n, i, 1:L],
                           "confidence": float(c.mean()) if c.size else float("nan")})
            out.append(hs)
        return out

    # ---------- 对外 ----------
    def detok(self, ids):
        return " ".join(self.tokenizer["id2word"][int(i)] for i in ids)

    def _hyps_to_result(self, uttid, dur, hyp):
        ids = [int(i) for i in hyp[0]["yseq"]]
        return {"uttid": uttid, "lang": self.detok(ids),
                "confidence": round(float(hyp[0]["confidence"]), 3),
                "dur_s": round(dur, 3),
                "allnbest": [{"lang": self.detok([int(i) for i in h["yseq"]]),
                              "confidence": round(float(h["confidence"]), 4)} for h in hyp]}

    def process(self, batch_uttid, batch_wav):
        """整段级：官方协议（beam3 / 最多 2 token）。逐条编码 —— 编码器图在"各行长度不同"的
        批次上会撞 RelPos 的 Reshape 折叠（等长批次实测逐元素相同），所以不做混长 batching。"""
        out = []
        for uttid, (sr, x) in zip(batch_uttid, batch_wav):
            x = np.asarray(x)
            feats, lengths = self.feats([(sr, x)])
            eo, sm = self.encode(feats, lengths)
            hyps = self.batch_beam_search(eo, sm, **self.cfg)
            out.append(self._hyps_to_result(uttid, x.shape[0] / sr, hyps[0]))
        return out

    # ---------- 帧块级 ----------
    FBANK_RATE = 100.0   # fbank 帧率（10ms/帧）
    FRAME_RATE = 25.0    # 编码器输出帧率 = 100/4（40ms/帧）

    def process_windows(self, wav, sr, win_s=3.0, hop_s=1.5, mode="slice"):
        """滑动窗逐窗判语种，返回带时间戳的窗序列（按帧精度的 LID）。

        mode="slice"    编码器整条只跑一次（窗之间共享上下文），窗＝编码器输出的时间切片，
                        每窗只重跑解码器。最省，特征与官方"整条编码"完全一致。
        mode="reencode" 每窗单独提特征+编码，看不到左右上下文，等价于独立短句。
        """
        win = max(2, int(round(win_s * self.FRAME_RATE)))
        hop = max(1, int(round(hop_s * self.FRAME_RATE)))
        spans = []

        if mode == "slice":
            feats, lengths = self.feats([(sr, wav)])
            eo, sm = self.encode(feats, lengths)
            valid = int(sm[0, 0].sum())
            if valid < 2:
                return []
            starts = list(range(0, max(1, valid - win + 1), hop))
            if starts[-1] + win < valid:
                starts.append(max(0, valid - win))
            for a in starts:
                b = min(valid, a + win)
                hyp = self.batch_beam_search(eo[:, a:b], sm[:, :, a:b], **self.cfg)[0][0]
                spans.append((a / self.FRAME_RATE, b / self.FRAME_RATE, hyp))
        else:
            nwin = int(round(win_s * sr))
            step = int(round(hop_s * sr))
            n = int(wav.shape[0])
            a = 0
            while a < n:
                seg = wav[a:a + nwin] if a + nwin <= n else wav[a:]
                if seg.shape[0] < int(1.5 * sr):
                    break
                feats, lengths = self.feats([(sr, seg)])
                eo, sm = self.encode(feats, lengths)
                he = int(sm[0, 0].sum())
                hyp = self.batch_beam_search(eo[:, :he], sm[:, :, :he], **self.cfg)[0][0]
                spans.append((a / sr, a / sr + he / self.FRAME_RATE, hyp))
                if a + nwin >= n:
                    break
                a += step

        return [{"start_s": round(s, 3), "end_s": round(e, 3),
                 "lang": self.detok([int(t) for t in h["yseq"]]),
                 "confidence": round(float(h["confidence"]), 4)}
                for s, e, h in spans]

    def segment_vote(self, windows, t0, t1):
        """把落在 [t0,t1) 内的窗按 重叠时长×置信度 加权求和，取总分最高的语种（段级标签）。"""
        acc, cnt = {}, 0
        for w in windows:
            ov = min(w["end_s"], t1) - max(w["start_s"], t0)
            if ov <= 0:
                continue
            cnt += 1
            acc[w["lang"]] = acc.get(w["lang"], 0.0) + ov * w["confidence"]
        if not acc:
            return {"lang": "", "windows": 0, "score": 0.0}
        lang, score = max(acc.items(), key=lambda kv: kv[1])
        return {"lang": lang, "windows": cnt, "score": round(score, 4)}
