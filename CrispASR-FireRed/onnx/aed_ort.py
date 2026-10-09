"""纯 onnxruntime 跑小红书 FireRedASR2-AED，不用 sherpa_onnx 的 C++ 运行时。

模型是 k2-fsa 从 FireRedTeam/FireRedASR2-AED 转出来的那两份 ONNX（encoder 出 16 层
cross-K/V，decoder 是单步图）。图有两处硬约定，都是实测出来的，照抄 sherpa 的 C++ 用法
（sherpa-onnx/csrc/offline-fire-red-asr-model.cc + greedy-search-decoder.cc）：

1. **自注意力 KV cache 是预分配 + 原地写**：`in_n_layer_self_k_cache` 一开始就要给足长度
   （`ScatterND` 把新 K/V 写到第 `offset` 个位置，输出张量长度不变）。喂 T=0 的空缓存会在
   `/Reshape_10` 直接崩：`input {1,0,20,64} -> requested {1,1,20,20,64}`；T≥1 才对。
   offset 从 0 起，每步 +1。
2. **tokens 是静态 [1,1]**，一次一条束；B 条束就是 B 次图调用（官方 torch 的 batch 维没导出来）。

前端照 fr2src/fireredasr2 与 sherpa 两处一致的参数：80 mel / 25 ms 窗 / 10 ms 移 / dither=0 /
snip_edges=True / high_freq=Nyquist，再全局 CMVN `(f-mean)*istd`（mean/istd 直接从 encoder 图的
metadata 读，转储在 lidref/aed_encoder_meta.json）。幅度必须是 int16 尺度（×32768）：sherpa 侧
`normalize_samples=false` 用的就是原始整数尺度。

解码两种模式：
- `greedy`：逐字 argmax、遇 eos 停，和 sherpa 的 greedy-search-decoder 逐字对齐（对拍用）。
- `beam`  ：官方 `transformer_decoder.batch_beam_search` 的 numpy 复刻 —— beam_size=3、
  nbest=1、decode_max_len=0(→编码器帧数 maxlen)、softmax_smoothing=1.25、length_penalty=0.6、
  eos_penalty=1.0，finished 束的分数按官方那样钉成 [0,-INF,...] 并把 token 钉成 eos。

文本 `aed_tokenizer.detokenize`：id→token 直接串，`▁`→空格，去 `<blank>`/`<sil>`，转小写。
"""
import os
import re
import time

import numpy as np

INF = 1e10
SUFFIX = {"int8": "int8.onnx", "f32": "f32.onnx", "f16": "f16.onnx"}
# mixed = 实测出来的组合拳：encoder f32（AVX-512 下比 int8 动态量化快 2.4~2.9 倍）、
# decoder int8（小 cache 时每步比 f32 快 ~20%，大 cache 两者都是 cache I/O 主导）
MIXED = ("f32", "int8")


class AedOnnx:
    def __init__(self, model_dir, threads=18, beam_size=3, nbest=1, decode_max_len=0,
                 smoothing=1.25, length_penalty=0.6, eos_penalty=1.0,
                 mode="greedy", mem_arena=True, graph="int8", scale=32768.0,
                 snip_edges=True, use_cmvn=True, intra=None, cache_len="auto",
                 token_per_sec=8.0, model_bytes_direct=False,
                 enc_providers=None, dec_providers=None):
        """enc_providers / dec_providers：执行提供者（EP）名单，None = 只用 CPU。

        给 CUDA 时必须是 ["CUDAExecutionProvider", "CPUExecutionProvider"] 这种
        带 CPU 兜底的写法：单步解码图里混着 ScatterND / 常量索引这类 ORT 不会放到
        CUDA 上的算子，不留兜底就直接在建会话时抛异常（10-09 读 ORT 文档，未实测）。
        建完会话后要看 self.enc_ep / self.dec_ep —— 那是 get_providers() 的实际结果，
        请求了 CUDA 但机器上装的是 CPU 版 onnxruntime 时 ORT 会静默降级，只有这里能看出来。
        """
        import onnxruntime as ort
        self.model_dir = model_dir
        self.graph = graph
        enc_path, dec_path = self._paths(model_dir, graph)
        enc_ep = list(enc_providers) if enc_providers else ["CPUExecutionProvider"]
        dec_ep = list(dec_providers) if dec_providers else ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(intra or threads)
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = bool(mem_arena)
        if model_bytes_direct:
            # 让权重直接从 .onnx 文件 mmap 读，不整份拷进私有内存；多进程并行时省 RAM
            so.add_session_config_entry("session.use_ort_model_bytes_directly", "1")
            so.add_session_config_entry("session.use_ort_model_bytes_for_initializers", "1")
        self.enc = ort.InferenceSession(enc_path, sess_options=so, providers=enc_ep)
        self.dec = ort.InferenceSession(dec_path, sess_options=so, providers=dec_ep)
        self.enc_ep = self.enc.get_providers()
        self.dec_ep = self.dec.get_providers()

        # 元数据直接从已经建好的编码器会话取。以前这里另开一个 InferenceSession 只为
        # 读 custom_metadata_map —— 在 CPU 上等于把 3.1 GB 权重再载入一遍，在 CUDA 上
        # 等于再显存里拷一份，10-09 改成复用 self.enc。
        meta = self.enc.get_modelmeta().custom_metadata_map
        self.mean = np.array([float(x) for x in meta["cmvn_mean"].split(",")], np.float32)
        self.istd = np.array([float(x) for x in meta["cmvn_inv_stddev"].split(",")], np.float32)
        self.sos_id, self.eos_id = int(meta["sos"]), int(meta["eos"])
        self.feat_dim = int(meta["feat_dim"])
        self.n_layers = int(meta["num_decoder_layers"])
        self.n_head = int(meta["num_head"])
        self.head_dim = int(meta["head_dim"])
        self.max_len = int(meta["max_len"])
        self.meta = meta
        self.weight_mb = round(sum(
            os.path.getsize(x) + (os.path.getsize(x + ".data")
                                  if os.path.isfile(x + ".data") else 0)
            for x in (enc_path, dec_path)) / 2**20, 1)

        self.e_in = [i.name for i in self.enc.get_inputs()]
        self.vocab = int(self.dec.get_outputs()[0].shape[-1]) if isinstance(
            self.dec.get_outputs()[0].shape[-1], int) else 8667
        self.id2w = self._load_tokens(os.path.join(model_dir, "tokens.txt"))

        self.mode = mode
        self.beam_size, self.nbest = int(beam_size), int(nbest)
        self.decode_max_len = int(decode_max_len)
        self.smoothing, self.length_penalty, self.eos_penalty = smoothing, length_penalty, eos_penalty
        self.scale, self.snip_edges, self.use_cmvn = scale, snip_edges, use_cmvn
        self.cache_len, self.token_per_sec = cache_len, token_per_sec
        self.times = {"feat": 0.0, "enc": 0.0, "dec": 0.0}
        self.steps = 0
        self.truncated = False

    # ---------- 装载 ----------
    @staticmethod
    def _paths(model_dir, graph):
        if graph == "mixed":
            suffixes = MIXED
        else:
            suffixes = (graph, graph)
        p = [os.path.join(model_dir, n + "." + SUFFIX[g])
             for n, g in zip(("encoder", "decoder"), suffixes)]
        if not all(os.path.isfile(x) for x in p):
            raise FileNotFoundError("缺图：" + " / ".join(x for x in p if not os.path.isfile(x)))
        return p

    @staticmethod
    def _load_tokens(path):
        id2w = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                t = line.rstrip("\n").split()
                if len(t) >= 2:
                    id2w[int(t[1])] = t[0]
        return id2w

    # ---------- 前端 ----------
    def fbank(self, wav, sr):
        import kaldi_native_fbank as knf
        t0 = time.perf_counter()
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0.0
        opts.frame_opts.snip_edges = self.snip_edges
        opts.mel_opts.num_bins = self.feat_dim
        opts.mel_opts.debug_mel = False
        f = knf.OnlineFbank(opts)
        f.accept_waveform(int(sr), (np.asarray(wav, dtype=np.float32) * self.scale).tolist())
        n = f.num_frames_ready
        out = np.vstack([f.get_frame(i) for i in range(n)]) if n else \
            np.zeros((0, self.feat_dim), np.float32)
        if self.use_cmvn:
            out = (out - self.mean) * self.istd
        self.times["feat"] += time.perf_counter() - t0
        return np.ascontiguousarray(out, dtype=np.float32)

    # ---------- 两张图 ----------
    def encode(self, feats):
        t0 = time.perf_counter()
        out = self.enc.run(None, {self.e_in[0]: feats[None],
                                  self.e_in[1]: np.array([feats.shape[0]], np.int64)})
        self.times["enc"] += time.perf_counter() - t0
        return out[0], out[1]                     # [16,1,T',1280] × 2

    def _dec(self, token, sk, sv, ck, cv, offset):
        t0 = time.perf_counter()
        lg, osk, osv = self.dec.run(None, {
            "tokens": np.array([[int(token)]], np.int64),
            "in_n_layer_self_k_cache": sk, "in_n_layer_self_v_cache": sv,
            "n_layer_cross_k": ck, "n_layer_cross_v": cv,
            "offset": np.array([offset], np.int64)})
        self.times["dec"] += time.perf_counter() - t0
        self.steps += 1
        return lg[0, 0], osk, osv

    def _new_cache(self, length):
        z = np.zeros((self.n_layers, 1, length, self.n_head, self.head_dim), np.float32)
        return z, z.copy()

    def _pick_cache_len(self, n_mel_frames, alloc_len=0):
        if isinstance(self.cache_len, int):
            L = self.cache_len
        elif self.cache_len == "max":
            L = self.max_len
        elif alloc_len > 0:
            L = alloc_len
        else:
            L = int(n_mel_frames / 100.0 * self.token_per_sec) + 4
        return max(1, min(int(L), self.max_len))

    # ---------- 贪心（与 sherpa 的 greedy-search-decoder 逐字对齐） ----------
    def greedy(self, ck, cv, n_mel_frames):
        L = self._pick_cache_len(n_mel_frames)
        sk, sv = self._new_cache(L)
        tok = self.sos_id
        ids = []
        for t in range(L - 1):
            lg, sk, sv = self._dec(tok, sk, sv, ck, cv, t)
            tok = int(lg.argmax())
            if tok == self.eos_id:
                break
            ids.append(tok)
        else:
            self.truncated = True
        return [ids]

    # ---------- 官方 beam ----------
    @staticmethod
    def _topk(x, k):
        idx = np.argsort(-x, axis=-1, kind="stable")[..., :k]
        return np.take_along_axis(x, idx, axis=-1), idx

    @staticmethod
    def _log_softmax(x):
        m = x.max(axis=-1, keepdims=True)
        return (x - m) - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))

    def beam(self, ck, cv, n_mel_frames):
        B = self.beam_size
        Ti = ck.shape[2]
        maxlen = self.decode_max_len if self.decode_max_len > 0 else Ti
        L = self._pick_cache_len(n_mel_frames)

        ys = np.full((B, 1), self.sos_id, np.int64)
        conf = np.zeros((B, 1), np.float32)
        scores = np.array([0.0] + [-INF] * (B - 1), np.float32).reshape(B, 1)
        fin = np.zeros((B, 1), np.float32)
        cache = [self._new_cache(L) for _ in range(B)]

        for t in range(maxlen):
            if t >= L:
                # 缓存按需翻倍：sherpa 用 8 token/s 估长度省每步 I/O，估计值只对贪心成立，
                # beam 可能更长；写满就长一截，上限是图自身的 max_len。
                nL = min(self.max_len, max(L * 2, t + 64))
                if nL <= L:
                    self.truncated = True
                    break
                for b in range(B):
                    sk, sv = cache[b]
                    nsk, nsv = self._new_cache(nL)
                    nsk[:, :, :t], nsv[:, :, :t] = sk[:, :, :t], sv[:, :, :t]
                    cache[b] = (nsk, nsv)
                L = nL
            lgs = np.empty((B, self.vocab), np.float32)
            for b in range(B):
                lg, sk, sv = self._dec(ys[b, 0], cache[b][0], cache[b][1], ck, cv, t)
                lgs[b] = lg
                cache[b] = (sk, sv)

            ts = self._log_softmax(lgs / self.smoothing)
            if self.eos_penalty != 1.0:
                ts[:, self.eos_id] *= self.eos_penalty

            tb_s, tb_y = self._topk(ts, B)
            tb_s = tb_s * (1 - fin) + np.array([0.0] + [-INF] * (B - 1), np.float32) * fin
            tb_y = np.where(fin.astype(bool), self.eos_id, tb_y)

            scores = scores + tb_s
            scores, ids = self._topk(scores.reshape(1, B * B), B)      # N=1
            scores = scores.reshape(B, 1)
            row = (ids // B).reshape(B).astype(np.int64)               # N=1 时 stride=0

            t_ys = np.take_along_axis(tb_y.reshape(1, B * B), ids, 1).reshape(B, 1)
            ys = np.concatenate([ys[row], t_ys], axis=1)
            t_cf = np.take_along_axis(tb_s.reshape(1, B * B), ids, 1).reshape(B, 1)
            conf = np.concatenate([conf[row], np.exp(t_cf)], axis=1)
            cache = [cache[int(r)] for r in row]

            fin = (t_ys == self.eos_id).astype(np.float32)
            if fin.sum() == B:
                break

        yl = np.sum(ys != self.eos_id, axis=-1).astype(np.int32)
        out = scores.reshape(1, B)
        if self.length_penalty > 0:
            out = out / np.power((5 + yl.astype(np.float32)) / 6.0, self.length_penalty)
        ns, ni = self._topk(out, self.nbest)
        h = []
        for j in range(self.nbest):
            b = int(ni[0, j])
            Lb = int(yl[b])
            h.append({"yseq": ys[b, 1:Lb], "confidence": float(conf[b, 1:Lb].mean()),
                      "score": float(-ns[0, j]), "steps": ys.shape[1] - 1})
        return h

    # ---------- 文本 ----------
    def detokenize(self, ids, join=""):
        s = join.join(self.id2w[int(i)] for i in ids)
        return re.sub(r"(<blank>)|(<sil>)", "", s).replace("▁", " ").strip().lower()

    # ---------- 一条音频 ----------
    def transcribe_wav(self, wav, sr):
        self.truncated = False
        f = self.fbank(wav, sr)
        if f.shape[0] < 1:
            return {"text": "", "ids": [], "confidence": 0.0, "frames": 0,
                    "fbank_frames": 0, "decode_steps": 0}
        ck, cv = self.encode(f)
        if self.mode == "greedy":
            ids = self.greedy(ck, cv, f.shape[0])[0]
            hyp = {"yseq": ids, "steps": len(ids)}
        else:
            hyp = self.beam(ck, cv, f.shape[0])[0]
        return {"text": self.detokenize(hyp["yseq"]), "ids": [int(i) for i in hyp["yseq"]],
                "confidence": round(hyp.get("confidence", 0.0), 3),
                "score": hyp.get("score"), "frames": int(ck.shape[2]),
                "fbank_frames": int(f.shape[0]),
                "decode_steps": hyp.get("steps", len(hyp["yseq"])),
                "truncated": bool(self.truncated)}
