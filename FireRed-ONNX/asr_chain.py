#!/usr/bin/env python3
"""FireRedASR2S ONNX 全链路驱动：FireRedVAD(ONNX) -> [可选 FireRedLID(自导 ONNX, 帧级)] -> FireRedASR2-AED(纯 onnxruntime, aed_ort.py) -> FireRedPunc(ONNX 官方口径：逐段跑)
LID 这一环用的是自己导出的 FireRedLID（lid_encoder.onnx + lid_decoder_step.onnx，纯 onnxruntime，不依赖 sherpa-onnx 和 torch）。默认不跑：ASR 不吃语言标记，跑了只作事后标注，要就加 --lid。
ASR 这一环也已脱 sherpa：sherpa_onnx 已从链路摘掉（它的 AED 只有贪心解码，纯 ORT 复刻后与它逐字相同）。
所有目录都能从外部指定：命令行参数 > 环境变量 FIREDASR_* > 默认（默认相对本文件所在目录，
所以整包放哪儿都能跑）。单独用这条链（不是批处理）时：

用法:
  python asr_chain.py 输入.wav [--models DIR] [--asr-dir DIR] [--vad-dir DIR] [--punc-dir DIR]
                     [--punc-file FILE] [--model aed] [--threads N] [--asr-mode greedy|beam]
                     [--asr-cache auto|max|N] [--asr-graph int8|f32|mixed]
                     [--no-vad] [--lid] [--no-punc] [--json 输出.json]
"""
import argparse, json, os, re, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent


def _env_path(name, default):
    """环境变量 FIREDASR_<name> 优先，没有就用默认（默认全部相对本文件所在目录）。"""
    v = os.environ.get("FIREDASR_" + name)
    return Path(v) if v else Path(default)


# 下面这几个是"默认值"，被 LID_ONNX_DIR 等派生量共用；批量驱动会在 import 后重新赋值，
# 所以两处都要改才一致 —— 单文件用命令行/环境变量，批处理用驱动的参数。
MODELS = _env_path("MODELS", ROOT / "models")
VAD_DIR = _env_path("VAD_DIR", MODELS / "fireredvad-onnx")
PUNC_DIR = _env_path("PUNC_DIR", MODELS / "fireredpunc-onnx")
PUNC_FILE = os.environ.get("FIREDASR_PUNC_FILE", "")
ASR_DIR = os.environ.get("FIREDASR_ASR_DIR", "")   # 空 = 按名字模式在 MODELS 下自动找
LID_ONNX_DIR = _env_path("LID_ONNX_DIR", MODELS / "FireRedLID-onnx")
LID_DICT_DIR = _env_path("LID_DICT_DIR", MODELS / "FireRedLID")
SR = 16000
LATIN = re.compile(r"[a-zA-Z0-9#]+")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "optional-lid"))   # lid_onnx.py 住这儿（LID 非必跑环）


def to_lid_scale(wav):
    """FireRedLID 的前端吃 int16 幅度（官方 kaldiio.load_mat 读出来的那个尺度），
    soundfile 给的是 [-1,1] float32（原整数 /32768，除的是 2 的幂所以可逆），乘回去即可；
    量纲错了会让 log-mel 整体掉到下限，实测同一句话从 zh mandarin(0.9994) 变成 nn(0.3287)。"""
    return (np.clip(wav, -1.0, 1.0) * 32768.0).astype(np.float32)



def find_asr_dir(kind):
    """ASR 模型目录：给了 ASR_DIR 就用它，否则在 MODELS 下按 sherpa 命名模式找。"""
    if ASR_DIR:
        return str(ASR_DIR)
    want_ctc = kind == "ctc"
    cands = [p for p in sorted(MODELS.glob("sherpa-onnx-fire-red-asr2*"))
             if p.is_dir() and ("ctc" in p.name) == want_ctc]
    if not cands:
        raise SystemExit(f"没解包出 {'ctc' if want_ctc else 'aed'} 目录，models 下现有："
                         f"{[q.name for q in MODELS.iterdir()]}")
    return str(cands[0])


def build_recognizer(kind, threads, mode="greedy", cache="auto", graph="int8"):
    """纯 onnxruntime 跑 FireRedASR2-AED（见 aed_ort.py）。

    sherpa_onnx 已从链路里摘掉：它的 C++ AED 只有贪心解码，纯 ORT 复刻前端+解码协议后
    与它逐字相同（test_wavs 4/4，见 probe_grid.json）；beam（官方 batch_beam_search
    协议）实现了但未调通（慢 9 倍、输出退化），仅留档。
    CTC 那套（FireRedASR2-CTC 单模型 / AED 头部的 CTC 对齐）不在这条链里。
    graph: int8=官方 int8 图（默认，对拍背书）；mixed=f32 编码器+int8 解码器（实测最快）；
    f32=逆量化后的全 f32 图（RAM 峰值最高）。
    """
    if kind != "aed":
        raise SystemExit("CTC 不在这条链里，只用 aed")
    base = find_asr_dir("aed")
    import aed_ort
    return aed_ort.AedOnnx(str(base), threads=threads, mode=mode, graph=graph,
                           cache_len=(cache if cache in ("auto", "max") else int(cache)))


def load_audio(path, want_sr=SR):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav[:, 0]
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != want_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, want_sr)
        wav = resample_poly(wav, want_sr // g, sr // g).astype("float32")
        sr = want_sr
    return np.ascontiguousarray(wav), sr


def run_vad(wav, sr):
    """返回 [(start_sec,end_sec), ...]，用 VAD 目录自带的 infer_onnx.py"""
    sys.path.insert(0, str(VAD_DIR))
    from infer_onnx import FireRedVadOnnx
    tmp = Path(os.environ.get("TEMP", "/tmp")) / "_chain_vad.wav"
    sf.write(str(tmp), wav, sr, subtype="PCM_16")
    v = FireRedVadOnnx(model_dir=str(VAD_DIR))
    res, probs = v.detect(str(tmp))
    segs = [(float(a), float(b)) for a, b in res["timestamps"]]
    return segs, float(res["dur"])


def lid_init(backend="firered", threads=18):
    """加载 LID 会话（firered 要读 2.9GB 编码器权重，单独计时）"""
    if backend == "firered":
        import lid_onnx
        return lid_onnx.LidOnnx(str(LID_ONNX_DIR), lid_dir=str(LID_DICT_DIR),
                                ort_threads=threads)
    import onnxruntime as ort
    d = MODELS / "voxlingua107-lid-onnx"
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(d / "voxlingua107.onnx"), sess_options=so,
                                providers=["CPUExecutionProvider"])
    lm = json.loads((d / "lang_map.json").read_text(encoding="utf-8"))
    return sess, sess.get_inputs()[0].name, [lm[str(i)]["iso"] for i in range(len(lm))]


def run_lid(wav, sr, segs, lid, backend="firered", win_s=3.0, hop_s=1.5, mode="slice"):
    """帧精度语种判定。

    backend="firered"：自己导出的 FireRedLID ONNX（lid_encoder.onnx +
    lid_decoder_step.onnx，见 lid_onnx.py），整条编码一次 -> win_s/hop_s 滑窗逐窗解码 ->
    每窗带时间戳 -> 再按 VAD 段的重叠时长×置信度加权汇成段级标签。

    backend="voxlingua" 是顶位方案（VoxLingua107 ECAPA-TDNN ONNX，只有整段级概率），
    FireRedLID 导出没做出来之前用的就是它，留着做对照。

    返回 (每段判定, 整条汇总, 窗序列或 None)。
    """
    if backend == "firered":
        windows = lid.process_windows(to_lid_scale(wav), sr, win_s=win_s,
                                      hop_s=hop_s, mode=mode)
        per = [{"start_s": round(s, 3), "end_s": round(e, 3),
                **lid.segment_vote(windows, s, e)} for s, e in segs]
        whole = lid.segment_vote(windows, 0.0, len(wav) / sr)
        whole["win_dist"] = {}
        for w in windows:
            whole["win_dist"][w["lang"]] = whole["win_dist"].get(w["lang"], 0) + 1
        return per, whole, windows

    sess, iname, iso = lid

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    per, acc = [], []
    for s, e in segs:
        seg = wav[int(s * sr): int(e * sr)]
        if seg.size < int(1.5 * sr):
            per.append({"start_s": round(s, 2), "end_s": round(e, 2),
                        "lang": "太短", "score": 0.0, "windows": 0})
            continue
        y = sess.run(None, {iname: seg[: int(20.0 * sr)].reshape(1, -1)
                            .astype(np.float32)})[0][0]
        p = softmax(y)
        acc.append(p)
        top = int(p.argmax())
        per.append({"start_s": round(s, 2), "end_s": round(e, 2),
                    "lang": iso[top], "score": float(p[top]), "windows": 1})
    mean = np.mean(acc, axis=0) if acc else np.zeros(len(iso), dtype="float32")
    ranked = sorted(range(len(iso)), key=lambda i: -mean[i])[:3]
    whole = {"lang": iso[ranked[0]], "score": float(mean[ranked[0]]), "windows": len(per),
             "top3": [(iso[i], float(mean[i])) for i in ranked]}
    return per, whole, None



class RuleBaedTxtFix:
    """逐字取自官方 fireredpunc/punc.py（RuleBaedTxtFix.fix，原样搬运）。"""

    @classmethod
    def fix(cls, txt_ori, capitalize_first=True):
        txt = txt_ori.lower()
        # English Punc
        txt = re.sub(r"([a-z])，([a-z])", r"\1, \2", txt)
        txt = re.sub(r"([a-z])。([a-z])", r"\1. \2", txt)
        txt = re.sub(r"([a-z])？([a-z])", r"\1? \2", txt)
        txt = re.sub(r"([a-z])！([a-z])", r"\1! \2", txt)
        txt = re.sub(r"^([a-z]+)，", r"\1,", txt)
        txt = re.sub(r"^([a-z]+)。", r"\1.", txt)
        txt = re.sub(r"^([a-z]+)？", r"\1?", txt)
        txt = re.sub(r"^([a-z]+)！", r"\1!", txt)
        txt = re.sub(r"( [a-zA-Z']+)，$", r"\1,", txt)
        txt = re.sub(r"( [a-zA-Z']+)。$", r"\1.", txt)
        txt = re.sub(r"( [a-zA-Z']+)？$", r"\1?", txt)
        txt = re.sub(r"( [a-zA-Z']+)！$", r"\1!", txt)
        # I
        txt = re.sub("^i ", "I ", txt)
        txt = re.sub("^i'm ", "I'm ", txt)
        txt = re.sub("^i'd ", "I'd ", txt)
        txt = re.sub("^i've ", "I've ", txt)
        txt = re.sub("^i'll ", "I'll ", txt)
        txt = re.sub(" i ", " I ", txt)
        txt = re.sub(" i'm ", " I'm ", txt)
        txt = re.sub(" i'd ", " I'd ", txt)
        txt = re.sub(" i've ", " I've ", txt)
        txt = re.sub(" i'll ", " I'll ", txt)
        # First English upper
        if capitalize_first and len(txt) > 0 and re.match("[a-z]", txt[0]):
            txt = txt[0].upper() + txt[1:]
        txt = re.sub(r'([.!?。？！])\s+([a-z])', lambda m: f"{m.group(1)} {m.group(2).upper()}", txt)

        return txt


class Punc:
    """FireRedPunc 的标点环。

    图输入只有 input_ids / attention_mask（都是 int64），喂 [CLS]+子词id，
    输出 (1, T-1, 5)：位置 i 的类别就是第 i 个子词后面该加的标点，和上游
    parity/fireredpunc.py 的用法一致。
    权重默认读 punc.f32.onnx —— 原导出 punc.q8w.onnx 是 weight-only 8-bit
    MatMulNBits，onnxruntime<=1.20 的 contrib 只支持 4-bit，会拒绝加载；
    f32 版由 q8_to_f32.py 逆量化得到，已逐元素核对类别 100% 一致。
    调用口径 = 官方逐段：每个 VAD 段单独进 punc（上游 sentence_max_length=-1、
    不切句），不做 400 字分块——文本超 BERT 512 token 上限时谁都会报错，
    正常 20 s VAD 段到不了。
    """
    def __init__(self, threads=4, model_dir=None, punc_file=None):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        d = Path(model_dir or PUNC_DIR)
        # 三档优先级：显式传参 > 环境变量/驱动全局 PUNC_FILE > 先 f32 后 q8w
        want = str(punc_file or PUNC_FILE or "").strip()
        if want:
            path = Path(want)
            if not path.is_absolute():
                path = d / want
        else:
            path = d / "punc.f32.onnx"
            if not path.exists():
                path = d / "punc.q8w.onnx"
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.model = path.name
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.cls_id = self.tok.token_to_id("[CLS]")
        table = {}
        for line in (d / "out_dict").read_text(encoding="utf-8").splitlines():
            if line.strip():
                w, i = line.split()
                table[int(i)] = " " if w == "<space>" else w
        self.out = [table[i] for i in sorted(table)]
        self.blank = self.out[0]

    def add(self, text):
        """一段文本的完整标点流程（tokenize -> 标签 -> 官方 RuleBaedTxtFix）。"""
        return RuleBaedTxtFix.fix(self._one(text))

    def _one(self, part):
        enc = self.tok.encode(part, add_special_tokens=False)
        ids = [self.cls_id] + list(enc.ids)
        feed = {"input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.ones((1, len(ids)), dtype=np.int64)}
        preds = self.sess.run(None, feed)[0][0].argmax(-1).tolist()
        toks = enc.tokens
        txt = ""
        for i, t in enumerate(toks):
            tag = self.out[preds[i]]
            if t.startswith("##"):
                t = t[2:]
            elif LATIN.search(t) and i > 0 and LATIN.search(toks[i - 1]) \
                    and self.out[preds[i - 1]] == self.blank:
                t = " " + t
            txt += t + ("" if tag == self.blank else tag)
        return txt.replace("  ", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--models", default="", metavar="DIR",
                    help="模型根目录，默认本文件旁的 models（也可用环境变量 FIREDASR_MODELS）")
    ap.add_argument("--asr-dir", default="", metavar="DIR",
                    help="AED 模型目录，默认在 --models 下按 sherpa-onnx-fire-red-asr2* 自动找")
    ap.add_argument("--vad-dir", default="", metavar="DIR",
                    help="FireRedVAD 目录，默认 <models>\\fireredvad-onnx")
    ap.add_argument("--punc-dir", default="", metavar="DIR",
                    help="FireRedPunc 目录，默认 <models>\\fireredpunc-onnx")
    ap.add_argument("--punc-file", default="", metavar="FILE",
                    help="标点模型文件名或路径，默认先 punc.f32.onnx 后 punc.q8w.onnx")
    ap.add_argument("--model", choices=["aed", "ctc"], default="aed")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--asr-mode", choices=["greedy", "beam"], default="greedy",
                    help="greedy=与 sherpa 逐字相同；beam=官方 batch_beam_search(B=3) "
                         "协议（未调通：慢 9 倍、输出退化，仅留档）")
    ap.add_argument("--asr-cache", default="auto",
                    help="KV cache 预分配长度：auto=按 8 token/s 估、max=1024、或给整数")
    ap.add_argument("--asr-graph", choices=["int8", "f32", "mixed"], default="int8",
                    help="int8=官方图（对拍背书）；mixed=f32 编码器+int8 解码器（实测最快）；"
                         "f32=全 f32")
    ap.add_argument("--no-vad", action="store_true")
    ap.add_argument("--lid", action=argparse.BooleanOptionalAction, default=False,
                    help="帧级语种判定（自导 FireRedLID ONNX；默认关，加 --lid 开启）")
    ap.add_argument("--lid-backend", choices=["firered", "voxlingua"], default="firered")
    ap.add_argument("--lid-mode", choices=["slice", "reencode"], default="slice",
                    help="slice=整条编码一次后按帧切片（窗之间有上下文）；reencode=每窗独立编码")
    ap.add_argument("--lid-win", type=float, default=3.0, help="LID 窗长（秒）")
    ap.add_argument("--lid-hop", type=float, default=1.5, help="LID 帧移（秒）")
    ap.add_argument("--no-punc", action="store_true")
    ap.add_argument("--punc-threads", type=int, default=4)
    ap.add_argument("--json", default=None)
    ap.add_argument("--max-secs", type=float, default=0, help="只截前 N 秒，0=全长")
    a = ap.parse_args()

    # 路径参数覆盖模块全局（下面 find_asr_dir / run_vad / Punc 都读这些全局）
    global MODELS, ASR_DIR, VAD_DIR, PUNC_DIR, PUNC_FILE, LID_ONNX_DIR, LID_DICT_DIR
    if a.models:
        MODELS = Path(a.models)
        # --models 必须把派生目录一起搬走。改之前只搬了 MODELS/LID 两个，VAD 和 Punc
        # 还停在导入时算好的"脚本旁 models"，表现是：ASR 换了地方、断句和标点却读旧目录，
        # 旧目录没那份脚本就直接 ModuleNotFoundError: No module named 'infer_onnx'
        # （2026-10-06 校机实测）。环境变量那条路没这个毛病，因为 FIREDASR_MODELS
        # 是在导入时就派生的；批量驱动也不受影响，它是 import 后逐个重新赋值的。
        LID_ONNX_DIR = MODELS / "FireRedLID-onnx"
        LID_DICT_DIR = MODELS / "FireRedLID"
        VAD_DIR = MODELS / "fireredvad-onnx"
        PUNC_DIR = MODELS / "fireredpunc-onnx"
    if a.asr_dir:
        ASR_DIR = a.asr_dir
    if a.vad_dir:
        VAD_DIR = Path(a.vad_dir)
    if a.punc_dir:
        PUNC_DIR = Path(a.punc_dir)
    if a.punc_file:
        PUNC_FILE = a.punc_file

    t = {}
    t0 = time.perf_counter()
    wav, sr = load_audio(a.wav)
    t["load"] = time.perf_counter() - t0
    if a.max_secs:
        wav = wav[: int(a.max_secs * sr)]
    audio_dur = len(wav) / sr
    print(f"[in ] {os.path.basename(a.wav)}  {audio_dur:.1f} s  sr={sr}")

    # --- VAD ---
    if a.no_vad:
        segs = [(0.0, audio_dur)]
        t["vad"] = 0.0
    else:
        t0 = time.perf_counter()
        segs, _ = run_vad(wav, sr)
        t["vad"] = time.perf_counter() - t0
    speech = sum(e - s for s, e in segs)
    print(f"[vad] {len(segs)} 段 / 语音 {speech:.1f} s  用时 {t['vad']:.2f} s")

    # --- LID（帧级）---
    lid_per = lid_top = lid_windows = None
    lid_stage = {}
    if a.lid:
        need = LID_ONNX_DIR if a.lid_backend == "firered" else MODELS / "voxlingua107-lid-onnx"
        if not need.is_dir():
            t["lid"] = t["lid_load"] = 0.0
            print(f"[lid] {need.name} 目录不存在，跳过")
        else:
            t0 = time.perf_counter()
            lid = lid_init(a.lid_backend, a.threads)
            t["lid_load"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            lid_per, lid_top, lid_windows = run_lid(
                wav, sr, segs, lid, backend=a.lid_backend,
                win_s=a.lid_win, hop_s=a.lid_hop, mode=a.lid_mode)
            t["lid"] = time.perf_counter() - t0
            if lid_windows:
                for k, v in lid.times.items():
                    lid_stage[k] = round(v, 3)
                print(f"[lid] 分阶段: " + "  ".join(f"{k}={v:.2f}s"
                                                    for k, v in lid_stage.items()))
            print(f"[lid] {a.lid_backend} 帧级：{len(lid_per)} 段  "
                  + (f"窗序列 {len(lid_windows)} 窗（{a.lid_mode} "
                     f"{a.lid_win:g}s/{a.lid_hop:g}s）" if lid_windows else "")
                  + f"  加载 {t['lid_load']:.2f} s / 推理 {t['lid']:.2f} s")
            for d in lid_per:
                print(f"       [{d['start_s']:7.2f},{d['end_s']:7.2f}] "
                      f"{d['lang']:14s} score={d['score']:.3f} win={d['windows']}")
            print(f"[lid] 整条: {lid_top['lang']} " + "  ".join(
                f"{c}={p:.3f}" for c, p in lid_top.get("top3", [])))
            if "win_dist" in lid_top:
                print(f"[lid] 窗分布: {lid_top['win_dist']}")
    else:
        t["lid"] = 0.0
        t["lid_load"] = 0.0

    # --- ASR ---
    # 官方 FireRedASR2 不吃语言标记（fireredasr2system.py 是先 ASR、LID 只做事后标注），
    # 所以这里的段级 lang 只随段一起进结果，不改解码行为。
    t0 = time.perf_counter()
    rec = build_recognizer(a.model, a.threads, mode=a.asr_mode, cache=a.asr_cache,
                           graph=a.asr_graph)
    t["asr_load"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    streams = []
    texts = []
    n_trunc = 0
    for i, (s, e) in enumerate(segs):
        seg = wav[int(s * sr): int(e * sr)]
        if seg.size < int(0.1 * sr):
            continue
        r = rec.transcribe_wav(seg, sr)
        n_trunc += bool(r["truncated"])
        texts.append(r["text"].strip())
        streams.append({"start_s": round(s, 3), "end_s": round(e, 3),
                        "lang": lid_per[i]["lang"] if lid_per else None,
                        "text": r["text"].strip(), "ids": len(r["ids"]),
                        "confidence": r["confidence"], "truncated": bool(r["truncated"])})
    t["asr"] = time.perf_counter() - t0
    raw = "".join(texts)
    print(f"[asr] {len(streams)} 段解码，用时 {t['asr']:.2f} s"
          + (f"（{n_trunc} 段撞上 1024 上限，建议 VAD 段切短些）" if n_trunc else ""))

    # --- PUNC（官方口径：逐段跑，再拼成整条）---
    if a.no_punc or not raw:
        final = raw
        t["punc_load"] = 0.0
        t["punc"] = 0.0
    else:
        t0 = time.perf_counter()
        p = Punc(threads=a.punc_threads)
        t["punc_load"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        final = "".join(p.add(x) for x in texts)
        t["punc"] = time.perf_counter() - t0
        print(f"[punc] {p.model} 加载 {t['punc_load']:.2f} s，逐段推理 {t['punc']:.2f} s（{len(texts)} 段）")

    total = sum(t.values())
    print("\n==== 计时 ====")
    for k, v in t.items():
        print(f"  {k:9s} {v:8.2f} s")
    print(f"  总计  {total:8.2f} s   音频 {audio_dur:.1f} s   RTF={total/audio_dur:.4f}   x{audio_dur/total:.2f} 实时")
    print("\n==== 文本（前 600 字）====")
    print(final[:600])

    if a.json:
        Path(a.json).write_text(json.dumps({
            "wav": os.path.basename(a.wav), "model": a.model, "threads": a.threads,
            "audio_dur": round(audio_dur, 3), "speech_dur": round(speech, 3),
            "segments": streams, "lid_backend": a.lid_backend if lid_per else None,
            "lid_per_segment": lid_per, "lid_summary": lid_top, "lid_windows": lid_windows,
            "lid_stage_times": lid_stage,
            "timings": {k: round(v, 4) for k, v in t.items()},
            "rtf": round(total / audio_dur, 5), "text_raw": raw, "text_punc": final,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[json] {a.json}")


if __name__ == "__main__":
    main()
