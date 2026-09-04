"""
VieNeu-TTS v3 Nano backend (ONNX, CPU-only, torch-free).
=========================================================
    from vieneu import Vieneu
    tts = Vieneu(mode="v3nano")                       # v3 Turbo stays the default
    wav = tts.infer("Xin chào", voice="Adam")         # preset voice
    tts.save(wav, "out.wav")                          # 24 kHz

v3 Nano is a 48M-parameter flow-matching model for machines where v3 Turbo is too
slow — old laptops, mini PCs, single-board computers. What you trade for that:

  * 24 kHz output (Turbo: 48 kHz).
  * Preset voices only — no voice cloning (the codec encoder needed to enroll a new
    reference clip is not shipped). ``encode_reference`` / ``add_voice`` raise.
  * Lower quality on English and code-switched text (trained on ~4,000 h of
    Vietnamese; English exposure is incidental).
  * Non-autoregressive: no frame-level streaming — ``infer_stream`` yields one
    finished chunk at a time.

Quality/speed knobs: ``steps`` (Euler steps, 16 default; 8 is ~2x faster and still
intelligible — pair it with ``sway=-1``) and ``cfg`` (classifier-free guidance, 3.0
default; 0 halves the compute but hurts intelligibility).
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from .base import BaseVieneuTTS
from vieneu_utils.phonemize_text import (
    phonemize_text_with_emotions,
    normalize_to_chunks_v3,
    normalize_to_chunks_v3_with_gaps,
)
from vieneu_utils.core_utils import join_audio_chunks, gaps_to_silence

logger = logging.getLogger("Vieneu.V3Nano")

_NANO_REPO = "pnnbao-ump/VieNeu-TTS-v3-Nano"
_GRAPH_FILES = ["text_encoder.onnx", "duration_predictor.onnx", "vector_estimator.onnx",
                "codec_decoder.onnx", "config.json", "constants.npz"]
_MAX_CHUNK_SECONDS = 15.0        # training clips were <= 15 s; longer targets drift
_MIN_FRAMES = 2


class _Dev:
    """So callers' ``engine.device.type == "cuda"`` checks work (always CPU here)."""
    type = "cpu"


class OnnxV3NanoEngine:
    """The four ONNX graphs + the sampling loop. Torch-free.

    Public surface mirrors the v3 Turbo engines closely enough for the shared web
    UI code: ``infer(phonemes=..., speaker_emb=..., ref_codes=...)`` where, for
    Nano, ``ref_codes`` is the ``[50, 256]`` float style-token array of the voice.
    """
    SAMPLE_RATE = 24_000

    def __init__(self, repo: str = _NANO_REPO, local_dir: Optional[str] = None,
                 threads: int = 0, hf_token: Optional[str] = None):
        import onnxruntime as ort

        self._lock = threading.RLock()
        self.device = _Dev()
        d = Path(local_dir) if local_dir else self._fetch(repo, hf_token)
        self.cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        c = np.load(d / "constants.npz")
        self.null_spk = c["null_spk"].astype(np.float32)[None]           # [1,192]
        self.null_style = c["null_style"].astype(np.float32)[None]       # [1,S,C]
        self.vocab: Dict[str, int] = self.cfg["vocab"]
        self.bos, self.eos, self.pad = int(self.cfg["bos_id"]), int(self.cfg["eos_id"]), int(self.cfg["pad_id"])
        self.fps = float(self.cfg.get("flow_fps", 24000 / 256 / 6))
        # <|emotion_k|> -> single vocab char (①②③...), exactly as in training
        self.emotion_map: Dict[str, str] = dict(self.cfg.get("emotion_tags", {}))
        self._oov_warned: set = set()

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        intra = int(threads) if threads and threads > 0 else min(max((os.cpu_count() or 8) // 2, 1), 8)
        so.intra_op_num_threads = intra
        self.ort_intra_op_threads = intra
        so.log_severity_level = 3
        prov = ["CPUExecutionProvider"]
        self.s_text = ort.InferenceSession(str(d / "text_encoder.onnx"), so, providers=prov)
        self.s_dur = ort.InferenceSession(str(d / "duration_predictor.onnx"), so, providers=prov)
        self.s_ve = ort.InferenceSession(str(d / "vector_estimator.onnx"), so, providers=prov)
        self.s_dec = ort.InferenceSession(str(d / "codec_decoder.onnx"), so, providers=prov)
        # the unconditional text context is voice-independent: build it once
        null_ids = np.array([[self.bos, self.eos]], dtype=np.int64)
        self._null_ctx = self.s_text.run(None, {"ids": null_ids, "style": self.null_style})[0]
        self._null_mask = null_ids != self.pad

    @staticmethod
    def _fetch(repo: str, token: Optional[str]) -> Path:
        from huggingface_hub import hf_hub_download
        last = None
        for fn in _GRAPH_FILES:
            last = hf_hub_download(repo, fn, repo_type="model", token=token)
        return Path(last).parent

    # ── text ──────────────────────────────────────────────────────────────────
    def encode_phones(self, phones: str) -> np.ndarray:
        """Phone string (may contain ``<|emotion_k|>``) -> ids [1, L]. Unknown characters
        are dropped with a one-time warning instead of failing the whole request."""
        s = phones
        for tag, ch in self.emotion_map.items():
            s = s.replace(tag, ch)
        ids = [self.bos]
        for ch in s:
            i = self.vocab.get(ch)
            if i is None:
                if ch not in self._oov_warned:
                    self._oov_warned.add(ch)
                    logger.warning("v3 Nano: bỏ qua ký tự ngoài vocab %r (U+%04X)", ch, ord(ch))
                continue
            ids.append(i)
        ids.append(self.eos)
        return np.array([ids], dtype=np.int64)

    # ── synthesis ─────────────────────────────────────────────────────────────
    def infer(self, phonemes: str, speaker_emb: np.ndarray, ref_codes: np.ndarray, *,
              steps: int = 16, cfg: float = 3.0, sway: float = 0.0, speed: float = 1.0,
              seed: Optional[int] = None, **_ignored: Any) -> np.ndarray:
        """One chunk of phonemes -> float32 waveform @ 24 kHz.

        ``ref_codes`` is the voice's ``[50, 256]`` style-token array (name kept for
        parity with the Turbo engine so the web UI can call both the same way).
        """
        with self._lock:
            spk = np.asarray(speaker_emb, dtype=np.float32).reshape(1, -1)
            style = np.asarray(ref_codes, dtype=np.float32)
            style = style.reshape(1, *style.shape[-2:])
            ids = self.encode_phones(phonemes)
            mask = ids != self.pad
            ctx = self.s_text.run(None, {"ids": ids, "style": style})[0]
            log_s = float(self.s_dur.run(None, {"ctx": ctx, "ctx_mask": mask, "spk": spk})[0][0])
            secs = min(math.exp(log_s) / max(float(speed), 1e-3), _MAX_CHUNK_SECONDS)
            T = max(_MIN_FRAMES, int(round(secs * self.fps)))
            rng = np.random.default_rng(seed)
            x = rng.standard_normal((1, 144, T)).astype(np.float32)
            n = max(1, int(steps))
            u = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
            tg = u + float(sway) * (np.cos(np.pi / 2 * u) - 1 + u)
            for i in range(n):
                t = np.array([tg[i]], dtype=np.float32)
                v = self.s_ve.run(None, {"x": x, "t": t, "ctx": ctx, "ctx_mask": mask,
                                         "spk": spk, "style": style})[0]
                if cfg > 0:
                    vu = self.s_ve.run(None, {"x": x, "t": t, "ctx": self._null_ctx, "ctx_mask": self._null_mask,
                                              "spk": self.null_spk, "style": self.null_style})[0]
                    v = vu + float(cfg) * (v - vu)
                x = x + np.float32(tg[i + 1] - tg[i]) * v
            wav = self.s_dec.run(None, {"x": x})[0][0, 0]
            return _trim_and_fade(np.clip(wav, -1.0, 1.0).astype(np.float32), self.SAMPLE_RATE)


def _trim_and_fade(wav: np.ndarray, sr: int, thresh_db: float = -45.0, keep_s: float = 0.04,
                   fade_s: float = 0.015) -> np.ndarray:
    """Cut the silence the model generated around the speech (keeping ``keep_s``) and
    fade both ends, so chunk joins never click and pauses come only from the gap list."""
    if wav.size == 0:
        return wav
    win = max(1, int(0.01 * sr))
    n_win = wav.size // win
    if n_win > 0:
        env = np.abs(wav[: n_win * win]).reshape(n_win, win).mean(1)
        above = np.flatnonzero(env > 10 ** (thresh_db / 20))
        if above.size:
            a = max(0, int(above[0]) * win - int(keep_s * sr))
            b = min(wav.size, (int(above[-1]) + 1) * win + int(keep_s * sr))
            wav = wav[a:b]
    n = min(int(fade_s * sr), wav.size // 2)
    if n > 0:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))).astype(np.float32)
        wav = wav.copy()
        wav[:n] *= ramp
        wav[-n:] *= ramp[::-1]
    return wav


class V3NanoVieNeuTTS(BaseVieneuTTS):
    """VieNeu-TTS v3 Nano (ONNX, CPU). Preset voices only."""

    def __init__(
        self,
        backbone_repo: str = _NANO_REPO,
        onnx_dir: Optional[str] = None,
        threads: int = 0,          # ORT intra-op threads; 0 = ~physical cores, cap 8
        steps: int = 16,           # default Euler steps for every call (8 = faster, slightly rougher)
        cfg: float = 3.0,          # default classifier-free guidance
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.sample_rate = OnnxV3NanoEngine.SAMPLE_RATE
        logger.info(f"⏳ Loading VieNeu-TTS v3 Nano (ONNX/CPU) from: {onnx_dir or backbone_repo} ...")
        self.engine = OnnxV3NanoEngine(repo=backbone_repo, local_dir=onnx_dir, threads=threads, hf_token=hf_token)
        self.backend = "onnx"
        self.default_steps = int(steps)
        self.default_cfg = float(cfg)
        self.default_style = "tu_nhien"
        self._preset_voices: dict = {}
        self._default_voice: Optional[str] = None
        self._load_nano_voices()
        logger.info("✅ VieNeu-TTS v3 Nano ready (24 kHz, CPU, %d preset voices)", len(self._preset_voices))

    # ── voices ────────────────────────────────────────────────────────────────
    def _load_nano_voices(self) -> None:
        """Presets from assets/voices_v3_nano.json: 192-d x-vector + [50,256] style tokens.

        The dict also exposes the style array under ``codes`` so code written for
        the v3 Turbo preset layout (``speaker_emb`` + ``codes``) works unchanged.
        """
        path = Path(__file__).parent / "assets" / "voices_v3_nano.json"
        if not path.exists():
            logger.warning("voices_v3_nano.json not found — no preset voices available.")
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, v in data.get("presets", {}).items():
            style = np.asarray(v["style"], dtype=np.float32)
            self._preset_voices[name] = {
                "description": v.get("description", ""),
                "gender": v.get("gender", ""),
                "style": style,
                "codes": style,                              # alias (see docstring)
                "speaker_emb": np.asarray(v["speaker_emb"], dtype=np.float32),
                "podcast": True,
            }
        self._default_voice = data.get("default_voice") or next(iter(self._preset_voices), None)
        logger.info(f"📢 Loaded {len(self._preset_voices)} preset voices (default: {self._default_voice})")

    def list_preset_voices(self) -> List[tuple]:
        return [(f"{n} — {v['description']}" if v["description"] else n, n)
                for n, v in self._preset_voices.items()]

    def get_preset_voice(self, voice_name: Optional[str] = None) -> dict:
        name = voice_name or self._default_voice
        if name not in self._preset_voices:
            raise ValueError(f"Voice '{name}' not found. Available: {list(self._preset_voices)}")
        return self._preset_voices[name]

    _NO_CLONE = ("VieNeu-TTS v3 Nano chỉ hỗ trợ giọng có sẵn (preset), không clone giọng từ audio. "
                 "Cần voice cloning thì dùng v3 Turbo: Vieneu(mode=\"v3turbo\").")

    def encode_reference(self, ref_audio: Union[str, Path], denoise: bool = True):
        raise NotImplementedError(self._NO_CLONE)

    def add_voice(self, name: str, ref_audio: Union[str, Path], **kwargs: Any) -> str:
        raise NotImplementedError(self._NO_CLONE)

    def _resolve_voice(self, voice, ref_audio) -> Tuple[np.ndarray, np.ndarray]:
        if ref_audio is not None:
            raise NotImplementedError(self._NO_CLONE)
        if isinstance(voice, dict):
            preset = voice
        elif isinstance(voice, str):
            preset = self.get_preset_voice(voice)
        else:
            preset = self.get_preset_voice(None)
        style = preset.get("style", preset.get("codes"))
        if style is None or preset.get("speaker_emb") is None:
            raise ValueError("Voice dict must carry 'speaker_emb' and 'style' (v3 Nano preset layout).")
        return preset["speaker_emb"], style

    # ── public API ────────────────────────────────────────────────────────────
    def _chunk_wavs(self, chunks: List[str], spk, style, steps, cfg, sway, speed, seed) -> List[np.ndarray]:
        out = []
        for i, chunk in enumerate(chunks):
            ph = phonemize_text_with_emotions(chunk)
            out.append(self.engine.infer(ph, spk, style, steps=steps, cfg=cfg, sway=sway, speed=speed,
                                         seed=None if seed is None else int(seed) + i))
        return out

    def infer(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: Any = None,          # accepted for API parity with v3 Turbo; ignored
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        sway: float = 0.0,
        speed: float = 1.0,
        seed: Optional[int] = None,
        max_chars: int = 140,       # Nano was trained on <= 15 s clips: keep chunks short
        apply_watermark: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Synthesize ``text`` into one 24 kHz waveform with a preset ``voice``."""
        spk, st = self._resolve_voice(voice, ref_audio)
        chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)
        wavs = self._chunk_wavs(chunks, spk, st, steps or self.default_steps,
                                self.default_cfg if cfg is None else cfg, sway, speed, seed)
        wav = join_audio_chunks(wavs, self.sample_rate, silence_ps=gaps_to_silence(gaps))
        return self._apply_watermark(wav) if apply_watermark else wav

    def infer_stream(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: Any = None,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        sway: float = 0.0,
        speed: float = 1.0,
        max_chars: int = 140,
        apply_watermark: bool = True,
        **kwargs: Any,
    ) -> Generator[np.ndarray, None, None]:
        """Chunk-level streaming: yields each finished chunk (no frame-level streaming —
        the flow model generates a whole chunk at once)."""
        spk, st = self._resolve_voice(voice, ref_audio)
        for chunk in normalize_to_chunks_v3(text, max_chars=max_chars):
            ph = phonemize_text_with_emotions(chunk)
            wav = self.engine.infer(ph, spk, st, steps=steps or self.default_steps,
                                    cfg=self.default_cfg if cfg is None else cfg, sway=sway, speed=speed)
            if wav.size:
                yield self._apply_watermark(wav) if apply_watermark else wav

    def infer_batch(
        self,
        texts: List[str],
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        apply_watermark: bool = True,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """Sequential on CPU (one waveform per text, same order)."""
        return [self.infer(t, ref_audio=ref_audio, voice=voice, apply_watermark=apply_watermark, **kwargs)
                for t in texts]

    def close(self) -> None:
        self.engine = None
