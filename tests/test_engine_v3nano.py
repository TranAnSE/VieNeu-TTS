"""v3 Nano engine tests — torch-free, ONNX sessions mocked (no model download)."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vieneu.v3nano import OnnxV3NanoEngine, V3NanoVieNeuTTS, _trim_and_fade, _MAX_CHUNK_SECONDS

S, C = 4, 8            # tiny style-token shape for the fakes
FPS = 15.625


class _FakeSession:
    """Shape-faithful stand-in for the four ONNX graphs."""
    dur_seconds = 1.2

    def __init__(self, path, *a, **k):
        self.kind = Path(path).stem

    def run(self, _outs, feeds):
        if self.kind == "text_encoder":
            return [np.zeros((1, feeds["ids"].shape[1], 16), np.float32)]
        if self.kind == "duration_predictor":
            return [np.array([np.log(self.dur_seconds)], np.float32)]
        if self.kind == "vector_estimator":
            return [np.ones_like(feeds["x"]) * 0.1]
        if self.kind == "codec_decoder":
            T = feeds["x"].shape[2]
            return [np.random.default_rng(0).standard_normal((1, 1, T * 6 * 256)).astype(np.float32) * 0.3]
        raise AssertionError(self.kind)


class _LongDurSession(_FakeSession):
    dur_seconds = 60.0


@pytest.fixture
def bundle(tmp_path):
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, " ": 3, "a": 4, "b": 5, ".": 6, "①": 7, "②": 8, "③": 9}
    (tmp_path / "config.json").write_text(json.dumps({
        "vocab": vocab, "bos_id": 1, "eos_id": 2, "pad_id": 0, "flow_fps": FPS,
        "emotion_tags": {"<|emotion_1|>": "①", "<|emotion_2|>": "②", "<|emotion_3|>": "③"},
    }), encoding="utf-8")
    np.savez(tmp_path / "constants.npz", null_spk=np.zeros(192, np.float32),
             null_style=np.zeros((S, C), np.float32))
    for n in ["text_encoder", "duration_predictor", "vector_estimator", "codec_decoder"]:
        (tmp_path / f"{n}.onnx").write_bytes(b"")
    return tmp_path


def _engine(bundle, session_cls=_FakeSession):
    with patch("onnxruntime.InferenceSession", session_cls):
        return OnnxV3NanoEngine(local_dir=str(bundle))


def test_encode_phones_maps_emotion_tags_and_drops_oov(bundle):
    eng = _engine(bundle)
    ids = eng.encode_phones("a b <|emotion_1|>. ZZ")
    # bos, a, ' ', b, ' ', ①, '.', ' ', eos — 'Z' is out of vocab: dropped, not fatal
    assert ids.tolist() == [[1, 4, 3, 5, 3, 7, 6, 3, 2]]
    assert ids.dtype == np.int64


def test_engine_infer_shapes(bundle):
    eng = _engine(bundle)
    wav = eng.infer("a b.", np.zeros(192, np.float32), np.zeros((S, C), np.float32), steps=4, cfg=3.0, seed=0)
    assert wav.dtype == np.float32 and wav.ndim == 1 and wav.size > 0
    assert np.abs(wav).max() <= 1.0
    assert wav.size <= round(1.2 * FPS) * 6 * 256          # never longer than the predicted 1.2 s


def test_engine_clamps_duration_to_training_ceiling(bundle):
    eng = _engine(bundle, _LongDurSession)                   # head predicts 60 s
    wav = eng.infer("a b.", np.zeros(192, np.float32), np.zeros((S, C), np.float32), steps=1, cfg=0.0)
    assert wav.size <= round(_MAX_CHUNK_SECONDS * FPS) * 6 * 256


def test_trim_and_fade_removes_silence_and_fades():
    sr = 24000
    wav = np.concatenate([np.zeros(sr), np.ones(sr) * 0.5, np.zeros(sr)]).astype(np.float32)
    out = _trim_and_fade(wav, sr)
    assert sr <= out.size <= sr + int(0.1 * sr)             # 1 s of speech (+ up to 2×40 ms kept)
    assert out[0] == pytest.approx(0.0, abs=1e-6) and out[-1] == pytest.approx(0.0, abs=1e-6)
    assert out[out.size // 2] == pytest.approx(0.5)


_PRESETS = {
    "V1": {"description": "d1", "gender": "male"},
    "V2": {"description": "", "gender": "female"},
}


def _fake_load_voices(self):
    self._preset_voices = {
        n: {"description": v["description"], "gender": v["gender"],
            "style": np.zeros((S, C), np.float32), "codes": np.zeros((S, C), np.float32),
            "speaker_emb": np.zeros(192, np.float32), "podcast": True}
        for n, v in _PRESETS.items()}
    self._default_voice = "V1"


def test_tts_presets_infer_stream_batch_and_clone_guard(bundle):
    with patch("onnxruntime.InferenceSession", _FakeSession), \
         patch.object(V3NanoVieNeuTTS, "_load_nano_voices", _fake_load_voices), \
         patch("vieneu.v3nano.phonemize_text_with_emotions", lambda t: "a b."), \
         patch("vieneu.v3nano.normalize_to_chunks_v3_with_gaps", lambda t, max_chars=140: (["x", "y"], ["sentence"])), \
         patch("vieneu.v3nano.normalize_to_chunks_v3", lambda t, max_chars=140: ["x", "y"]):
        tts = V3NanoVieNeuTTS(onnx_dir=str(bundle))
        assert tts.sample_rate == 24000 and tts.backend == "onnx"
        assert [v for _, v in tts.list_preset_voices()] == ["V1", "V2"]
        assert tts.get_preset_voice()["description"] == "d1"      # default voice

        wav = tts.infer("bất kỳ", voice="V2", apply_watermark=False)
        assert isinstance(wav, np.ndarray) and wav.dtype == np.float32 and wav.size > 0
        assert len(list(tts.infer_stream("bất kỳ", apply_watermark=False))) == 2
        assert len(tts.infer_batch(["a", "b"], apply_watermark=False)) == 2

        # the web UI calls the engine the same way it calls v3 Turbo's
        vd = tts.get_preset_voice("V1")
        out = tts.engine.infer(phonemes="a b.", speaker_emb=vd["speaker_emb"], ref_codes=vd["codes"],
                               use_ref_codes=True, temperature=0.8, max_new_frames=300)
        assert out.size > 0

        with pytest.raises(NotImplementedError):
            tts.encode_reference("clip.wav")
        with pytest.raises(NotImplementedError):
            tts.infer("x", ref_audio="clip.wav")
        with pytest.raises(ValueError):
            tts.infer("x", voice="không có")
