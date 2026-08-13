import numpy as np
import pytest
from vieneu.utils import _linear_overlap_add, extract_speech_ids
from vieneu_utils.core_utils import (
    split_text_into_chunks,
    split_into_sentences,
    pack_sentences_into_chunks,
    join_audio_chunks,
)

# --- Text Utils Tests ---

def test_split_text_into_chunks():
    text = "Đây là một câu ngắn. Đây là một câu dài hơn một chút để kiểm tra xem nó có bị chia ra không nếu chúng ta đặt giới hạn ký tự thấp."
    chunks = split_text_into_chunks(text, max_chars=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 50

def test_split_text_paragraphs():
    text = "Đoạn 1.\n\nĐoạn 2."
    chunks = split_text_into_chunks(text, max_chars=100)
    assert len(chunks) == 2
    assert "Đoạn 1" in chunks[0]
    assert "Đoạn 2" in chunks[1]

# --- Sentence splitting (quote-aware) ---

def test_split_into_sentences_basic():
    assert split_into_sentences("Câu một. Câu hai! Câu ba?") == [
        "Câu một.", "Câu hai!", "Câu ba?",
    ]

def test_split_into_sentences_keeps_quoted_question_together():
    """Dấu .!? trong trích dẫn KHÔNG phải ranh giới câu."""
    text = 'Có phải ý anh là, kiểu như: "Rồi sao nữa? Đến bao giờ?", đúng không anh?'
    assert split_into_sentences(text) == [text]

def test_split_into_sentences_brackets():
    assert split_into_sentences("Cái đó (tôi nghĩ vậy. chắc thế) là đúng. Xong.") == [
        "Cái đó (tôi nghĩ vậy. chắc thế) là đúng.", "Xong.",
    ]

def test_split_into_sentences_unbalanced_quote_falls_back():
    """Một dấu nháy lạc không được nuốt phần còn lại thành một câu khổng lồ."""
    assert split_into_sentences('Câu một. Câu hai lệch " ở đây. Câu ba.') == [
        "Câu một.", 'Câu hai lệch " ở đây.', "Câu ba.",
    ]

def test_split_into_sentences_apostrophe_is_not_a_quote():
    assert split_into_sentences("He said don't worry. Then he left.") == [
        "He said don't worry.", "Then he left.",
    ]

def test_split_into_sentences_decimal_not_split():
    """`.` không theo sau bởi khoảng trắng thì không phải ranh giới câu."""
    assert split_into_sentences("Giá 3.5 triệu. Lúc 8.30 sáng.") == [
        "Giá 3.5 triệu.", "Lúc 8.30 sáng.",
    ]

def test_split_into_sentences_repeated_punct():
    assert split_into_sentences("Câu một... Câu hai?! Câu ba.") == [
        "Câu một...", "Câu hai?!", "Câu ba.",
    ]

# --- Packing ---

def test_pack_sentences_respects_max_chars():
    sents = ["A" * 30 + ".", "B" * 30 + ".", "C" * 30 + "."]
    chunks = pack_sentences_into_chunks(sents, max_chars=70)
    assert all(len(c) <= 70 for c in chunks)
    assert "".join(chunks).count("A") == 30

def test_pack_sentences_splits_oversized_sentence_at_commas():
    sent = "phần một, " + "x" * 40 + ", phần hai, " + "y" * 40 + "."
    chunks = pack_sentences_into_chunks([sent], max_chars=60)
    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks)

def test_chunking_cuts_at_sentence_boundary_not_mid_quote():
    """Regression: cắt ở ranh giới câu, không đẻ ra mảnh vụn mở đầu bằng dấu phẩy."""
    text = (
        "Nghe anh chia sẻ mà em thấy như đang nói trúng tim đen của chính mình và "
        "rất nhiều bạn bè xung quanh vậy. Có phải ý anh là cái cảm giác khi mình "
        "đạt được KPI, hoàn thành một mục tiêu to tát, nhận được những lời khen "
        "ngợi từ sếp hay đồng nghiệp, nhưng thay vì thấy hạnh phúc thì trong lòng "
        'lại trống rỗng, kiểu như: "Rồi sao nữa? Mình phải tiếp tục vòng quay này '
        'đến bao giờ?", đúng không anh?'
    )
    assert len(text) > 384
    chunks = split_text_into_chunks(text, max_chars=384)
    assert len(chunks) == 2
    assert chunks[0].endswith("xung quanh vậy.")
    assert chunks[1].startswith("Có phải ý anh")
    assert not any(c.lstrip().startswith(",") for c in chunks)

# --- Audio Utils Tests ---

def test_linear_overlap_add():
    # Create two overlapping frames
    frame_len = 100
    stride = 50
    frame1 = np.ones(frame_len, dtype=np.float32)
    frame2 = np.ones(frame_len, dtype=np.float32)

    frames = [frame1, frame2]
    out = _linear_overlap_add(frames, stride)

    # Total length should be stride * (len(frames) - 1) + frame_len = 50 * 1 + 100 = 150
    assert out.shape == (150,)
    assert np.any(out != 0)
    # With all ones and linear OLA, the result should be close to 1.0 where it overlaps
    assert np.allclose(out[50:100], 1.0)

def test_linear_overlap_add_empty():
    assert _linear_overlap_add([], 50).shape == (0,)

def test_join_audio_chunks_simple():
    chunks = [np.ones(100), np.zeros(100)]
    joined = join_audio_chunks(chunks, sr=16000)
    assert joined.shape == (200,)
    assert np.array_equal(joined[:100], np.ones(100))
    assert np.array_equal(joined[100:], np.zeros(100))

def test_join_audio_chunks_silence():
    chunks = [np.ones(100), np.ones(100)]
    sr = 16000
    silence_p = 0.1 # 0.1s * 16000 = 1600 samples
    joined = join_audio_chunks(chunks, sr=sr, silence_p=silence_p)
    assert joined.shape == (100 + 1600 + 100,)
    assert np.all(joined[100:1700] == 0)

def test_join_audio_chunks_crossfade():
    chunks = [np.ones(1000), np.zeros(1000)]
    sr = 16000
    crossfade_p = 0.01 # 0.01s * 16000 = 160 samples
    joined = join_audio_chunks(chunks, sr=sr, crossfade_p=crossfade_p)
    # Length should be 1000 + 1000 - 160 = 1840
    assert joined.shape == (1840,)
    assert joined[0] == 1.0
    assert joined[-1] == 0.0
    # Mid-point of crossfade should be 0.5
    assert np.allclose(joined[1000 - 80], 0.5, atol=0.01)

def test_join_audio_chunks_empty():
    assert join_audio_chunks([], 16000).shape == (0,)

def test_join_audio_chunks_single():
    chunk = np.ones(100)
    assert np.array_equal(join_audio_chunks([chunk], 16000), chunk)

def test_extract_speech_ids():
    codes_str = "<|speech_100|><|speech_101|><|speech_102|>"
    assert extract_speech_ids(codes_str) == [100, 101, 102]
    assert extract_speech_ids("no speech here") == []
    assert extract_speech_ids("<|speech_abc|>") == []
