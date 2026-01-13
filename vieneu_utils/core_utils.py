import re
import os
from typing import List
import numpy as np
import warnings

def split_text_into_chunks(text: str, max_chars: int = 256) -> List[str]:
    """
    Split raw text into chunks no longer than max_chars.
    """
    # 1. First split by newlines - each line/paragraph is handled independently
    paragraphs = re.split(r"[\r\n]+", text.strip())
    final_chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # 2. Split current paragraph into sentences
        sentences = re.split(r"(?<=[\.\!\?\…])\s+", para)
        
        buffer = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If sentence itself is longer than max_chars, we must split it by minor punctuation or words
            if len(sentence) > max_chars:
                # Flush buffer before handling a giant sentence
                if buffer:
                    final_chunks.append(buffer)
                    buffer = ""
                
                # Split giant sentence by minor punctuation (, ; : -)
                sub_parts = re.split(r"(?<=[\,\;\:\-\–\—])\s+", sentence)
                for part in sub_parts:
                    part = part.strip()
                    if not part: continue
                    
                    if len(buffer) + 1 + len(part) <= max_chars:
                        buffer = (buffer + " " + part) if buffer else part
                    else:
                        if buffer: final_chunks.append(buffer)
                        buffer = part
                        
                        # If even a sub-part is too long, split by spaces (words)
                        if len(buffer) > max_chars:
                            words = buffer.split()
                            current = ""
                            for word in words:
                                if current and len(current) + 1 + len(word) > max_chars:
                                    final_chunks.append(current)
                                    current = word
                                else:
                                    current = (current + " " + word) if current else word
                            buffer = current
            else:
                # Normal sentence: check if it fits in current buffer
                if buffer and len(buffer) + 1 + len(sentence) > max_chars:
                    final_chunks.append(buffer)
                    buffer = sentence
                else:
                    buffer = (buffer + " " + sentence) if buffer else sentence
        
        # End of paragraph: flush whatever is in buffer
        if buffer:
            final_chunks.append(buffer)
            buffer = ""

    return [c.strip() for c in final_chunks if c.strip()]


def join_audio_chunks(chunks: list[np.ndarray], sr: int, silence_p: float = 0.0, crossfade_p: float = 0.0, max_crossfade_ratio: float = 0.5) -> np.ndarray:
    """Join audio chunks with optional silence padding and crossfading.
    
    Args:
        chunks: List of audio arrays to join
        sr: Sample rate
        silence_p: Seconds of silence to pad between chunks
        crossfade_p: Crossfade duration in seconds (applied after silence padding)
        max_crossfade_ratio: Maximum crossfade as ratio of chunk length (0.5 = 50%)
    
    Note:
        - Use `silence_p` for clear gaps between chunks (e.g., pauses between sentences)
        - Use `crossfade_p` for smooth transitions without gaps
        - If both are set, silence is added first, then crossfade blends audio edges 
          with silence (may produce unexpected fade-in/fade-out effects)
    
    Returns:
        Concatenated audio array with applied silence padding and/or crossfading
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    
    # Warn about potentially conflicting parameters
    if silence_p > 0 and crossfade_p > 0:
        warnings.warn(
            "Using both silence_p and crossfade_p simultaneously. "
            "Crossfade will blend audio edges with silence, creating fade-in/fade-out effects. "
            "For clearer results, use only one: silence_p for gaps or crossfade_p for smooth transitions.",
            UserWarning,
            stacklevel=2
        )
    
    # 1. Interleave silence if requested
    if silence_p > 0:
        silence_samples = int(sr * silence_p)
        silence_chunk = np.zeros(silence_samples, dtype=np.float32)
        expanded_chunks = []
        for i, chunk in enumerate(chunks):
            expanded_chunks.append(chunk)
            if i < len(chunks) - 1:  # Don't add silence after last chunk
                expanded_chunks.append(silence_chunk)
        chunks = expanded_chunks

    if len(chunks) == 1:
        return chunks[0]
    
    # 2. Apply crossfade between chunks (including silence if present)
    crossfade_samples = int(sr * crossfade_p)
    final_wav = chunks[0].copy()
    
    for next_chunk in chunks[1:]:
        if crossfade_samples > 0:
            # Limit crossfade to avoid consuming entire chunks
            max_overlap_prev = int(len(final_wav) * max_crossfade_ratio)
            max_overlap_next = int(len(next_chunk) * max_crossfade_ratio)
            overlap = min(max_overlap_prev, max_overlap_next, crossfade_samples)
            
            if overlap > 10:  # Minimum samples for meaningful crossfade
                # Equal-power crossfade for smoother transitions
                fade_out = np.sqrt(np.linspace(1.0, 0.0, overlap, dtype=np.float32))
                fade_in = np.sqrt(np.linspace(0.0, 1.0, overlap, dtype=np.float32))
                
                blended = (final_wav[-overlap:] * fade_out + 
                          next_chunk[:overlap] * fade_in)
                
                final_wav = np.concatenate([
                    final_wav[:-overlap],
                    blended,
                    next_chunk[overlap:]
                ])
            else:
                # Chunks too short for crossfade, simple concat
                final_wav = np.concatenate([final_wav, next_chunk])
        else:
            # No crossfade, simple concatenation
            final_wav = np.concatenate([final_wav, next_chunk])
    
    return final_wav


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")