from pathlib import Path
from typing import Generator
import librosa
import numpy as np
import torch
from neucodec import NeuCodec, DistillNeuCodec
from vieneu_utils.phonemize_text import phonemize_with_dict
from vieneu_utils.core_utils import split_text_into_chunks
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import re
import gc
import requests

# ============================================================================
# Shared Utilities
# ============================================================================

def _linear_overlap_add(frames: list[np.ndarray], stride: int) -> np.ndarray:
    """Linear overlap-add for smooth audio concatenation"""
    assert len(frames)
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]

    total_size = 0
    for i, frame in enumerate(frames):
        frame_end = stride * i + frame.shape[-1]
        total_size = max(total_size, frame_end)

    sum_weight = np.zeros(total_size, dtype=dtype)
    out = np.zeros(*shape, total_size, dtype=dtype)

    offset: int = 0
    for frame in frames:
        frame_length = frame.shape[-1]
        t = np.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]
        weight = np.abs(0.5 - (t - 0.5))

        out[..., offset : offset + frame_length] += weight * frame
        sum_weight[offset : offset + frame_length] += weight
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


def _join_audio_chunks(chunks: list[np.ndarray], sr: int, silence_p: float = 0.0, crossfade_p: float = 0.0) -> np.ndarray:
    """Join audio chunks with optional silence padding and crossfading."""
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]
    
    silence_samples = int(sr * silence_p)
    crossfade_samples = int(sr * crossfade_p)
    
    final_wav = chunks[0]
    
    for i in range(1, len(chunks)):
        next_chunk = chunks[i]
        
        if silence_samples > 0:
            # 1. Add silence between chunks
            silence = np.zeros(silence_samples, dtype=np.float32)
            final_wav = np.concatenate([final_wav, silence, next_chunk])
        elif crossfade_samples > 0:
            # 2. Crossfade between chunks
            overlap = min(len(final_wav), len(next_chunk), crossfade_samples)
            if overlap > 0:
                fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                
                blended = (final_wav[-overlap:] * fade_out + next_chunk[:overlap] * fade_in)
                final_wav = np.concatenate([
                    final_wav[:-overlap],
                    blended,
                    next_chunk[overlap:]
                ])
            else:
                final_wav = np.concatenate([final_wav, next_chunk])
        else:
            # 3. Simple concatenation
            final_wav = np.concatenate([final_wav, next_chunk])
            
    return final_wav


def _compile_codec_with_triton(codec):
    """Compile codec with Triton for faster decoding (Windows/Linux compatible)"""
    try:
        import triton
        
        if hasattr(codec, 'dec') and hasattr(codec.dec, 'resblocks'):
            if len(codec.dec.resblocks) > 2:
                codec.dec.resblocks[2].forward = torch.compile(
                    codec.dec.resblocks[2].forward,
                    mode="reduce-overhead",
                    dynamic=True
                )
                print("   ✅ Triton compilation enabled for codec")
        return True
        
    except ImportError:
        # Silently fail for optional triton optimization
        return False
    except Exception:
        return False


# ============================================================================
# VieNeuTTS - Standard implementation (CPU/GPU compatible)
# Supports: PyTorch Transformers, GGUF/GGML quantized models
# ============================================================================

class VieNeuTTS:
    """
    Standard VieNeu-TTS implementation.
    
    Supports:
    - PyTorch + Transformers backend (CPU/GPU)
    - GGUF quantized models via llama-cpp-python (CPU optimized)
    
    Use this for:
    - CPU-only environments
    - Standard PyTorch workflows
    - GGUF quantized models
    """
    
    def __init__(
        self,
        backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
        backbone_device="cpu",
        codec_repo="neuphonic/distill-neucodec",
        codec_device="cpu",
    ):
        """
        Initialize VieNeu-TTS.
        
        Args:
            backbone_repo: Model repository or path to GGUF file
            backbone_device: Device for backbone ('cpu', 'cuda', 'gpu')
            codec_repo: Codec repository
            codec_device: Device for codec
        """

        # Constants
        self.sample_rate = 24_000
        self.max_context = 2048
        self.hop_length = 480
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 25
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        # Flags
        self._is_quantized_model = False
        self._is_onnx_codec = False

        # HF tokenizer
        self.tokenizer = None

        # Load models
        if backbone_repo:
            self._load_backbone(backbone_repo, backbone_device)
        self._load_codec(codec_repo, codec_device)

        # Asset path
        self.assets_dir = Path(__file__).parent / "assets" / "samples"
        self._preset_voices = {
            "Binh": "Bình (nam miền Bắc)",
            "Tuyen": "Tuyên (nam miền Bắc)",
            "Vinh": "Vĩnh (nam miền Nam)",
            "Doan": "Đoan (nữ miền Nam)",
            "Ly": "Ly (nữ miền Bắc)",
            "Ngoc": "Ngọc (nữ miền Bắc)",
        }

        # Load watermarker (optional)
        try:
            import perth
            self.watermarker = perth.PerthImplicitWatermarker()
            print("   🔒 Audio watermarking initialized (Perth)")
        except (ImportError, AttributeError):
            self.watermarker = None
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        """Finalizer to ensure resources are released."""
        try:
            self.close()
        except:
            pass

    def close(self):
        """Explicitly release model resources."""
        # Use a local reference to modules to avoid NoneType errors during shutdown
        _gc = globals().get("gc", None)
        _torch = globals().get("torch", None)
        
        try:
            if hasattr(self, "backbone") and self.backbone is not None:
                # For GGUF models, call close() to avoid shutdown errors
                if getattr(self, "_is_quantized_model", False):
                    try:
                        # Defensive check if backbone still has close method
                        close_fn = getattr(self.backbone, "close", None)
                        if callable(close_fn):
                            close_fn()
                    except:
                        pass
                self.backbone = None
            
            if hasattr(self, "codec") and self.codec is not None:
                self.codec = None
                
            # Final memory cleanup - safely check for locally captured modules
            if _gc is not None:
                _gc.collect()
            
            if _torch is not None:
                if hasattr(_torch, "cuda") and _torch.cuda is not None:
                    if callable(getattr(_torch.cuda, "is_available", None)) and _torch.cuda.is_available():
                        if callable(getattr(_torch.cuda, "empty_cache", None)):
                            _torch.cuda.empty_cache()
        except:
            # Silence all exit errors as we are shutting down anyway
            pass
    
    def _load_backbone(self, backbone_repo, backbone_device):
        # MPS device validation
        if backbone_device == "mps":
            if not torch.backends.mps.is_available():
                print("Warning: MPS not available, falling back to CPU")
                backbone_device = "cpu"

        print(f"Loading backbone from: {backbone_repo} on {backbone_device} ...")

        if backbone_repo.lower().endswith("gguf") or "gguf" in backbone_repo.lower():
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "Failed to import `llama_cpp`. "
                    "Xem hướng dẫn cài đặt llama_cpp_python phiên bản tối thiểu 0.3.16 tại: https://llama-cpp-python.readthedocs.io/en/latest/"
                ) from e
            self.backbone = Llama.from_pretrained(
                repo_id=backbone_repo,
                filename="*.gguf",
                verbose=False,
                n_gpu_layers=-1 if backbone_device == "gpu" else 0,
                n_ctx=self.max_context,
                mlock=True,
                flash_attn=True if backbone_device == "gpu" else False,
            )
            self._is_quantized_model = True
            
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo)
            self.backbone = AutoModelForCausalLM.from_pretrained(backbone_repo).to(
                torch.device(backbone_device)
            )
    
    def _load_codec(self, codec_repo, codec_device):
        # MPS device validation
        if codec_device == "mps":
            if not torch.backends.mps.is_available():
                print("Warning: MPS not available for codec, falling back to CPU")
                codec_device = "cpu"

        print(f"Loading codec from: {codec_repo} on {codec_device} ...")
        match codec_repo:
            case "neuphonic/neucodec":
                self.codec = NeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/distill-neucodec":
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/neucodec-onnx-decoder-int8":
                if codec_device != "cpu":
                    raise ValueError("Onnx decoder only currently runs on CPU.")
                try:
                    from neucodec import NeuCodecOnnxDecoder
                except ImportError as e:
                    raise ImportError(
                        "Failed to import the onnx decoder."
                        "Ensure you have onnxruntime installed as well as neucodec >= 0.0.4."
                    ) from e
                self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
                self._is_onnx_codec = True
            case _:
                raise ValueError(f"Unsupported codec repository: {codec_repo}")
    
    def load_lora_adapter(self, lora_repo_id: str, hf_token: str = None):
        """
        Load LoRA adapter.
        """
        if self._is_quantized_model:
            raise NotImplementedError("LoRA not supported for GGUF quantized models. Use PyTorch backbone.")
        
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError("PEFT library required for LoRA. Install with: pip install peft")
        
        print(f"🎯 Loading LoRA adapter from: {lora_repo_id}")
        
        # Save original clean backbone reference if not already saved
        if not hasattr(self, '_lora_loaded') or not self._lora_loaded:
            self._current_lora_repo = None
            self._lora_loaded = False
        
        # If LoRA already loaded, unload it first to start from a clean base
        if self._lora_loaded:
            self.unload_lora_adapter()
        
        try:
            # Load LoRA adapter (we keep it as a PeftModel, NO merging to allow reversal)
            self.backbone = PeftModel.from_pretrained(
                self.backbone,
                lora_repo_id,
                token=hf_token
            )
            self._lora_loaded = True
            self._current_lora_repo = lora_repo_id
            
            print(f"   ✅ LoRA adapter loaded: {lora_repo_id}")
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to load LoRA adapter: {str(e)}") from e
    
    def unload_lora_adapter(self):
        """
        Unload LoRA adapter and restore original backbone weights using PEFT's unload().
        """
        if not hasattr(self, '_lora_loaded') or not self._lora_loaded:
            return False
        
        print(f"   🔄 Unloading LoRA adapter: {self._current_lora_repo}")
        
        try:
            # PEFT's unload() removes the lora layers and returns the clean base model
            self.backbone = self.backbone.unload()
            self._lora_loaded = False
            self._current_lora_repo = None
            
            # Cleanup memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("   ✅ LoRA adapter unloaded, original weights restored")
            return True
        except Exception as e:
            print(f"   ⚠️ Error during unload: {e}")
            return False


    def list_preset_voices(self):
        """List available preset voices included in the package."""
        return list(self._preset_voices.keys())

    def get_preset_voice(self, voice_name: str):
        """
        Get reference codes and text for a preset voice.
        
        Returns:
            dict: { 'codes': torch.Tensor, 'text': str }
        """
        if voice_name not in self._preset_voices:
            raise ValueError(f"Voice '{voice_name}' not found. Available: {self.list_preset_voices()}")
        
        base_name = self._preset_voices[voice_name]
        audio_path = self.assets_dir / f"{base_name}.wav"
        text_path = self.assets_dir / f"{base_name}.txt"
        
        # Prefer pre-encoded codes if they exist (faster)
        pt_path = self.assets_dir / f"{base_name}.pt"
        if pt_path.exists():
            ref_codes = torch.load(pt_path, map_location="cpu", weights_only=True)
        else:
            ref_codes = self.encode_reference(audio_path)
            
        with open(text_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
            
        return {"codes": ref_codes, "text": ref_text}

    def clone_voice(self, audio_path: str | Path, text: str, name: str = None):
        """
        Create a new custom voice from reference audio.
        
        Args:
            audio_path: Path to the reference audio file
            text: The exact transcript of the reference audio
            name: Optional name for saving this voice permanently.
            
        Returns:
            dict: { 'codes': torch.Tensor, 'text': str }
        """
        ref_codes = self.encode_reference(audio_path)
        voice = {"codes": ref_codes, "text": text}
        
        if name:
            self.save_voice(name, voice)
            
        return voice

    def save_voice(self, name: str, voice: dict):
        """Save a voice to the local assets directory for future use."""
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
        base_path = self.assets_dir / safe_name
        
        try:
            # Save codes
            torch.save(voice['codes'], base_path.with_suffix('.pt'))
            
            # Save text
            with open(base_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                f.write(voice['text'])
                
            print(f"✅ Voice '{name}' saved to {self.assets_dir}")
            
            # Update internal cache if applicable
            self._preset_voices[name] = safe_name
            
        except Exception as e:
            print(f"⚠️ Failed to save voice '{name}': {e}")

    def encode_reference(self, ref_audio_path: str | Path):
        """Encode reference audio to codes"""
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes

    def infer(self, text: str, ref_codes: np.ndarray | torch.Tensor = None, ref_text: str = None, max_chars: int = 256, silence_p: float = 0.0, crossfade_p: float = 0.0, voice: dict = None, temperature: float = 1.0, top_k: int = 50) -> np.ndarray:
        """
        Perform inference to generate speech from text using the TTS model and reference audio.
        Automatically splits long text into chunks.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Reference text for reference audio.
            max_chars (int): Maximum characters per chunk for splitting.
            silence_p (float): Seconds of silence to pad between chunks.
            crossfade_p (float): Seconds of crossfade between chunks (ignored if silence_p > 0).
            voice (dict): Optional dictionary containing 'codes' and 'text' (overrides ref_codes/ref_text).
            temperature (float): Sampling temperature (default 1.0).
            top_k (int): Top-k sampling (default 50).
        Returns:
            np.ndarray: Generated speech waveform.
        """
        if voice is not None:
            ref_codes = voice.get('codes', ref_codes)
            ref_text = voice.get('text', ref_text)
            
        if ref_codes is None or ref_text is None:
             raise ValueError("Must provide either 'voice' dict or both 'ref_codes' and 'ref_text'.")

        # Split text into chunks for better processing of long text
        chunks = split_text_into_chunks(text, max_chars=max_chars)
        
        if not chunks:
            return np.array([], dtype=np.float32)

        all_wavs = []
        for chunk in chunks:
            # Generate tokens
            if self._is_quantized_model:
                output_str = self._infer_ggml(ref_codes, ref_text, chunk, temperature, top_k)
            else:
                prompt_ids = self._apply_chat_template(ref_codes, ref_text, chunk)
                output_str = self._infer_torch(prompt_ids, temperature, top_k)

            # Decode
            wav = self._decode(output_str)
            all_wavs.append(wav)

        # Join all chunks with optional padding/crossfade
        final_wav = _join_audio_chunks(all_wavs, self.sample_rate, silence_p, crossfade_p)

        # Apply watermark if available
        if self.watermarker:
            final_wav = self.watermarker.apply_watermark(final_wav, sample_rate=self.sample_rate)

        return final_wav

    def infer_stream(self, text: str, ref_codes: np.ndarray | torch.Tensor = None, ref_text: str = None, max_chars: int = 256, voice: dict = None, temperature: float = 1.0, top_k: int = 50) -> Generator[np.ndarray, None, None]:
        """
        Perform streaming inference to generate speech from text using the TTS model and reference audio.
        Automatically splits long text into chunks and streams them.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Reference text for reference audio.
            max_chars (int): Maximum characters per chunk for splitting.
            voice (dict): Optional dictionary containing 'codes' and 'text'.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling.
        Yields:
            np.ndarray: Generated speech waveform.
        """
        if voice is not None:
            ref_codes = voice.get('codes', ref_codes)
            ref_text = voice.get('text', ref_text)
            
        if ref_codes is None or ref_text is None:
             raise ValueError("Must provide either 'voice' dict or both 'ref_codes' and 'ref_text'.")

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        
        for chunk in chunks:
            if self._is_quantized_model:
                yield from self._infer_stream_ggml(ref_codes, ref_text, chunk, temperature, top_k)
            else:
                # Fallback for torch backend (no internal streaming, but can stream by chunks)
                prompt_ids = self._apply_chat_template(ref_codes, ref_text, chunk)
                output_str = self._infer_torch(prompt_ids, temperature, top_k)
                wav = self._decode(output_str)
                if self.watermarker:
                    wav = self.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
                yield wav

    def _decode(self, codes: str):
        """Decode speech tokens to audio waveform."""
        # Extract speech token IDs using regex
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes)]
        
        if len(speech_ids) == 0:
            raise ValueError(
                "No valid speech tokens found in the output. "
                "Lỗi này có thể do GPU của bạn không hỗ trợ định dạng bfloat16 (ví dụ: dòng T4, RTX 20-series) "
                "dẫn đến sai số khi tính toán. Bạn hãy thử chuyển sang dùng phiên bản GGUF Q4/Q8 hoặc "
                "bỏ chọn 'LMDeploy' trong Tùy chọn nâng cao."
            )
        
        # Onnx decode
        if self._is_onnx_codec:
            codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
            recon = self.codec.decode_code(codes)
        # Torch decode
        else:
            with torch.no_grad():
                codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(
                    self.codec.device
                )
                recon = self.codec.decode_code(codes).cpu().numpy()
        
        return recon[0, 0, :]
    
    def _apply_chat_template(self, ref_codes: list[int], ref_text: str, input_text: str) -> list[int]:
        input_text = phonemize_with_dict(ref_text) + " " + phonemize_with_dict(input_text)

        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")

        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        chat = """user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"""
        ids = self.tokenizer.encode(chat)

        text_replace_idx = ids.index(text_replace)
        ids = (
            ids[:text_replace_idx]
            + [text_prompt_start]
            + input_ids
            + [text_prompt_end]
            + ids[text_replace_idx + 1 :]  # noqa
        )

        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [speech_gen_start] + list(codes)

        return ids

    def _infer_torch(self, prompt_ids: list[int], temperature: float = 1.0, top_k: int = 50) -> str:
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                prompt_tensor,
                max_length=self.max_context,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                use_cache=True,
                min_new_tokens=50,
            )
        input_length = prompt_tensor.shape[-1]
        output_str = self.tokenizer.decode(
            output_tokens[0, input_length:].cpu().numpy().tolist(), add_special_tokens=False
        )
        return output_str

    def _infer_ggml(self, ref_codes: list[int], ref_text: str, input_text: str, temperature: float = 1.0, top_k: int = 50) -> str:
        ref_text = phonemize_with_dict(ref_text)
        input_text = phonemize_with_dict(input_text)

        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )
        output = self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=temperature,
            top_k=top_k,
            stop=["<|SPEECH_GENERATION_END|>"],
        )
        output_str = output["choices"][0]["text"]
        return output_str

    def _infer_stream_ggml(self, ref_codes: torch.Tensor, ref_text: str, input_text: str, temperature: float = 1.0, top_k: int = 50) -> Generator[np.ndarray, None, None]:
        ref_text = phonemize_with_dict(ref_text)
        input_text = phonemize_with_dict(input_text)

        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

        audio_cache: list[np.ndarray] = []
        token_cache: list[str] = [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_samples: int = 0
        n_decoded_tokens: int = len(ref_codes)

        for item in self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=temperature,
            top_k=top_k,
            stop=["<|SPEECH_GENERATION_END|>"],
            stream=True
        ):
            output_str = item["choices"][0]["text"]
            token_cache.append(output_str)

            if len(token_cache[n_decoded_tokens:]) >= self.streaming_frames_per_chunk + self.streaming_lookforward:

                # decode chunk
                tokens_start = max(
                    n_decoded_tokens
                    - self.streaming_lookback
                    - self.streaming_overlap_frames,
                    0
                )
                tokens_end = (
                    n_decoded_tokens
                    + self.streaming_frames_per_chunk
                    + self.streaming_lookforward
                    + self.streaming_overlap_frames
                )
                sample_start = (
                    n_decoded_tokens - tokens_start
                ) * self.hop_length
                sample_end = (
                    sample_start
                    + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames) * self.hop_length
                )
                curr_codes = token_cache[tokens_start:tokens_end]
                recon = self._decode("".join(curr_codes))
                recon = recon[sample_start:sample_end]
                audio_cache.append(recon)

                # postprocess
                processed_recon = _linear_overlap_add(
                    audio_cache, stride=self.streaming_stride_samples
                )
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                processed_recon = processed_recon[
                    n_decoded_samples:new_samples_end
                ]
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk
                yield processed_recon

        # final decoding handled separately as non-constant chunk size
        remaining_tokens = len(token_cache) - n_decoded_tokens
        if len(token_cache) > n_decoded_tokens:
            tokens_start = max(
                len(token_cache)
                - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens), 
                0
            )
            sample_start = (
                len(token_cache) 
                - tokens_start 
                - remaining_tokens 
                - self.streaming_overlap_frames
            ) * self.hop_length
            curr_codes = token_cache[tokens_start:]
            recon = self._decode("".join(curr_codes))
            recon = recon[sample_start:]
            audio_cache.append(recon)

            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            processed_recon = processed_recon[n_decoded_samples:]
            yield processed_recon


# ============================================================================
# FastVieNeuTTS - GPU-optimized implementation
# Requires: LMDeploy with CUDA
# ============================================================================

class FastVieNeuTTS:
    """
    GPU-optimized VieNeu-TTS using LMDeploy TurbomindEngine.
    """
    
    def __init__(
        self,
        backbone_repo="pnnbao-ump/VieNeu-TTS",
        backbone_device="cuda",
        codec_repo="neuphonic/distill-neucodec",
        codec_device="cuda",
        memory_util=0.3,
        tp=1,
        enable_prefix_caching=True,
        quant_policy=0,
        enable_triton=True,
        max_batch_size=2,
    ):
        """
        Initialize FastVieNeuTTS with LMDeploy backend and optimizations.
        
        Args:
            backbone_repo: Model repository
            backbone_device: Device for backbone (must be CUDA)
            codec_repo: Codec repository
            codec_device: Device for codec
            memory_util: GPU memory utilization (0.0-1.0)
            tp: Tensor parallel size for multi-GPU
            enable_prefix_caching: Enable prefix caching for faster batch processing
            quant_policy: KV cache quantization (0=off, 8=int8, 4=int4)
            enable_triton: Enable Triton compilation for codec
            max_batch_size: Maximum batch size for inference (prevent GPU overload)
        """
        
        if backbone_device != "cuda" and not backbone_device.startswith("cuda:"):
            raise ValueError("LMDeploy backend requires CUDA device")
        
        # Constants
        self.sample_rate = 24_000
        self.max_context = 2048
        self.hop_length = 480
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 50
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length
        
        self.max_batch_size = max_batch_size
        
        self._ref_cache = {}
        
        self.stored_dict = defaultdict(dict)
        
        # Flags
        self._is_onnx_codec = False
        self._triton_enabled = False
        
        # Load models
        self._load_backbone_lmdeploy(backbone_repo, memory_util, tp, enable_prefix_caching, quant_policy)
        self._load_codec(codec_repo, codec_device, enable_triton)

        # Load watermarker (optional)
        try:
            import perth
            self.watermarker = perth.PerthImplicitWatermarker()
            print("   🔒 Audio watermarking initialized (Perth)")
        except (ImportError, AttributeError):
            self.watermarker = None

        self._warmup_model()
        
        print("✅ FastVieNeuTTS with optimizations loaded successfully!")
        print(f"   Max batch size: {self.max_batch_size} (adjustable to prevent GPU overload)")
    
    def _load_backbone_lmdeploy(self, repo, memory_util, tp, enable_prefix_caching, quant_policy):
        """Load backbone using LMDeploy's TurbomindEngine"""
        print(f"Loading backbone with LMDeploy from: {repo}")
        
        try:
            from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
        except ImportError as e:
            raise ImportError(
                "Failed to import `lmdeploy`. Bạn cần cài đặt phiên bản hỗ trợ GPU bằng lệnh: "
                "pip install vieneu[gpu]. \n"
                "Xem thêm hướng dẫn tại: https://github.com/pnnbao97/VieNeu-TTS"
            ) from e
        
        backend_config = TurbomindEngineConfig(
            cache_max_entry_count=memory_util,
            tp=tp,
            enable_prefix_caching=enable_prefix_caching,
            dtype='bfloat16',
            quant_policy=quant_policy
        )
        
        self.backbone = pipeline(repo, backend_config=backend_config)
        
        self.gen_config = GenerationConfig(
            top_p=0.95,
            top_k=50,
            temperature=1.0,
            max_new_tokens=2048,
            do_sample=True,
            min_new_tokens=40,
        )
        
        print(f"   LMDeploy TurbomindEngine initialized")
        print(f"   - Memory util: {memory_util}")
        print(f"   - Tensor Parallel: {tp}")
        print(f"   - Prefix caching: {enable_prefix_caching}")
        print(f"   - KV quant: {quant_policy} ({'Enabled' if quant_policy > 0 else 'Disabled'})")
    
    def _load_codec(self, codec_repo, codec_device, enable_triton):
        """Load codec with optional Triton compilation"""
        print(f"Loading codec from: {codec_repo} on {codec_device}")
        
        match codec_repo:
            case "neuphonic/neucodec":
                self.codec = NeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/distill-neucodec":
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/neucodec-onnx-decoder-int8":
                if codec_device != "cpu":
                    raise ValueError("ONNX decoder only runs on CPU")
                try:
                    from neucodec import NeuCodecOnnxDecoder
                except ImportError as e:
                    raise ImportError(
                        "Failed to import ONNX decoder. "
                        "Ensure onnxruntime and neucodec >= 0.0.4 are installed."
                    ) from e
                self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
                self._is_onnx_codec = True
            case _:
                raise ValueError(f"Unsupported codec repository: {codec_repo}")
        
        if enable_triton and not self._is_onnx_codec and codec_device != "cpu":
            self._triton_enabled = _compile_codec_with_triton(self.codec)
    
    def _warmup_model(self):
        """Warmup inference pipeline to reduce first-token latency"""
        print("🔥 Warming up model...")
        try:
            dummy_codes = list(range(10))
            dummy_prompt = self._format_prompt(dummy_codes, "warmup", "test")
            _ = self.backbone([dummy_prompt], gen_config=self.gen_config, do_preprocess=False)
            print("   ✅ Warmup complete")
        except Exception as e:
            print(f"   ⚠️ Warmup failed (non-critical): {e}")
    
    def encode_reference(self, ref_audio_path: str | Path):
        """Encode reference audio to codes"""
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes
    
    def clone_voice(self, audio_path: str | Path, text: str):
        """
        Create a new custom voice from reference audio.
        
        Args:
            audio_path: Path to the reference audio file
            text: The exact transcript of the reference audio
            
        Returns:
            dict: { 'codes': torch.Tensor, 'text': str }
        """
        ref_codes = self.encode_reference(audio_path)
        return {"codes": ref_codes, "text": text}
    
    def get_cached_reference(self, voice_name: str, audio_path: str, ref_text: str = None):
        """
        Get or create cached reference codes.
        
        Args:
            voice_name: Unique identifier for this voice
            audio_path: Path to reference audio
            ref_text: Optional reference text (stored with codes)
            
        Returns:
            ref_codes: Encoded reference codes
        """
        cache_key = f"{voice_name}_{audio_path}"
        
        if cache_key not in self._ref_cache:
            ref_codes = self.encode_reference(audio_path)
            self._ref_cache[cache_key] = {
                'codes': ref_codes,
                'ref_text': ref_text
            }
        
        return self._ref_cache[cache_key]['codes']
    
    def add_speaker(self, user_id: int, audio_file: str, ref_text: str):
        """
        Add a speaker to the stored dictionary for easy access.
        
        Args:
            user_id: Unique user ID
            audio_file: Reference audio file path
            ref_text: Reference text
            
        Returns:
            user_id: The user ID for use in streaming
        """
        codes = self.encode_reference(audio_file)
        
        if isinstance(codes, torch.Tensor):
            codes = codes.cpu().numpy()
        if isinstance(codes, np.ndarray):
            codes = codes.flatten().tolist()
        
        self.stored_dict[f"{user_id}"]['codes'] = codes
        self.stored_dict[f"{user_id}"]['ref_text'] = ref_text
        
        return user_id
    
    def _decode(self, codes: str):
        """Decode speech tokens to audio waveform"""
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes)]
        
        if len(speech_ids) == 0:
            raise ValueError(
                "No valid speech tokens found in the output. "
                "Lỗi này có thể do GPU của bạn không hỗ trợ định dạng bfloat16 (ví dụ: dòng T4, RTX 20-series) "
                "khiến mô hình chạy không ổn định trên LMDeploy (Turbomind). Bạn hãy thử bỏ chọn 'LMDeploy' "
                "trong Tùy chọn nâng cao hoặc chuyển sang dùng phiên bản GGUF Q4/Q8 để chạy ổn định hơn."
            )
        
        if self._is_onnx_codec:
            codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
            recon = self.codec.decode_code(codes)
        else:
            with torch.no_grad():
                codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(
                    self.codec.device
                )
                recon = self.codec.decode_code(codes).cpu().numpy()
        
        return recon[0, 0, :]
    
    def _decode_batch(self, codes_list: list[str], max_workers: int = None):
        """
        Decode multiple code strings in parallel.
        
        Args:
            codes_list: List of code strings to decode
            max_workers: Number of parallel workers (auto-tuned if None)
            
        Returns:
            List of decoded audio arrays
        """
        # Auto-tune workers based on GPU memory and batch size
        if max_workers is None:
            if torch.cuda.is_available():
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                # 1 worker per 4GB VRAM, max 4 workers
                max_workers = min(max(1, int(gpu_mem_gb / 4)), 4)
            else:
                max_workers = 2
        
        # For small batches, use sequential to avoid overhead
        if len(codes_list) <= 2:
            return [self._decode(codes) for codes in codes_list]
        
        # Parallel decoding with controlled workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._decode, codes) for codes in codes_list]
            results = [f.result() for f in futures]
        return results
    
    def _format_prompt(self, ref_codes: list[int], ref_text: str, input_text: str) -> str:
        """Format prompt for LMDeploy"""
        ref_text_phones = phonemize_with_dict(ref_text)
        input_text_phones = phonemize_with_dict(input_text)
        
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text_phones} {input_text_phones}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )
        
        return prompt
    
    def infer(self, text: str, ref_codes: np.ndarray | torch.Tensor = None, ref_text: str = None, max_chars: int = 256, silence_p: float = 0.0, crossfade_p: float = 0.0, voice: dict = None, temperature: float = 1.0, top_k: int = 50) -> np.ndarray:
        """
        Single inference (automatically splits long text and uses batching for speed).
        
        Args:
            text: Input text to synthesize
            ref_codes: Encoded reference audio codes
            ref_text: Reference text for reference audio
            max_chars: Maximum characters per chunk for splitting.
            voice: Optional dict with 'codes' and 'text'.
            temperature: Sampling temperature.
            top_k: Top-k sampling.
            
        Returns:
            Generated speech waveform as numpy array
        """
        if voice is not None:
            ref_codes = voice.get('codes', ref_codes)
            ref_text = voice.get('text', ref_text)
            
        if ref_codes is None or ref_text is None:
             raise ValueError("Must provide either 'voice' dict or both 'ref_codes' and 'ref_text'.")
        
        # Update generation config if needed
        self.gen_config.temperature = temperature
        self.gen_config.top_k = top_k

        # Split text into chunks
        chunks = split_text_into_chunks(text, max_chars=max_chars)
        
        if not chunks:
            return np.array([], dtype=np.float32)
            
        if len(chunks) == 1:
            # Single chunk optimization
            if isinstance(ref_codes, torch.Tensor):
                ref_codes = ref_codes.cpu().numpy()
            if isinstance(ref_codes, np.ndarray):
                ref_codes = ref_codes.flatten().tolist()
            
            prompt = self._format_prompt(ref_codes, ref_text, chunks[0])
            responses = self.backbone([prompt], gen_config=self.gen_config, do_preprocess=False)
            wav = self._decode(responses[0].text)
        else:
            # Multiple chunks: use batching for parallel generation
            all_wavs = self.infer_batch(chunks, ref_codes, ref_text, voice=voice, temperature=temperature, top_k=top_k)
            wav = _join_audio_chunks(all_wavs, self.sample_rate, silence_p, crossfade_p)

        # Apply watermark if available
        if self.watermarker:
            wav = self.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
            
        return wav
    
    def infer_batch(self, texts: list[str], ref_codes: np.ndarray | torch.Tensor = None, ref_text: str = None, max_batch_size: int = None, voice: dict = None, temperature: float = 1.0, top_k: int = 50) -> list[np.ndarray]:
        """
        Batch inference for multiple texts.
        """
        if max_batch_size is None:
            max_batch_size = self.max_batch_size

        if voice is not None:
            ref_codes = voice.get('codes', ref_codes)
            ref_text = voice.get('text', ref_text)
            
        if ref_codes is None or ref_text is None:
             raise ValueError("Must provide either 'voice' dict or both 'ref_codes' and 'ref_text'.")
            
        if not isinstance(texts, list):
            texts = [texts]
        
        # Update generation config
        self.gen_config.temperature = temperature
        self.gen_config.top_k = top_k
        self.gen_config.repetition_penalty = 1.0 # default
        
        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()
        if isinstance(ref_codes, np.ndarray):
            ref_codes = ref_codes.flatten().tolist()
        
        all_wavs = []
        
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i:i+max_batch_size]
            prompts = [self._format_prompt(ref_codes, ref_text, text) for text in batch_texts]
            responses = self.backbone(prompts, gen_config=self.gen_config, do_preprocess=False)
            batch_codes = [response.text for response in responses]
            
            if len(batch_codes) > 3:
                batch_wavs = self._decode_batch(batch_codes)
            else:
                batch_wavs = [self._decode(codes) for codes in batch_codes]
            
            # Apply watermark if available
            if self.watermarker:
                batch_wavs = [self.watermarker.apply_watermark(w, sample_rate=self.sample_rate) for w in batch_wavs]
                
            all_wavs.extend(batch_wavs)
            
            if i + max_batch_size < len(texts):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return all_wavs
    
    def infer_stream(self, text: str, ref_codes: np.ndarray | torch.Tensor = None, ref_text: str = None, max_chars: int = 256, voice: dict = None, temperature: float = 1.0, top_k: int = 50) -> Generator[np.ndarray, None, None]:
        """
        Streaming inference with low latency (supports long text by splitting into chunks).
        
        Args:
            text: Input text to synthesize
            ref_codes: Encoded reference audio codes
            ref_text: Reference text for reference audio
            max_chars: Maximum characters per chunk for splitting.
            voice: Optional dict with 'codes' and 'text'.
            temperature: Sampling temperature.
            top_k: Top-k sampling.
            
        Yields:
            Audio chunks as numpy arrays
        """
        if voice is not None:
            ref_codes = voice.get('codes', ref_codes)
            ref_text = voice.get('text', ref_text)
            
        if ref_codes is None or ref_text is None:
             raise ValueError("Must provide either 'voice' dict or both 'ref_codes' and 'ref_text'.")
        
        # Update generation config
        self.gen_config.temperature = temperature
        self.gen_config.top_k = top_k
        self.gen_config.repetition_penalty = 1.0

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        
        for chunk in chunks:
            yield from self._infer_stream_single(chunk, ref_codes, ref_text)

    def _infer_stream_single(self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str) -> Generator[np.ndarray, None, None]:
        """Internal method for streaming a single short text chunk"""
        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()
        if isinstance(ref_codes, np.ndarray):
            ref_codes = ref_codes.flatten().tolist()
        
        prompt = self._format_prompt(ref_codes, ref_text, text)
        
        audio_cache = []
        token_cache = [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_samples = 0
        n_decoded_tokens = len(ref_codes)
        
        for response in self.backbone.stream_infer([prompt], gen_config=self.gen_config, do_preprocess=False):
            output_str = response.text
            
            # Extract new tokens
            new_tokens = output_str[len("".join(token_cache[len(ref_codes):])):] if len(token_cache) > len(ref_codes) else output_str
            
            if new_tokens:
                token_cache.append(new_tokens)
            
            # Check if we have enough tokens to decode a chunk
            if len(token_cache[n_decoded_tokens:]) >= self.streaming_frames_per_chunk + self.streaming_lookforward:
                
                # Decode chunk with context
                tokens_start = max(
                    n_decoded_tokens - self.streaming_lookback - self.streaming_overlap_frames,
                    0
                )
                tokens_end = (
                    n_decoded_tokens
                    + self.streaming_frames_per_chunk
                    + self.streaming_lookforward
                    + self.streaming_overlap_frames
                )
                sample_start = (n_decoded_tokens - tokens_start) * self.hop_length
                sample_end = (
                    sample_start
                    + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames) * self.hop_length
                )
                
                curr_codes = token_cache[tokens_start:tokens_end]
                recon = self._decode("".join(curr_codes))
                recon = recon[sample_start:sample_end]
                audio_cache.append(recon)
                
                # Overlap-add processing
                processed_recon = _linear_overlap_add(
                    audio_cache, stride=self.streaming_stride_samples
                )
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                processed_recon = processed_recon[n_decoded_samples:new_samples_end]
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk
                
                yield processed_recon
        
        # Final chunk
        remaining_tokens = len(token_cache) - n_decoded_tokens
        if remaining_tokens > 0:
            tokens_start = max(
                len(token_cache) - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens),
                0
            )
            sample_start = (
                len(token_cache) - tokens_start - remaining_tokens - self.streaming_overlap_frames
            ) * self.hop_length
            
            curr_codes = token_cache[tokens_start:]
            recon = self._decode("".join(curr_codes))
            recon = recon[sample_start:]
            audio_cache.append(recon)
            
            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            processed_recon = processed_recon[n_decoded_samples:]
            yield processed_recon
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Memory cleaned up")
    
    def get_optimization_stats(self) -> dict:
        """
        Get current optimization statistics.
        
        Returns:
            Dictionary with optimization info
        """
        return {
            'triton_enabled': self._triton_enabled,
            'max_batch_size': self.max_batch_size,
            'cached_references': len(self._ref_cache),
            'active_sessions': len(self.stored_dict),
            'kv_quant': self.gen_config.__dict__.get('quant_policy', 0),
            'prefix_caching': True,  # Always enabled in our config
        }


# ============================================================================
# RemoteVieNeuTTS - Instant-load client for remote servers
# ============================================================================

class RemoteVieNeuTTS(VieNeuTTS):
    """
    Client for VieNeu-TTS running on a remote LMDeploy server.
    Extremely fast to initialize as it only loads the local codec.
    
    Use this for:
    - Production/SaaS environments
    - Instant SDK loading in multi-process applications
    - Connecting to a centralized high-performance GPU server
    """
    
    def __init__(
        self, 
        api_base="http://localhost:23333/v1", 
        model_name="pnnbao-ump/VieNeu-TTS",
        codec_repo="neuphonic/distill-neucodec", 
        codec_device="cpu"
    ):
        """
        Initialize Remote Client.
        
        Args:
            api_base: Base URL of LMDeploy api_server
            model_name: Name of the model as registered on the server
            codec_repo: Local codec for decoding
            codec_device: Device for local codec (usually 'cpu' is enough)
        """
        self.api_base = api_base.rstrip('/')
        self.model_name = model_name
        
        # Initialize VieNeuTTS without backbone
        super().__init__(
            backbone_repo=None,
            codec_repo=codec_repo,
            codec_device=codec_device
        )
        
        print(f"📡 RemoteVieNeuTTS ready! Using backend: {self.api_base}")

    def _load_backbone(self, backbone_repo, backbone_device):
        pass # Explicitly skip

    def _format_prompt(self, ref_codes: list[int], ref_text: str, input_text: str) -> str:
        """Format prompt for remote LMDeploy server"""
        ref_text_phones = phonemize_with_dict(ref_text)
        input_text_phones = phonemize_with_dict(input_text)
        
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text_phones} {input_text_phones}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )
        return prompt

    def infer(self, text: str, ref_codes: np.ndarray | torch.Tensor = None, ref_text: str = None, max_chars: int = 256, silence_p: float = 0.0, crossfade_p: float = 0.0, voice: dict = None, temperature: float = 1.0, top_k: int = 50) -> np.ndarray:
        """
        Remote inference (automatically splits long text).
        
        Args:
            text: Input text to synthesize
            ref_codes: Encoded reference audio codes
            ref_text: Reference text for reference audio
            max_chars: Maximum characters per chunk for splitting.
            silence_p (float): Seconds of silence to pad between chunks.
            crossfade_p (float): Seconds of crossfade between chunks (ignored if silence_p > 0).
            voice: Optional dict with 'codes' and 'text'.
            temperature: Sampling temperature.
            top_k: Top-k sampling.
            
        Returns:
            Generated speech waveform as numpy array
        """
        if voice is not None:
            ref_codes = voice.get('codes', ref_codes)
            ref_text = voice.get('text', ref_text)
            
        if ref_codes is None or ref_text is None:
             raise ValueError("Must provide either 'voice' dict or both 'ref_codes' and 'ref_text'.")

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        
        if not chunks:
            return np.array([], dtype=np.float32)

        all_wavs = []
        for chunk in chunks:
            if isinstance(ref_codes, torch.Tensor):
                ref_codes_list = ref_codes.cpu().numpy().flatten().tolist()
            elif isinstance(ref_codes, np.ndarray):
                ref_codes_list = ref_codes.flatten().tolist()
            else:
                ref_codes_list = ref_codes

            prompt = self._format_prompt(ref_codes_list, ref_text, chunk)
            
            # Use chat/completions endpoint as it is standard in lmdeploy serve api_server
            # even if we are sending a pre-formatted prompt string.
            # We wrap the prompt in a user message. LMDeploy might re-template it, but
            # usually if the model doesn't have a strict template or we rely on the model's
            # ability to follow instructions within the user message, this works for now.
            # Ideally we should construct messages properly without pre-formatting if possible,
            # but _format_prompt does heavy lifting (phonemization etc).
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": temperature,
                "top_k": top_k,
                "stop": ["<|SPEECH_GENERATION_END|>"],
                "stream": False
            }
            
            try:
                # Use chat/completions
                response = requests.post(f"{self.api_base}/chat/completions", json=payload, timeout=60)
                response.raise_for_status()
                
                # Parse chat completion response
                output_str = response.json()["choices"][0]["message"]["content"]
                
                # Local decode is extremely fast
                wav = self._decode(output_str)
                all_wavs.append(wav)
            except Exception as e:
                print(f"Error during remote inference: {e}")
                continue

        # Join all chunks with optional padding/crossfade
        final_wav = _join_audio_chunks(all_wavs, self.sample_rate, silence_p, crossfade_p)

        if self.watermarker:
            final_wav = self.watermarker.apply_watermark(final_wav, sample_rate=self.sample_rate)
            
        return final_wav


def Vieneu(mode="standard", **kwargs):
    """
    Factory function for VieNeu-TTS.
    
    Args:
        mode: 'standard' (CPU/GPU-GGUF), 'remote' (API)
        **kwargs: Arguments for chosen class
        
    Returns:
        VieNeuTTS | RemoteVieNeuTTS instance
    """
    match mode:
        case "remote" | "api":
            return RemoteVieNeuTTS(**kwargs)
        case _:
            return VieNeuTTS(**kwargs)
