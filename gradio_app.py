import gradio as gr
print("⏳ Đang khởi động... Vui lòng chờ...")
import soundfile as sf
import tempfile
import torch
from vieneu import VieNeuTTS, FastVieNeuTTS
import os
import sys
import time
import numpy as np
from typing import Generator, Optional, Tuple
import queue
import threading
import yaml
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks, env_bool
from functools import lru_cache
import gc

print("⏳ Đang khởi động VieNeu-TTS...")

# --- CONSTANTS & CONFIG ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}
except Exception as e:
    raise RuntimeError(f"Không thể đọc config.yaml: {e}")

BACKBONE_CONFIGS = _config.get("backbone_configs", {})
CODEC_CONFIGS = _config.get("codec_configs", {})
VOICE_SAMPLES = _config.get("voice_samples", {})

_text_settings = _config.get("text_settings", {})
MAX_CHARS_PER_CHUNK = _text_settings.get("max_chars_per_chunk", 256)
MAX_TOTAL_CHARS_STREAMING = _text_settings.get("max_total_chars_streaming", 3000)

if not BACKBONE_CONFIGS or not CODEC_CONFIGS:
    raise ValueError("config.yaml thiếu backbone_configs hoặc codec_configs")
if not VOICE_SAMPLES:
    raise ValueError("config.yaml thiếu voice_samples")

# --- 1. MODEL CONFIGURATION ---
# Global model instance
tts = None
current_backbone = None
current_codec = None
model_loaded = False
using_lmdeploy = False

# Cache for reference texts
_ref_text_cache = {}

def get_available_devices() -> list[str]:
    """Get list of available devices for current platform."""
    devices = ["Auto", "CPU"]

    if sys.platform == "darwin":
        # macOS - check MPS
        if torch.backends.mps.is_available():
            devices.append("MPS")
    else:
        # Windows/Linux - check CUDA
        if torch.cuda.is_available():
            devices.append("CUDA")

    return devices

def get_model_status_message() -> str:
    """Reconstruct status message from global state"""
    global model_loaded, tts, using_lmdeploy, current_backbone, current_codec
    if not model_loaded or tts is None:
        return "⏳ Chưa tải model."
    
    backbone_config = BACKBONE_CONFIGS.get(current_backbone, {})
    codec_config = CODEC_CONFIGS.get(current_codec, {})
    
    backend_name = "🚀 LMDeploy (Optimized)" if using_lmdeploy else "📦 Standard"
    
    # We don't track the exact device strings perfectly in global state, so we estimate
    device_info = "GPU" if using_lmdeploy else "Auto"
    codec_device = "CPU" if "ONNX" in (current_codec or "") else ("GPU/MPS" if torch.cuda.is_available() or torch.backends.mps.is_available() else "CPU")
    
    preencoded_note = "\n⚠️ Codec ONNX không hỗ trợ chức năng clone giọng nói." if codec_config.get('use_preencoded') else ""
    
    opt_info = ""
    if using_lmdeploy and hasattr(tts, 'get_optimization_stats'):
        stats = tts.get_optimization_stats()
        opt_info = (
            f"\n\n🔧 Tối ưu hóa:"
            f"\n  • Triton: {'✅' if stats['triton_enabled'] else '❌'}"
            f"\n  • Max Batch Size (Default): {stats.get('max_batch_size', 'N/A')}"
            f"\n  • Reference Cache: {stats['cached_references']} voices"
            f"\n  • Prefix Caching: ✅"
        )

    return (
        f"✅ Model đã tải thành công!\n\n"
        f"🔧 Backend: {backend_name}\n"
        f"🦜 Backbone: {current_backbone}\n"
        f"🎵 Codec: {current_codec}{preencoded_note}{opt_info}"
    )

def restore_ui_state():
    """Update UI components based on persistence"""
    global model_loaded
    msg = get_model_status_message()
    return (
        msg, 
        gr.update(interactive=model_loaded), # btn_generate
        gr.update(interactive=False)         # btn_stop
    )

def should_use_lmdeploy(backbone_choice: str, device_choice: str) -> bool:
    """Determine if we should use LMDeploy backend."""
    # LMDeploy not supported on macOS
    if sys.platform == "darwin":
        return False

    if "gguf" in backbone_choice.lower():
        return False

    if device_choice == "Auto":
        has_gpu = torch.cuda.is_available()
    elif device_choice == "CUDA":
        has_gpu = torch.cuda.is_available()
    else:
        has_gpu = False

    return has_gpu

@lru_cache(maxsize=32)
def get_ref_text_cached(text_path: str) -> str:
    """Cache reference text loading"""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()

def cleanup_gpu_memory():
    """Aggressively cleanup GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

def load_model(backbone_choice: str, codec_choice: str, device_choice: str, 
               force_lmdeploy: bool):
    """Load model with optimizations and max batch size control"""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    lmdeploy_error_reason = None
    
    yield (
        "⏳ Đang tải model với tối ưu hóa... Lưu ý: Quá trình này sẽ tốn thời gian. Vui lòng kiên nhẫn.",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False)
    )
    
    try:
        # Cleanup before loading new model
        if model_loaded and tts is not None:
            del tts
            cleanup_gpu_memory()
        
        backbone_config = BACKBONE_CONFIGS[backbone_choice]
        codec_config = CODEC_CONFIGS[codec_choice]
        
        use_lmdeploy = force_lmdeploy and should_use_lmdeploy(backbone_choice, device_choice)
        
        if use_lmdeploy:
            lmdeploy_error_reason = None
            print(f"🚀 Using LMDeploy backend with optimizations")
            
            backbone_device = "cuda"
            
            if "ONNX" in codec_choice:
                codec_device = "cpu"
            else:
                codec_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"📦 Loading optimized model...")
            print(f"   Backbone: {backbone_config['repo']} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")
            print(f"   Triton: Enabled")
            
            try:
                tts = FastVieNeuTTS(
                    backbone_repo=backbone_config["repo"],
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device,
                    memory_util=0.3,
                    tp=1,
                    enable_prefix_caching=True,
                    enable_triton=True,
                )
                using_lmdeploy = True
                
                # Pre-cache voice references
                print("📝 Pre-caching voice references...")
                for voice_name, voice_info in VOICE_SAMPLES.items():
                    audio_path = voice_info["audio"]
                    text_path = voice_info["text"]
                    if os.path.exists(audio_path) and os.path.exists(text_path):
                        ref_text = get_ref_text_cached(text_path)
                        tts.get_cached_reference(voice_name, audio_path, ref_text)
                print(f"   ✅ Cached {len(VOICE_SAMPLES)} voices")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                
                error_str = str(e)
                if "$env:CUDA_PATH" in error_str:
                    lmdeploy_error_reason = "Không tìm thấy biến môi trường CUDA_PATH. Vui lòng cài đặt NVIDIA GPU Computing Toolkit."
                else:
                    lmdeploy_error_reason = f"{error_str}"
                
                yield (
                    f"⚠️ LMDeploy Init Error: {lmdeploy_error_reason}. Đang loading model với backend mặc định - tốc độ chậm hơn so với lmdeploy...",
                    gr.update(interactive=False),
                    gr.update(interactive=False)
                )
                time.sleep(1)
                use_lmdeploy = False
                using_lmdeploy = False
        
        if not use_lmdeploy:
            print(f"📦 Using original backend")

            if device_choice == "Auto":
                if "gguf" in backbone_choice.lower():
                    # GGUF: uses Metal on Mac, CUDA on Windows/Linux
                    if sys.platform == "darwin":
                        backbone_device = "gpu"  # llama-cpp-python uses Metal
                    else:
                        backbone_device = "gpu" if torch.cuda.is_available() else "cpu"
                else:
                    # PyTorch model
                    if sys.platform == "darwin":
                        backbone_device = "mps" if torch.backends.mps.is_available() else "cpu"
                    else:
                        backbone_device = "cuda" if torch.cuda.is_available() else "cpu"

                # Codec device
                if "ONNX" in codec_choice:
                    codec_device = "cpu"
                elif sys.platform == "darwin":
                    codec_device = "mps" if torch.backends.mps.is_available() else "cpu"
                else:
                    codec_device = "cuda" if torch.cuda.is_available() else "cpu"

            elif device_choice == "MPS":
                backbone_device = "mps"
                codec_device = "mps" if "ONNX" not in codec_choice else "cpu"

            else:
                backbone_device = device_choice.lower()
                codec_device = device_choice.lower()

                if "ONNX" in codec_choice:
                    codec_device = "cpu"

            if "gguf" in backbone_choice.lower() and backbone_device == "cuda":
                backbone_device = "gpu"
            
            print(f"📦 Loading model...")
            print(f"   Backbone: {backbone_config['repo']} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")
            
            tts = VieNeuTTS(
                backbone_repo=backbone_config["repo"],
                backbone_device=backbone_device,
                codec_repo=codec_config["repo"],
                codec_device=codec_device
            )
            using_lmdeploy = False
        
        current_backbone = backbone_choice
        current_codec = codec_choice
        model_loaded = True
        
        # Success message with optimization info
        backend_name = "🚀 LMDeploy (Optimized)" if using_lmdeploy else "📦 Standard"
        device_info = "cuda" if use_lmdeploy else (backbone_device if not use_lmdeploy else "N/A")
        
        streaming_support = "✅ Có" if backbone_config['supports_streaming'] else "❌ Không"
        preencoded_note = "\n⚠️ Codec này cần sử dụng pre-encoded codes (.pt files)" if codec_config['use_preencoded'] else ""
        
        opt_info = ""
        if using_lmdeploy and hasattr(tts, 'get_optimization_stats'):
            stats = tts.get_optimization_stats()
            opt_info = (
                f"\n\n🔧 Tối ưu hóa:"
                f"\n  • Triton: {'✅' if stats['triton_enabled'] else '❌'}"
                f"\n  • Max Batch Size (Default): {stats.get('max_batch_size', 'N/A')}"
                f"\n  • Reference Cache: {stats['cached_references']} voices"
                f"\n  • Prefix Caching: ✅"
            )
        
        warning_msg = ""
        if lmdeploy_error_reason:
             warning_msg = (
                 f"\n\n⚠️ **Cảnh báo:** Không thể kích hoạt LMDeploy (Optimized Backend) do lỗi sau:\n"
                 f"👉 {lmdeploy_error_reason}\n"
                 f"💡 Hệ thống đã tự động chuyển về chế độ Standard (chậm hơn)."
             )

        success_msg = get_model_status_message()
        if warning_msg:
            success_msg += warning_msg
            
        yield (
            success_msg,
            gr.update(interactive=True), # btn_generate
            gr.update(interactive=True), # btn_load
            gr.update(interactive=False) # btn_stop
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        model_loaded = False
        using_lmdeploy = False

        if "$env:CUDA_PATH" in str(e):
            yield (
                "❌ Lỗi khi tải model: Không tìm thấy biến môi trường CUDA_PATH. Vui lòng cài đặt NVIDIA GPU Computing Toolkit (https://developer.nvidia.com/cuda/toolkit)",
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(interactive=False)
            )
        else: 
            yield (
                f"❌ Lỗi khi tải model: {str(e)}",
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(interactive=False)
            )


# --- 2. DATA & HELPERS ---
GGUF_ALLOWED_VOICES = [
    "Bình (nam miền Bắc)",
    "Tuyên (nam miền Bắc)",
    "Vĩnh (nam miền Nam)",
    "Đoan (nữ miền Nam)",
    "Ly (nữ miền Bắc)",
    "Ngọc (nữ miền Bắc)",
]

def get_voice_options(backbone_choice: str):
    """Filter voice options: GGUF only shows the 4 allowed voices."""
    if "gguf" in backbone_choice.lower():
        return [v for v in GGUF_ALLOWED_VOICES if v in VOICE_SAMPLES]
    return list(VOICE_SAMPLES.keys())

def update_voice_dropdown(backbone_choice: str, current_voice: str):
    options = get_voice_options(backbone_choice)
    new_value = current_voice if current_voice in options else (options[0] if options else None)
    return gr.update(choices=options, value=new_value)

# --- 3. CORE LOGIC FUNCTIONS ---
def load_reference_info(voice_choice: str) -> Tuple[Optional[str], str]:
    """Load reference audio and text with caching"""
    if voice_choice in VOICE_SAMPLES:
        audio_path = VOICE_SAMPLES[voice_choice]["audio"]
        text_path = VOICE_SAMPLES[voice_choice]["text"]
        try:
            if os.path.exists(text_path):
                ref_text = get_ref_text_cached(text_path)
                return audio_path, ref_text
            else:
                return audio_path, "⚠️ Không tìm thấy file text mẫu."
        except Exception as e:
            return None, f"❌ Lỗi: {str(e)}"
    return None, ""

def synthesize_speech(text: str, voice_choice: str, custom_audio, custom_text: str, 
                     mode_tab: str, generation_mode: str, use_batch: bool, max_batch_size_run: int,
                     lora_repo_id: str, lora_hf_token: str, lora_audio, lora_text: str,
                     temperature: float = 1.0, crossfade_p: float = 0.05, max_chars_chunk: int = 256):
    """Synthesis with optimization support, max batch size control, and LoRA adapter support"""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    
    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return
    
    if not text or text.strip() == "":
        yield None, "⚠️ Vui lòng nhập văn bản!"
        return
    
    raw_text = text.strip()
    
    codec_config = CODEC_CONFIGS[current_codec]
    use_preencoded = codec_config['use_preencoded']
    
    # Handle LoRA mode
    lora_loaded = False
    if hasattr(tts, '_lora_loaded') and tts._lora_loaded:
        lora_loaded = True

    # If not in LoRA mode but a LoRA is loaded, unload it now to prevent conflicts
    if mode_tab != "lora_mode" and lora_loaded:
        yield None, "🔄 Đang dọn dẹp LoRA adapter để quay về model gốc..."
        try:
            tts.unload_lora_adapter()
            lora_loaded = False
        except Exception as e:
            print(f"Error unloading LoRA: {e}")

    if mode_tab == "lora_mode":
        # Check if using LMDeploy backend
        if using_lmdeploy:
            yield None, (
                "❌ LoRA adapter không hỗ trợ LMDeploy backend!\n\n"
                "💡 Giải pháp:\n"
                "1. Bỏ tick '🚀 Optimize with LMDeploy' ở phần cấu hình\n"
                "2. Click '🔄 Tải Model' lại\n"
                "3. Quay lại tab LoRA và thử lại\n\n"
                "📝 Lưu ý: Khi dùng LoRA, tốc độ sẽ chậm hơn LMDeploy. Hoặc bạn có thể cân nhắc merge LoRA vào model gốc rồi dùng LMDeploy để tối ưu tốc độ."
            )
            return
        
        if not lora_repo_id or not lora_repo_id.strip():
            yield None, "⚠️ Vui lòng nhập HuggingFace Repo ID của LoRA adapter!"
            return
        
        if not lora_audio or not lora_text or not lora_text.strip():
            yield None, "⚠️ Thiếu Audio hoặc Text reference từ tập train của LoRA!"
            return
        
        # Only load if not already loaded or if repo changed
        current_lora = getattr(tts, '_current_lora_repo', None)
        if not lora_loaded or current_lora != lora_repo_id:
            yield None, f"📦 Đang tải LoRA adapter từ {lora_repo_id}..."
            try:
                # Use the new load_lora_adapter method from VieNeuTTS class
                hf_token = lora_hf_token.strip() if lora_hf_token and lora_hf_token.strip() else None
                tts.load_lora_adapter(lora_repo_id, hf_token=hf_token)
                lora_loaded = True
                yield None, "✅ LoRA adapter loaded! Đang xử lý..."
            except NotImplementedError as e:
                yield None, f"❌ {str(e)}\n\nVui lòng chọn backbone PyTorch (VieNeu-TTS hoặc VieNeu-TTS-0.3B GPU), không dùng GGUF."
                return
            except RuntimeError as e:
                error_msg = str(e)
                # Detect backbone mismatch
                suggestion = ""
                if "size mismatch" in error_msg.lower() or "shape" in error_msg.lower():
                    current_backbone_name = BACKBONE_CONFIGS[current_backbone]['repo']
                    suggestion = (
                        f"\n\n💡 **Có thể do backbone không khớp!**\n"
                        f"- Backbone hiện tại: `{current_backbone_name}`\n"
                        f"- Hãy kiểm tra LoRA repo của bạn được train trên model nào\n"
                        f"- Nếu train trên VieNeu-TTS-0.3B → Chọn **VieNeu-TTS-0.3B (GPU)**\n"
                        f"- Nếu train trên VieNeu-TTS (0.5B) → Chọn **VieNeu-TTS (GPU)**"
                    )
                yield None, f"❌ Lỗi khi tải LoRA adapter: {error_msg}{suggestion}"
                return
            except Exception as e:
                import traceback
                traceback.print_exc()
                yield None, f"❌ Lỗi khi tải LoRA adapter: {str(e)}\n\nKiểm tra:\n- Repo ID có đúng không?\n- Token có hợp lệ không (nếu private)?"
                return
        else:
            yield None, f"✅ Sử dụng LoRA đã load: {lora_repo_id}"
        
        # Use LoRA reference audio/text
        ref_audio_path = lora_audio
        ref_text_raw = lora_text
        ref_codes_path = None
        
    # Setup Reference (non-LoRA modes)
    elif mode_tab == "custom_mode":
        if custom_audio is None or not custom_text:
            yield None, "⚠️ Thiếu Audio hoặc Text mẫu custom."
            return
        ref_audio_path = custom_audio
        ref_text_raw = custom_text
        ref_codes_path = None
    else:
        if voice_choice not in VOICE_SAMPLES:
            yield None, "⚠️ Vui lòng chọn giọng mẫu."
            return
        ref_audio_path = VOICE_SAMPLES[voice_choice]["audio"]
        text_path = VOICE_SAMPLES[voice_choice]["text"]
        ref_codes_path = VOICE_SAMPLES[voice_choice]["codes"]
        
        if not os.path.exists(ref_audio_path):
            yield None, "❌ Không tìm thấy file audio mẫu."
            return
        
        ref_text_raw = get_ref_text_cached(text_path)
    
    yield None, "📄 Đang xử lý Reference..."
    
    # Encode or get cached reference
    try:
        if use_preencoded and ref_codes_path and os.path.exists(ref_codes_path):
            ref_codes = torch.load(ref_codes_path, map_location="cpu", weights_only=True)
        else:
            # Use cached reference if available (LMDeploy only)
            if using_lmdeploy and hasattr(tts, 'get_cached_reference') and mode_tab == "preset_mode":
                ref_codes = tts.get_cached_reference(voice_choice, ref_audio_path, ref_text_raw)
            else:
                ref_codes = tts.encode_reference(ref_audio_path)
        
        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()
    except Exception as e:
        yield None, f"❌ Lỗi xử lý reference: {e}"
        return
    
    # === STANDARD MODE ===
    if generation_mode == "Standard (Một lần)":
        backend_name = "LMDeploy" if using_lmdeploy else "Standard"
        
        # Split text here so we can show progress
        text_chunks = split_text_into_chunks(raw_text, max_chars=max_chars_chunk)
        total_chunks = len(text_chunks)
        
        yield None, f"🚀 Bắt đầu tổng hợp {backend_name} ({total_chunks} đoạn)..."
        
        sr = 24000
        start_time = time.time()
        all_wavs = []
        
        try:
            # Case 1: LMDeploy with Batching (Shows batch progress)
            if using_lmdeploy and use_batch and hasattr(tts, 'infer_batch') and total_chunks > 1:
                # Calculate how many mini-batches
                num_batches = (total_chunks + max_batch_size_run - 1) // max_batch_size_run
                yield None, f"⚡ Xử lý {total_chunks} đoạn theo {num_batches} batches (Max size: {max_batch_size_run})..."
                
                # We reuse infer_batch directly for speed but it returns everything at once
                all_wavs = tts.infer_batch(
                    text_chunks, 
                    ref_codes, 
                    ref_text_raw, 
                    max_batch_size=max_batch_size_run,
                    temperature=temperature
                )
            
            # Case 2: Sequential (Shows segment-by-segment progress)
            else:
                for i, chunk in enumerate(text_chunks):
                    yield None, f"⏳ Đang xử lý đoạn {i+1}/{total_chunks}..."
                    
                    wav = tts.infer(
                        chunk, 
                        ref_codes, 
                        ref_text_raw,
                        temperature=temperature
                    )
                    
                    if wav is not None and len(wav) > 0:
                        all_wavs.append(wav)
            
            if not all_wavs:
                yield None, "❌ Không sinh được audio nào."
                return
            
            yield None, "💾 Đang ghép nối và áp dụng hiệu ứng..."
            
            # Use public join_audio_chunks from core for consistent quality
            final_wav = join_audio_chunks(
                all_wavs, 
                sr, 
                crossfade_p=crossfade_p
            )
            
            # Apply watermark manually here since we bypassed tts.infer(whole_text)
            if hasattr(tts, 'watermarker') and tts.watermarker:
                final_wav = tts.watermarker.apply_watermark(final_wav, sample_rate=sr)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                output_path = tmp.name
            
            process_time = time.time() - start_time
            backend_info = f" (Backend: {'LMDeploy 🚀' if using_lmdeploy else 'Standard 📦'})"
            speed_info = f", Tốc độ: {len(final_wav)/sr/process_time:.2f}x realtime" if process_time > 0 else ""
            lora_info = f" [LoRA: {lora_repo_id}]" if lora_loaded else ""
            
            yield output_path, f"✅ Hoàn tất! (Thời gian: {process_time:.2f}s{speed_info}){backend_info}{lora_info}"
            
            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()
            cleanup_gpu_memory()
            
        except torch.cuda.OutOfMemoryError as e:
            cleanup_gpu_memory()
            
            # Build helpful suggestions based on current settings
            suggestions = []
            
            if using_lmdeploy and use_batch:
                suggestions.append(f"• Giảm Max Batch Size (hiện tại: {max_batch_size_run})")
                suggestions.append("• Bỏ tick 'Batch Processing'")
            
            suggestions.append("• Restart và chọn model nhỏ hơn (0.3B)")
            
            yield None, (
                f"❌ GPU hết VRAM! Hãy thử:\n" +
                "\n".join(suggestions) +
                f"\n\nChi tiết: {str(e)}"
            )
            return
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            cleanup_gpu_memory()
            yield None, f"❌ Lỗi Standard Mode: {str(e)}"
            return
    
    # === STREAMING MODE ===
    else:
        sr = 24000
        crossfade_samples = int(sr * crossfade_p)
        audio_queue = queue.Queue(maxsize=100)
        PRE_BUFFER_SIZE = 3
        
        # Split text into chunks for streaming
        text_chunks = split_text_into_chunks(raw_text, max_chars=max_chars_chunk)
        
        end_event = threading.Event()
        error_event = threading.Event()
        error_msg = ""
        
        def producer_thread():
            nonlocal error_msg
            try:
                previous_tail = None
                
                for i, chunk_text in enumerate(text_chunks):
                    stream_gen = tts.infer_stream(chunk_text, ref_codes, ref_text_raw, temperature=temperature)
                    
                    for part_idx, audio_part in enumerate(stream_gen):
                        if audio_part is None or len(audio_part) == 0:
                            continue
                        
                        if previous_tail is not None and len(previous_tail) > 0:
                            overlap = min(len(previous_tail), len(audio_part), crossfade_samples)
                            if overlap > 0:
                                fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                                fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                                
                                blended = (audio_part[:overlap] * fade_in + 
                                         previous_tail[-overlap:] * fade_out)
                                
                                processed = np.concatenate([
                                    previous_tail[:-overlap] if len(previous_tail) > overlap else np.array([]),
                                    blended,
                                    audio_part[overlap:]
                                ])
                            else:
                                processed = np.concatenate([previous_tail, audio_part])
                            
                            tail_size = min(crossfade_samples, len(processed))
                            previous_tail = processed[-tail_size:].copy()
                            output_chunk = processed[:-tail_size] if len(processed) > tail_size else processed
                        else:
                            tail_size = min(crossfade_samples, len(audio_part))
                            previous_tail = audio_part[-tail_size:].copy()
                            output_chunk = audio_part[:-tail_size] if len(audio_part) > tail_size else audio_part
                        
                        if len(output_chunk) > 0:
                            audio_queue.put((sr, output_chunk))
                
                if previous_tail is not None and len(previous_tail) > 0:
                    audio_queue.put((sr, previous_tail))
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                error_event.set()
            finally:
                end_event.set()
                audio_queue.put(None)
        
        threading.Thread(target=producer_thread, daemon=True).start()
        
        yield (sr, np.zeros(int(sr * 0.05))), "📄 Đang buffering..."
        
        pre_buffer = []
        while len(pre_buffer) < PRE_BUFFER_SIZE:
            try:
                item = audio_queue.get(timeout=5.0)
                if item is None:
                    break
                pre_buffer.append(item)
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    return
                break
        
        full_audio_buffer = []
        backend_info = "🚀 LMDeploy" if using_lmdeploy else "📦 Standard"
        for sr, audio_data in pre_buffer:
            full_audio_buffer.append(audio_data)
            yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."
        
        while True:
            try:
                item = audio_queue.get(timeout=0.05)
                if item is None:
                    break
                sr, audio_data = item
                full_audio_buffer.append(audio_data)
                yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    break
                if end_event.is_set() and audio_queue.empty():
                    break
                continue
        
        if full_audio_buffer:
            final_wav = np.concatenate(full_audio_buffer)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                
                lora_info = f" [LoRA: {lora_repo_id}]" if lora_loaded else ""
                yield tmp.name, f"✅ Hoàn tất Streaming! ({backend_info}){lora_info}"
            
            # Cleanup memory
            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()
            
            cleanup_gpu_memory()


# --- 4. UI SETUP ---
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui'],
).set(
    button_primary_background_fill="linear-gradient(90deg, #6366f1 0%, #0ea5e9 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4f46e5 0%, #0284c7 100%)",
)

css = """
.container { max-width: 1400px; margin: auto; }
.header-box {
    text-align: center;
    margin-bottom: 25px;
    padding: 25px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 12px;
    color: white !important;
}
.header-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: white !important;
}
.gradient-text {
    background: -webkit-linear-gradient(45deg, #60A5FA, #22D3EE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-icon {
    color: white;
}
.status-box {
    font-weight: 500;
    border: 1px solid rgba(99, 102, 241, 0.1);
    background: rgba(99, 102, 241, 0.03);
    border-radius: 8px;
}
.status-box textarea {
    text-align: center;
    font-family: inherit;
}
.model-card-content {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 15px;
    font-size: 0.9rem;
    text-align: center;
    color: white !important;
}
.model-card-item {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    color: white !important;
}
.model-card-item strong {
    color: white !important;
}
.model-card-item span {
    color: white !important;
}
.model-card-link {
    color: #60A5FA;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s;
}
.model-card-link:hover {
    color: #22D3EE;
    text-decoration: underline;
}
.warning-banner {
    background-color: #fffbeb;
    border: 1px solid #fef3c7;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
}
.warning-banner-title {
    color: #92400e;
    font-weight: 700;
    font-size: 1.1rem;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}
.warning-banner-grid {
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
}
.warning-banner-item {
    flex: 1;
    min-width: 240px;
    background: #fef3c7;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #fde68a;
}
.warning-banner-item strong {
    color: #b45309;
    display: block;
    margin-bottom: 4px;
    font-size: 0.95rem;
}
.warning-banner-content {
    color: #78350f;
    font-size: 0.9rem;
    line-height: 1.5;
}
.warning-banner-content b {
    color: #451a03;
    background: rgba(251, 191, 36, 0.2);
    padding: 1px 4px;
    border-radius: 4px;
}
"""

EXAMPLES_LIST = [
    ["Về miền Tây không chỉ để ngắm nhìn sông nước hữu tình, mà còn để cảm nhận tấm chân tình của người dân nơi đây.", "Vĩnh (nam miền Nam)"],
    ["Hà Nội những ngày vào thu mang một vẻ đẹp trầm mặc và cổ kính đến lạ thường.", "Bình (nam miền Bắc)"],
]

with gr.Blocks(theme=theme, css=css, title="VieNeu-TTS") as demo:
    with gr.Column(elem_classes="container"):
        gr.HTML("""
<div class="header-box">
    <h1 class="header-title">
        <span class="header-icon">🦜</span>
        <span class="gradient-text">VieNeu-TTS Studio</span>
    </h1>
    <div class="model-card-content">
        <div class="model-card-item">
            <strong>Models:</strong>
            <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS" target="_blank" class="model-card-link">VieNeu-TTS</a>
            <span>•</span>
            <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B" target="_blank" class="model-card-link">VieNeu-TTS-0.3B</a>
        </div>
        <div class="model-card-item">
            <strong>Repository:</strong>
            <a href="https://github.com/pnnbao97/VieNeu-TTS" target="_blank" class="model-card-link">GitHub</a>
        </div>
        <div class="model-card-item">
            <strong>Tác giả:</strong>
            <a href="https://www.facebook.com/bao.phamnguyenngoc.5" target="_blank" class="model-card-link">Phạm Nguyễn Ngọc Bảo</a>
        </div>
        <div class="model-card-item">
            <strong>Discord:</strong>
            <a href="https://discord.gg/yJt8kzjzWZ" target="_blank" class="model-card-link">Tham gia cộng đồng</a>
        </div>
    </div>
</div>
        """)
        
        # --- CONFIGURATION ---
        with gr.Group():
            with gr.Row():
                backbone_select = gr.Dropdown(
                    list(BACKBONE_CONFIGS.keys()), 
                    value="VieNeu-TTS (GPU)", 
                    label="🦜 Backbone"
                )
                codec_select = gr.Dropdown(list(CODEC_CONFIGS.keys()), value="NeuCodec (Distill)", label="🎵 Codec")
                device_choice = gr.Radio(get_available_devices(), value="Auto", label="🖥️ Device")
            
            with gr.Row():
                use_lmdeploy_cb = gr.Checkbox(
                    value=True, 
                    label="🚀 Optimize with LMDeploy (Khuyên dùng cho NVIDIA GPU)",
                    info="Tick nếu bạn dùng GPU để tăng tốc độ tổng hợp đáng kể."
                )
            
            gr.HTML("""
            <div class="warning-banner">
                <div class="warning-banner-title">
                    🦜 Gợi ý tối ưu hiệu năng
                </div>
                <div class="warning-banner-grid">
                    <div class="warning-banner-item">
                        <strong>🐢 Hệ máy CPU</strong>
                        <div class="warning-banner-content">
                            Sử dụng <b>VieNeu-TTS-0.3B-q4-gguf</b> để đạt tốc độ xử lý nhanh nhất. Nếu ưu tiên độ chính xác thì dùng <b>VieNeu-TTS-0.3B-q8-gguf</b>.
                        </div>
                    </div>
                    <div class="warning-banner-item">
                        <strong>🐆 Hệ máy GPU</strong>
                        <div class="warning-banner-content">
                            Chọn <b>VieNeu-TTS-0.3B (GPU)</b> để x2 tốc độ (độ chính xác ~95% bản gốc).
                        </div>
                    </div>
                </div>
            </div>
            """)

            btn_load = gr.Button("🔄 Tải Model", variant="primary")
            model_status = gr.Markdown("⏳ Chưa tải model.")
        
        with gr.Row(elem_classes="container"):
            # --- INPUT ---
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label=f"Văn bản",
                    lines=4,
                    value="Hà Nội, trái tim của Việt Nam, là một thành phố ngàn năm văn hiến với bề dày lịch sử và văn hóa độc đáo. Bước chân trên những con phố cổ kính quanh Hồ Hoàn Kiếm, du khách như được du hành ngược thời gian, chiêm ngưỡng kiến trúc Pháp cổ điển hòa quyện với nét kiến trúc truyền thống Việt Nam. Mỗi con phố trong khu phố cổ mang một tên gọi đặc trưng, phản ánh nghề thủ công truyền thống từng thịnh hành nơi đây như phố Hàng Bạc, Hàng Đào, Hàng Mã. Ẩm thực Hà Nội cũng là một điểm nhấn đặc biệt, từ tô phở nóng hổi buổi sáng, bún chả thơm lừng trưa hè, đến chè Thái ngọt ngào chiều thu. Những món ăn dân dã này đã trở thành biểu tượng của văn hóa ẩm thực Việt, được cả thế giới yêu mến. Người Hà Nội nổi tiếng với tính cách hiền hòa, lịch thiệp nhưng cũng rất cầu toàn trong từng chi tiết nhỏ, từ cách pha trà sen cho đến cách chọn hoa sen tây để thưởng trà.",
                )
                
                with gr.Tabs() as tabs:
                    with gr.TabItem("👤 Preset", id="preset_mode") as tab_preset:
                        initial_voices = get_voice_options("VieNeu-TTS (GPU)")
                        default_voice = initial_voices[0] if initial_voices else None
                        voice_select = gr.Dropdown(initial_voices, value=default_voice, label="Giọng mẫu")
                    
                    with gr.TabItem("🦜 Voice Cloning", id="custom_mode") as tab_custom:
                        custom_audio = gr.Audio(label="Audio giọng mẫu (3-5 giây) (.wav)", type="filepath")
                        custom_text = gr.Textbox(label="Nội dung audio mẫu - vui lòng gõ đúng nội dung của audio mẫu - kể cả dấu câu vì model rất nhạy cảm với dấu câu (.,?!)")
                        gr.Examples(
                            examples=[
                                [os.path.join("examples", "audio_ref", "example.wav"), "Ví dụ 2. Tính trung bình của dãy số."],
                                [os.path.join("examples", "audio_ref", "example_2.wav"), "Trên thực tế, các nghi ngờ đã bắt đầu xuất hiện."],
                                [os.path.join("examples", "audio_ref", "example_3.wav"), "Cậu có nhìn thấy không?"],
                                [os.path.join("examples", "audio_ref", "example_4.wav"), "Tết là dịp mọi người háo hức đón chào một năm mới với nhiều hy vọng và mong ước."]
                            ],
                            inputs=[custom_audio, custom_text],
                            label="Ví dụ mẫu để thử nghiệm clone giọng"
                        )
                        
                        gr.Markdown("""
                        **💡 Mẹo nhỏ:** Nếu kết quả Zero-shot Voice Cloning chưa như ý, bạn hãy cân nhắc **Finetune (LoRA)** để đạt chất lượng tốt nhất. 
                        Hướng dẫn chi tiết có tại file: `finetune/README.md` hoặc xem trên [GitHub](https://github.com/pnnbao97/VieNeu-TTS/tree/main/finetune).
                        """)
                    
                    with gr.TabItem("🎯 LoRA Adapter", id="lora_mode") as tab_lora:
                        gr.Markdown("""
                        ### 🎓 Sử dụng LoRA Adapter đã fine-tune
                        
                        Tải LoRA adapter từ HuggingFace để sử dụng giọng nói đã được fine-tune.
                        
                        ⚠️ **QUAN TRỌNG - Yêu cầu:**
                        
                        **1. Backbone phải khớp:**
                        - Nếu train LoRA trên **VieNeu-TTS-0.3B** → Phải chọn backbone **VieNeu-TTS-0.3B (GPU)** ở trên
                        - Nếu train LoRA trên **VieNeu-TTS** (0.5B) → Phải chọn backbone **VieNeu-TTS (GPU)** ở trên
                        
                        **2. KHÔNG dùng với:**
                        - ❌ GGUF models (chỉ hỗ trợ PyTorch backbone)
                        - ❌ LMDeploy optimization (bỏ tick "🚀 Optimize with LMDeploy")
                        
                        💡 Kiểm tra model base trong file `adapter_config.json` của LoRA repo để biết model nào được dùng.
                        """)
                        
                        with gr.Row():
                            lora_repo_id = gr.Textbox(
                                label="🤗 HuggingFace Repo ID",
                                placeholder="vd: pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen",
                                value="pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen",
                                info="Nhập repo ID của LoRA adapter trên HuggingFace"
                            )
                            lora_hf_token = gr.Textbox(
                                label="🔑 HF Token (nếu repo private)",
                                placeholder="Để trống nếu repo public",
                                type="password",
                                info="Token để truy cập repo private"
                            )
                        
                        gr.Markdown("**📤 Upload Audio mẫu từ tập train của LoRA**")
                        lora_audio = gr.Audio(
                            label="Audio reference (phải là audio từ tập train của LoRA)",
                            type="filepath",
                            value=os.path.join("examples", "audio_ref", "example_ngoc_huyen.wav")
                        )
                        lora_text = gr.Textbox(
                            label="Text tương ứng với audio reference",
                            placeholder="Nhập chính xác nội dung của audio reference...",
                            value="Tác phẩm dự thi bảo đảm tính khoa học, tính đảng, tính chiến đấu, tính định hướng."
                        )

                        gr.Examples(
                            examples=[
                                [
                                    "pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen",
                                    "", # hf token
                                    os.path.join("examples", "audio_ref", "example_ngoc_huyen.wav"),
                                    "Tác phẩm dự thi bảo đảm tính khoa học, tính đảng, tính chiến đấu, tính định hướng."
                                ]
                            ],
                            inputs=[lora_repo_id, lora_hf_token, lora_audio, lora_text],
                            label="Ví dụ mẫu LoRA Ngọc Huyền"
                        )

                
                generation_mode = gr.Radio(
                    ["Standard (Một lần)"],
                    value="Standard (Một lần)",
                    label="Chế độ sinh"
                )
                with gr.Row():
                    use_batch = gr.Checkbox(
                        value=True, 
                        label="⚡ Batch Processing",
                        info="Xử lý nhiều đoạn cùng lúc (chỉ áp dụng khi sử dụng GPU và đã cài đặt LMDeploy)"
                    )
                    max_batch_size_run = gr.Slider(
                        minimum=1, 
                        maximum=16, 
                        value=4, 
                        step=1, 
                        label="📊 Batch Size (Generation)",
                        info="Số lượng đoạn văn bản xử lý cùng lúc. Giá trị cao = nhanh hơn nhưng tốn VRAM hơn. Giảm xuống nếu gặp lỗi Out of Memory."
                    )
                
                # Advanced settings
                with gr.Accordion("⚙️ Cài đặt nâng cao", open=False):
                    with gr.Row():
                        temperature_slider = gr.Slider(
                            minimum=0.4,
                            maximum=1.4,
                            value=1.0,
                            step=0.05,
                            label="🌡️ Temperature",
                            info="Độ sáng tạo. Thấp = ổn định, Cao = đa dạng nhưng có thể kém tự nhiên."
                        )
                        crossfade_slider = gr.Slider(
                            minimum=0.0,
                            maximum=0.2,
                            value=0.05,
                            step=0.01,
                            label="🎵 Crossfade (giây)",
                            info="Độ dài fade giữa các đoạn. 0 = không fade, 0.05-0.1 = mượt."
                        )
                    
                    max_chars_slider = gr.Slider(
                        minimum=128,
                        maximum=512,
                        value=256,
                        step=32,
                        label="📝 Max Chars Per Chunk",
                        info="Độ dài tối đa mỗi đoạn văn bản. Nhỏ = ổn định hơn, Lớn = ít chunk hơn."
                    )
                
                # State to track current mode (replaces unreliable Textbox/Tabs input)
                current_mode_state = gr.State("preset_mode")
                
                with gr.Row():
                    btn_generate = gr.Button("🎵 Bắt đầu", variant="primary", scale=2, interactive=False)
                    btn_stop = gr.Button("⏹️ Dừng", variant="stop", scale=1, interactive=False)
            
            # --- OUTPUT ---
            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="Kết quả",
                    type="filepath",
                    autoplay=True
                )
                status_output = gr.Textbox(
                    label="Trạng thái", 
                    elem_classes="status-box",
                    lines=2,
                    max_lines=10,
                    show_copy_button=True
                )
                gr.Markdown("<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>🔒 Audio được đóng dấu bản quyền ẩn (Watermarker) để bảo mật và định danh AI.</div>")
        
        # # --- EVENT HANDLERS ---
        # def update_info(backbone: str) -> str:
        #     return f"Streaming: {'✅' if BACKBONE_CONFIGS[backbone]['supports_streaming'] else '❌'}"
        
        # backbone_select.change(update_info, backbone_select, model_status)
        backbone_select.change(update_voice_dropdown, [backbone_select, voice_select], voice_select)
        
        # Handler to show/hide Voice Cloning tab
        def on_codec_change(codec: str):
            is_onnx = "onnx" in codec.lower()
            # If switching to ONNX and we are on custom mode, switch back to preset
            return gr.update(visible=not is_onnx), gr.update(selected="preset_mode" if is_onnx else None)
        
        codec_select.change(
            on_codec_change, 
            inputs=[codec_select], 
            outputs=[tab_custom, tabs]
        )
        
        # Bind tab events to update state
        tab_preset.select(lambda: "preset_mode", outputs=current_mode_state)
        tab_custom.select(lambda: "custom_mode", outputs=current_mode_state)
        tab_lora.select(lambda: "lora_mode", outputs=current_mode_state)
        
        btn_load.click(
            fn=load_model,
            inputs=[backbone_select, codec_select, device_choice, use_lmdeploy_cb],
            outputs=[model_status, btn_generate, btn_load, btn_stop]
        )
        
        generate_event = btn_generate.click(
            fn=synthesize_speech,
            inputs=[text_input, voice_select, custom_audio, custom_text, current_mode_state, 
                    generation_mode, use_batch, max_batch_size_run,
                    lora_repo_id, lora_hf_token, lora_audio, lora_text,
                    temperature_slider, crossfade_slider, max_chars_slider],
            outputs=[audio_output, status_output]
        )
        
        # When generation starts, enable stop button
        btn_generate.click(lambda: gr.update(interactive=True), outputs=btn_stop)
        # When generation ends/stops, disable stop button
        generate_event.then(lambda: gr.update(interactive=False), outputs=btn_stop)
        
        btn_stop.click(fn=None, cancels=[generate_event])
        btn_stop.click(lambda: (None, "⏹️ Đã dừng tạo giọng nói."), outputs=[audio_output, status_output])
        btn_stop.click(lambda: gr.update(interactive=False), outputs=btn_stop)

        # Persistence: Restore UI state on load
        demo.load(
            fn=restore_ui_state,
            outputs=[model_status, btn_generate, btn_stop]
        )

if __name__ == "__main__":
    # Cho phép override từ biến môi trường (hữu ích cho Docker)
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    # Check running in Colab
    is_on_colab = os.getenv("COLAB_RELEASE_TAG") is not None

    # Default:
    # - Colab: share=True (convenient)
    # - Docker/local: share=False (safe)
    share = env_bool("GRADIO_SHARE", default=is_on_colab)
    #
    # If server_name is "0.0.0.0" and GRADIO_SHARE is not set, disable sharing
    if server_name == "0.0.0.0" and os.getenv("GRADIO_SHARE") is None:
        share = False

    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)
