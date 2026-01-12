# VieNeu-TTS

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/pnnbao97/VieNeu-TTS)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-0.5B-yellow)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-0.3B-orange)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-0.3B--GGUF-green)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B-q8-gguf)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/yJt8kzjzWZ)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V1DjG-KdmurCAhvXrxxTLsa9tteDxSVO?usp=sharing) 

<img width="899" height="615" alt="Untitled" src="https://github.com/user-attachments/assets/7eb9b816-6ab7-4049-866f-f85e36cb9c6f" />

**VieNeu-TTS** is an advanced on-device Vietnamese Text-to-Speech (TTS) model with **instant voice cloning**.

> [!TIP]
> **Voice Cloning:** All model variants (including GGUF) support instant voice cloning with just **3-5 seconds** of reference audio. 

This project features two core architectures trained on the [VieNeu-TTS-1000h](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h) dataset:
- **VieNeu-TTS (0.5B):** An enhanced model fine-tuned from the NeuTTS Air architecture for maximum stability.
- **VieNeu-TTS-0.3B:** A specialized model **trained from scratch**, delivering 2x faster inference and ultra-low latency.

These represent a significant upgrade from the previous VieNeu-TTS-140h with the following improvements:

- **Enhanced pronunciation**: More accurate and stable Vietnamese pronunciation
- **Code-switching support**: Seamless transitions between Vietnamese and English
- **Better voice cloning**: Higher fidelity and speaker consistency
- **Real-time synthesis**: 24 kHz waveform generation on CPU or GPU
- **Multiple model formats**: Support for PyTorch, GGUF Q4/Q8 (CPU optimized), and ONNX codec

VieNeu-TTS delivers production-ready speech synthesis fully offline.

**Author:** Phạm Nguyễn Ngọc Bảo

---

[<img width="600" height="595" alt="VieNeu-TTS" src="https://github.com/user-attachments/assets/6b32df9d-7e2e-474f-94c8-43d6fa586d15" />](https://github.com/user-attachments/assets/6b32df9d-7e2e-474f-94c8-43d6fa586d15)

---

## 🔬 Model Overview

- **Backbone:** 
  - **VieNeu-TTS (0.5B):** Qwen-0.5B fine-tuned from [NeuTTS Air](https://huggingface.co/neuphonic/neutts-air).
  - **VieNeu-TTS-0.3B:** Custom 0.3B model **trained from scratch**, optimized for extreme speed (2x faster).
- **Audio codec:** NeuCodec (torch implementation; ONNX & quantized variants supported)
- **Context window:** 2,048 tokens shared by prompt text and speech tokens
- **Output watermark:** Enabled by default
- **Training data:** [VieNeu-TTS-1000h](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h) — 443,641 curated Vietnamese samples (Used for both versions).

### Model Variants

| Model                   | Format  | Device  | Quality    | Speed                   |
| ----------------------- | ------- | ------- | ---------- | ----------------------- |
| VieNeu-TTS              | PyTorch | GPU/CPU | ⭐⭐⭐⭐⭐ | Very Fast with lmdeploy |
| VieNeu-TTS-0.3B         | PyTorch | GPU/CPU | ⭐⭐⭐⭐   | **Ultra Fast (2x)**     |
| VieNeu-TTS-q8-gguf      | GGUF Q8 | CPU/GPU | ⭐⭐⭐⭐   | Fast                    |
| VieNeu-TTS-q4-gguf      | GGUF Q4 | CPU/GPU | ⭐⭐⭐     | Very Fast               |
| VieNeu-TTS-0.3B-q8-gguf | GGUF Q8 | CPU/GPU | ⭐⭐⭐⭐   | **Ultra Fast (1.5x)**   |
| VieNeu-TTS-0.3B-q4-gguf | GGUF Q4 | CPU/GPU | ⭐⭐⭐     | **Extreme Speed (2x)**  |

**Recommendations:**

- **GPU users**: Use `VieNeu-TTS` (PyTorch) for best quality
- **CPU users**: Use `VieNeu-TTS-0.3B-q4-gguf` for fastest inference or `VieNeu-TTS-0.3B-q8-gguf` for best CPU quality.
- **Streaming**: Only GGUF models support streaming inference (Requires `llama-cpp-python >= 0.3.16`)

---

## ✅ Todo & Status

- [x] Publish safetensor artifacts
- [x] Release GGUF Q4 / Q8 models
- [x] Release datasets (1000h and 140h)
- [x] Enable streaming on GPU
- [x] Provide Dockerized setup
- [x] Release fine-tuning code (LoRA)
- [x] LoRA Adapter integration in Gradio

---

## 🌟 New Feature: LoRA Adapters

VieNeu-TTS now officially supports **LoRA (Low-Rank Adaptation)**. This allows you to:
- Use custom fine-tuned voices from Hugging Face.
- Achieve much higher quality and similarity than zero-shot voice cloning.
- Switch between different adapters seamlessly in the Gradio UI.

For more details, see [docs/LORA_USAGE.md](docs/LORA_USAGE.md).

---

## 🛠️ Fine-tuning

You can now train VieNeu-TTS on your own voice dataset! 
- **Simple Workflow**: Follow the step-by-step guide in [finetune/README.md](finetune/README.md).
- **Notebook Support**: Use `finetune/finetune_VieNeu-TTS.ipynb` for an interactive experience.

---

## 🏁 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/pnnbao97/VieNeu-TTS.git
cd VieNeu-TTS
```

### 2. Install eSpeak NG (Required)
Phonemizer requires eSpeak NG to function.

- **Windows:** Download installer from [eSpeak NG Releases](https://github.com/espeak-ng/espeak-ng/releases) (Recommended: `.msi`).
- **macOS:** `brew install espeak`
- **Ubuntu/Debian:** `sudo apt install espeak-ng`
- **Arch Linux:** `paru -S aur/espeak-ng`

---

### 3. Environment Setup (Recommended)

**A. Install `uv`** (Fast Python package manager):
- **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **Linux/macOS:** `curl -LsSf https://astral.sh/uv/install.sh | sh`

**B. Install dependencies:**

> [!TIP]
> **For NVIDIA GPU Users:** To use LMDeploy (Turbo mode), ensure you have updated drivers and **CUDA Toolkit 12.8+**.

```bash
# Default setup (Includes GPU support)
uv sync

# CPU-only mode (No GPU dependencies)
uv sync --no-default-groups
```

---

### 4. Quick Start (Web UI)

Once environment is ready, start the Web UI with a single command:

```bash
uv run gradio_app.py
```

Access the Web UI at `http://127.0.0.1:7860`.

---


---

### 📦 Using as a Python SDK (via `pip`)

If you want to integrate VieNeu-TTS into your own project:

#### 1. Windows (Hassle-free setup)
We provide pre-built CPU wheels for `llama-cpp-python` (version 0.3.16) for Python 3.10 to 3.14 to avoid compilation errors.

```bash
pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/
```

#### 2. Linux / macOS / Others
```bash
pip install vieneu
```

#### 3. GPU Support (Remote Server)

For high-performance GPU inference without local complexity, you can set up a remote server using `lmdeploy`.

**A. On the Server (with GPU):**
1. Install LMDeploy: `pip install lmdeploy[all]`
2. Launch the API Server:
```bash
lmdeploy serve api_server pnnbao-ump/VieNeu-TTS-0.3B --server-port 23333 --tp 1
```

**B. On the Client (CPU/Laptop):**
Connect to the server using the SDK:
```python
from vieneu import Vieneu

# Connect to the remote server
tts = Vieneu(mode="remote", api_base="http://your-server-ip:23333/v1", model_name="pnnbao-ump/VieNeu-TTS-0.3B")
```

#### 4. Advanced Usage Example (Full Features)

Here is a comprehensive example showing how to initialize, manage voices, clone custom voices, and control generation.

```python
"""
Demo VieNeuSDK v1.1.3 - Full Features Guide
"""

import time
import soundfile as sf
from vieneu import Vieneu
from pathlib import Path

def main():
    print("🚀 Initializing VieNeu SDK (v1.1.3)...")
    
    # Initialize SDK
    # Default: "pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf" (Speed & CPU Optimized)
    #
    # You can change 'backbone_repo' to balance Quality vs Speed:
    # 1. Better Quality (slower than q4): "pnnbao-ump/VieNeu-TTS-0.3B-q8-gguf"
    # 2. PyTorch 0.3B (Fast, uncompressed): "pnnbao-ump/VieNeu-TTS-0.3B"
    # 3. PyTorch 0.5B (Best Quality, heavy): "pnnbao-ump/VieNeu-TTS"
    # You can also use a GGUF version merged with your own LoRA adapter.
    # See finetuning guide: https://github.com/pnnbao97/VieNeu-TTS/tree/main/finetune
    
    # Mode selection:
    # - mode="standard" (Default): Runs locally using GGUF (CPU) or PyTorch
    # - mode="remote": Connects to the LMDeploy server setup above for max speed
    
    tts = Vieneu()
    # Or to use Remote mode (Must start 'lmdeploy serve api_server pnnbao-ump/VieNeu-TTS-0.3B --server-port 23333 --tp 1' in another tab/machine first):
    # tts = Vieneu(mode="remote", api_base="http://localhost:23333/v1", model_name="pnnbao-ump/VieNeu-TTS-0.3B")
    # Example for using Q8 for better quality:
    # tts = Vieneu(backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-q8-gguf")

    # ---------------------------------------------------------
    # PART 1: PRESET VOICES
    # ---------------------------------------------------------
    print("\n--- 1. Available Preset Voices ---")
    available_voices = tts.list_preset_voices()
    print("📋 Voices:", available_voices)
    
    # Select a preset voice
    current_voice = tts.get_preset_voice("Binh")
    print("✅ Selected voice: Binh")


    # ---------------------------------------------------------
    # PART 2: CREATE & SAVE CUSTOM VOICE
    # ---------------------------------------------------------
    print("\n--- 2. Create Custom Voice ---")
    
    # Replace with your actual .wav file path and its exact transcript (including punctuation)
    sample_audio = Path(__file__).parent / "example.wav"
    sample_text = "ví dụ 2. tính trung bình của dãy số."

    if sample_audio.exists():
        voice_name = "MyCustomVoice"
        
        print(f"🎙️ Cloning voice from: {sample_audio.name}")
        
        # 'clone_voice' now supports saving directly with 'name' argument
        custom_voice = tts.clone_voice(
            audio_path=sample_audio,
            text=sample_text,
            name=voice_name  # <-- Automatically saves voice to system
        )
        
        print(f"✅ Voice created and saved as: '{voice_name}'")
        
        # Verify functionality
        print("📋 Voice list after adding:", tts.list_preset_voices())
        
        # Switch to new voice
        current_voice = custom_voice
    else:
        print("⚠️ Sample audio not found. Skipping...")


    # ---------------------------------------------------------
    # PART 3: SYNTHESIS WITH ADVANCED PARAMETERS
    # ---------------------------------------------------------
    print("\n--- 3. Speech Synthesis ---")
    
    text_input = "Xin chào, tôi là VieNeu-TTS. Tôi có thể giúp bạn đọc sách, làm chatbot thời gian thực, hoặc thậm chí clone giọng nói của bạn."
    
    # Generate with specific temperature
    print("🎧 Generating...")
    audio = tts.infer(
        text=text_input,
        voice=current_voice,
        temperature=1.0,  # Adjustable: Lower (0.1) -> Stable, Higher (1.0+) -> Expressive
        top_k=50
    )
    sf.write("output.wav", audio, 24000)
    print("💾 Saved: output.wav")

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------
    tts.close()
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
```


#### Method 2: Automatic with Makefile (Alternative)
Best if you have `make` installed (standard on Linux/macOS, or via Git Bash on Windows). It handles configuration swaps automatically.

- **Setup:** `make setup`
- **Run Demo:** `make demo`


Then access the Web UI at `http://127.0.0.1:7860`.

---

---

## 🐋 Docker Deployment

For a quick start or production deployment without manually installing dependencies, use Docker.

### Quick Start

Copy .env.example to .env

```
cp .env.example .env
```

Build and start container

```bash
# Run with CPU
docker compose --profile cpu up

# Run with GPU (requires NVIDIA Container Toolkit)
docker compose --profile gpu up
```

Access the Web UI at `http://localhost:7860`.

For detailed deployment instructions, including production setup, see [docs/Deploy.md](docs/Deploy.md).

---

## 📦 Project Structure

```
VieNeu-TTS/
├── vieneu/            # Core engine implementation (VieNeuTTS & FastVieNeuTTS)
├── finetune/              # LoRA training pipeline
│   ├── configs/           # Training & LoRA configurations
│   ├── data_scripts/      # Data filtering & VQ encoding tools
│   ├── dataset/           # Training data storage
│   ├── output/            # Saved checkpoints & LoRA adapters
│   └── train.py           # Main training script
├── utils/                 # Text normalization and phonemization logic
├── sample/                # Built-in reference voices (audio + transcript + codes)
├── docs/                  # Detailed documentation for LoRA, Deployment, and Docker
├── examples/              # Usage examples and testing audio references
├── gradio_app.py          # Modern Web UI with LoRA & Streaming support
├── config.yaml            # Model, Codec, and Voice registry
├── pyproject.toml         # Unified dependency management (UV/PIP)
├── Makefile               # Shortcuts for setup and execution
└── docker-compose.yml     # Docker orchestration for CPU/GPU modes
```

---

## 📚 References

- [GitHub Repository](https://github.com/pnnbao97/VieNeu-TTS)
- [Hugging Face Model (0.5B)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)
- [Hugging Face Model (0.3B)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B)
- [LoRA Usage Guide](docs/LORA_USAGE.md)
- [Fine-tuning Guide](finetune/README.md)
- [VieNeu-TTS-1000h dataset](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h)

---

## 📄 License

- **VieNeu-TTS (0.5B):** Original terms (Apache 2.0).
- **VieNeu-TTS-0.3B:** Released under **CC BY-NC 4.0** (Non-Commercial). 
  - This version is currently **experimental**.
  - **Commercial use is prohibited** without authorization. Please contact the author for commercial licensing.

---

## 📑 Citation

```bibtex
@misc{vieneutts2026,
  title        = {VieNeu-TTS: Vietnamese Text-to-Speech with Instant Voice Cloning},
  author       = {Pham Nguyen Ngoc Bao},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/pnnbao-ump/VieNeu-TTS}}
}
```

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "Add amazing feature"`
4. Push the branch: `git push origin feature/amazing-feature`
5. Open a pull request

---

## 📞 Support

- GitHub Issues: [github.com/pnnbao97/VieNeu-TTS/issues](https://github.com/pnnbao97/VieNeu-TTS/issues)
- Hugging Face: [huggingface.co/pnnbao-ump](https://huggingface.co/pnnbao-ump)
- Discord: [Join with us](https://discord.gg/yJt8kzjzWZ)
- Facebook: [Phạm Nguyễn Ngọc Bảo](https://www.facebook.com/bao.phamnguyenngoc.5)

---

## 🙏 Acknowledgements

This project builds upon [NeuTTS Air](https://huggingface.co/neuphonic/neutts-air) for the original 0.5B model. The 0.3B version is a custom architecture trained from scratch using the [VieNeu-TTS-1000h](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h) dataset.

---

**Made with ❤️ for the Vietnamese TTS community**
