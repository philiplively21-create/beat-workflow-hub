import sys
import torch
import librosa
import soundfile
import subprocess

print("=" * 50)
print("  BEAT WORKFLOW HUB - LOCAL ENVIRONMENT CHECK")
print("=" * 50)

# 1. Check Python & PyTorch
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")

# 2. Check Apple Silicon MPS (Metal Performance Shaders) Acceleration
if torch.backends.mps.is_available():
    print("🚀 Apple Silicon MPS (GPU) Acceleration: AVAILABLE & READY")
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("🚀 CUDA GPU Acceleration: AVAILABLE")
    device = torch.device("cuda")
else:
    print("⚠️ GPU Acceleration not found. Falling back to CPU (processing will be slower).")
    device = torch.device("cpu")

# 3. Check FFmpeg availability
try:
    ffmpeg_version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    print("✅ FFmpeg: INSTALLED")
except Exception:
    print("❌ FFmpeg: NOT FOUND in PATH (required for audio rendering)")

# 4. Check Librosa & SoundFile
print(f"✅ Librosa Version: {librosa.__version__}")
print("✅ SoundFile: READY")
print("=" * 50)
