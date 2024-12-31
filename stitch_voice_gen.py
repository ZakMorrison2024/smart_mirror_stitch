# Tacotron 2 for Synthesizing Stitch Voice

# Step 1: Clone Tacotron 2 Repository
# Make sure you have Git installed before running this shell command.
!git clone https://github.com/NVIDIA/tacotron2.git

# Step 2: Install Dependencies
# Install required libraries for Tacotron 2.
!pip install -r tacotron2/requirements.txt

# Install PyTorch and TorchAudio compatible with your CUDA version.
!pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

# Step 3: Prepare Dataset
# Organize dataset folder: Ensure audio files are in `data/audio/` and transcripts are in `data/transcripts.txt`.
import shutil

data_dir = "data"
os.makedirs(f"{data_dir}/audio", exist_ok=True)

# Move your Stitch voice clips into `data/audio/` and create a `transcripts.txt` file.
# Example transcript format: "audio_001.wav|Ohana means family."

transcript_content = """
audio_001.wav|Ohana means family.
audio_002.wav|I am Stitch!
audio_003.wav|Stitch is fluffy!
"""
with open(f"{data_dir}/transcripts.txt", "w") as f:
    f.write(transcript_content)

print("Dataset prepared.")

# Step 4: Preprocess Data
# Preprocess dataset into mel-spectrograms and text.
!python tacotron2/preprocess.py --output_directory="preprocessed_data" --dataset_path="data"

# Step 5: Train Tacotron 2 Model
# Customize hyperparameters and train the model.
!python tacotron2/train.py \
    --output_directory="output_dir" \
    --log_directory="log_dir" \
    --dataset_path="data"

# Monitor training with TensorBoard.
!tensorboard --logdir log_dir

# Step 6: Generate Speech (Inference)
# Load trained Tacotron 2 model and synthesize text.
import torch
from tacotron2.text import text_to_sequence
from tacotron2.hparams import create_hparams
from tacotron2.model import Tacotron2
from tacotron2.layers import TacotronSTFT

# Load Tacotron 2 model
hparams = create_hparams()
hparams.sampling_rate = 22050
model = Tacotron2(hparams)
checkpoint_path = "output_dir/tacotron2_checkpoint.pth"
model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
model.eval()

# Convert text to mel-spectrogram
text = "Ohana means family."
sequence = torch.tensor(text_to_sequence(text, ["english_cleaners"]))
sequence = sequence.unsqueeze(0).cuda()

with torch.no_grad():
    mel_outputs, mel_outputs_postnet, _, _ = model.inference(sequence)

# Step 8: Convert Mel-Spectrogram to Audio
# Use WaveGlow vocoder to generate audio from mel-spectrogram.
from waveglow.denoiser import Denoiser

waveglow = torch.load("pretrained_models/waveglow_256channels_universal_v5.pt")["model"]
waveglow.cuda().eval()
denoser = Denoiser(waveglow).cuda()

with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    audio = denoser(audio, strength=0.01)[:, 0].cpu().numpy()

# Save audio
from scipy.io.wavfile import write
write("output_audio.wav", hparams.sampling_rate, audio[0])

print("Audio generated and saved as output_audio.wav.")

# Step 9: Post-Processing for Stitch-Like Voice
# Use audio editing software (e.g., Audacity) to:
# - Adjust pitch slightly higher.
# - Add raspiness or distortion effects.
# - Fine-tune formants and EQ settings for alien-like quality.
