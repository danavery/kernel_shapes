import os

import torch
import torchaudio
from torch import nn
from tqdm import tqdm

from .esc50_dataset import ESC50Dataset
from .utils import LogNormalize

# Your transform
spec_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=44100, n_fft=1024, hop_length=256, n_mels=128
    ),
    LogNormalize(top_db=80),
)

# Dataset to get filenames and load audio
dataset = ESC50Dataset(
    root_dir="ESC-50-master/audio",
    meta_csv="ESC-50-master/meta/esc50.csv",
    precomputed=False,
    transform=None,  # We’ll do it manually
)

output_dir = "./data/specs"
os.makedirs(output_dir, exist_ok=True)

for i in tqdm(range(len(dataset))):
    record = dataset[i]
    audio = record["data"]  # waveform
    filename = record["filename"]
    spec = spec_transform(audio)
    if i == 0:
        print(audio.shape)
        print(spec.shape)
    torch.save(spec, os.path.join(output_dir, filename.replace(".wav", ".pt")))
