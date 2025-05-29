import os

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class ESC50Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        meta_csv,
        fold=None,
        exclude=False,
        transform=None,
        target_transform=None,
        precomputed=True,
        precomputed_dir="./data/specs",
        target_classes=None,
    ):
        """
        Args:
            root_dir (str): Path to audio/ directory.
            meta_csv (str): Path to esc50.csv metadata file.
            fold (int or list[int], optional): Limit to specific fold(s).
            transform (callable, optional): Audio preprocessing (e.g., spectrogram).
            target_transform (callable, optional): Label transformation.
            precomputed (bool, optional): Load precomputed spectrograms.
            precomputed_dir (str, optional): Directory for precomputed spectrograms.
        """
        self.root_dir = root_dir
        self.meta: pd.DataFrame = pd.read_csv(meta_csv)
        self.precomputed = precomputed
        self.precomputed_dir = precomputed_dir

        # Inside __init__ of ESC50Dataset:
        if target_classes is not None:
            self.meta = self.meta[self.meta["target"].isin(target_classes)] # type: ignore

        if fold is not None:
            if isinstance(fold, int):
                fold = [fold]
            if exclude is True:
                self.meta = self.meta[~self.meta["fold"].isin(fold)] # type: ignore
            else:
                self.meta = self.meta[self.meta["fold"].isin(fold)] # type: ignore

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        if self.precomputed:
            spec_path = os.path.join(self.precomputed_dir, row["filename"].replace(".wav", ".pt"))
            data = torch.load(spec_path)
        else:
            wav_path = os.path.join(self.root_dir, row["filename"])
            data, sample_rate = torchaudio.load(wav_path)
            if self.transform:
                data = self.transform(data)
        label = row["target"]
        if self.target_transform:
            label = self.target_transform(label)
        return {"data": data, "label": label, "filename": row["filename"]}

    def get_by_filename(self, filename):
        """Fetch a sample by its filename."""
        match = self.meta[self.meta["filename"] == filename]
        if match.empty:
            raise ValueError(f"Filename {filename} not found in dataset.")
        idx = match.index[0]
        return self[idx]
