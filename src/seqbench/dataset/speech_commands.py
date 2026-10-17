# SPDX-License-Identifier: MIT AND BSD-3-Clause
# Copyright (c) 2025-present, SeqBench Contributors
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
# SPDX-FileNotice: Modified from the original sparch project version

"""
Dataset loader for Speech Commands dataset.
"""

import os
import logging
from pathlib import Path
from os.path import join
from collections import defaultdict

import torch
try:
    import torchaudio
    from torchaudio.transforms import MFCC
except ImportError:
    torchaudio = None
    MFCC = None

from seqbench.dataset.base import BaseDataset

logger = logging.getLogger(__name__)

SC_labels = [
    'yes',
    'no',
    'up',
    'down',
    'left',
    'right',
    'on',
    'off',
    'stop',
    'go',
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
    'bed',
    'bird',
    'cat',
    'dog',
    'happy',
    'house',
    'marvin',
    'sheila',
    'tree',
    'wow',
    'backward',
    'forward',
    'follow',
    'learn',
    'visual'
]


class SpeechCommands(BaseDataset):
    """
    Dataset class for the original non-spiking Speech Commands (SC)
    dataset. Generated mel-spectrograms use 40 bins by default.

    Arguments
    ---------
    data_folder : str
        Path to folder containing the Speech Commands dataset.
    split : str
        Split of the SC dataset, must be either "training", "validation", or "testing".
    return_raw : bool
        If True, returns raw audio waveforms. If False, returns MFCC features (default).
    """

    def __init__(self,
        data_folder,
        split,
        return_raw=False
    ):
        if torchaudio is None:
            raise ImportError(
                "torchaudio is required for the SpeechCommands dataset. "
                "Install SeqBench with the audio extra: pip install 'seqbench[audio]'."
            )
        if split not in ["training", "validation", "testing"]:
            raise ValueError(f"Invalid split {split}")

        # Get paths to all audio files
        self.data_folder = data_folder
        EXCEPT_FOLDER = "_background_noise_"

        def load_list(filename):
            filepath = join(self.data_folder, filename)
            with open(filepath) as f:
                return [join(self.data_folder, i.strip()) for i in f]

        self.labels = SC_labels

        if split == "training":
            files = sorted(str(p) for p in Path(data_folder).glob("*/*.wav"))
            exclude = load_list("validation_list.txt") + load_list("testing_list.txt")
            exclude = set(exclude)
            self.file_list = []
            class_dict = defaultdict(list)
            i=0
            for w in files:
                if w not in exclude and EXCEPT_FOLDER not in w:
                    self.file_list.append(w)
                    relpath = os.path.relpath(w, self.data_folder)
                    label, _ = os.path.split(relpath)  
                    class_index = self.labels.index(label)
                    class_dict[class_index].append(i)
                    i += 1            
        else:
            self.file_list = load_list(str(split) + "_list.txt")
            class_dict = defaultdict(list)
            for i, w in enumerate(self.file_list):
                relpath = os.path.relpath(w, self.data_folder)
                label, _ = os.path.split(relpath)  
                class_index = self.labels.index(label)
                class_dict[class_index].append(i)

        self.class_dict = dict(class_dict)
        self.return_raw = return_raw

        if not self.return_raw:
            sample_rate = 16000
            n_mfcc = 13
            n_mels = 40
            n_fft = 512
            win_length = int(sample_rate * 0.025)  # 25ms
            hop_length = int(sample_rate * 0.01)   # 10ms

            assert MFCC is not None, "MFCC not loaded"

            self.mfcc = MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "n_mels": n_mels,
                    "hop_length": hop_length,
                    "win_length": win_length,
                    'window_fn': torch.hann_window
                },
            )

    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of audio files in the dataset
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
        
        Returns:
            tuple: (x, y) where:
                - x: Audio data (torch.Tensor) - either raw waveform or MFCC features
                - y: Label index (torch.Tensor) - integer class label
        """
        # Read waveform
        filename = self.file_list[index]
        x, _ = torchaudio.load(filename)

        if not(self.return_raw):
            # Compute MFCC features
            x = self.mfcc(x).squeeze().permute(1,0)

        # Get label
        relpath = os.path.relpath(filename, self.data_folder)
        label, _ = os.path.split(relpath)
        y = torch.tensor(self.labels.index(label))

        return x, y

    def generateBatch(self, batch):
        """
        Generate a batched representation of samples for RNN processing.
        
        This method pads sequences to the same length and returns tensors suitable
        for use with PyTorch RNN layers.
        
        Args:
            batch: List of (x, y) tuples from __getitem__
        
        Returns:
            tuple: (xs, xlens, ys) where:
                - xs: Padded sequence tensor of shape (batch_size, max_len, features)
                - xlens: Tensor of sequence lengths for each sample in the batch
                - ys: Long tensor of class labels
        """
        xs, ys = zip(*batch)
        xlens = torch.tensor([x.shape[0] for x in xs])
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys)

        return xs, xlens, ys
