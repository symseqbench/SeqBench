# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
MFCC (Mel-frequency cepstral coefficients) transform for audio feature extraction.
"""

import torch
import logging

from seqbench.transforms.base import TransformTimeSpec

logger = logging.getLogger(__name__)

try:
    from torchaudio.transforms import MFCC as MFCC_TorchAudio

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    logger.warning("MFCC not loaded. torchaudio not installed?")
    TORCHAUDIO_AVAILABLE = False


class MFCC:
    """
    Compute MFCC features from an input signal.
    """

    def __init__(self, n_mfcc=13, n_mels=40, n_fft=512, sample_rate=16000, win_length=400, hop_length=160, **kwargs):
        """

        Parameters
        ----------
        n_mfcc : int
            Number of MFCC coefficients to compute.
        n_mels : int
            Number of Mel bands to use.
        n_fft : int
            Number of FFT points.
        sample_rate : int
            Audio sample rate in Hz.
        win_length : int
            Window length for the STFT, in audio samples.
        hop_length : int
            Hop length for the STFT, in audio samples. The output time-grid
            interval is ``hop_length / sample_rate`` seconds.
        """
        if not TORCHAUDIO_AVAILABLE:
            raise ImportError(
                "torchaudio is required for MFCC. "
                "Install SeqBench with the audio extra: pip install 'seqbench[audio]'."
            )

        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.melkwargs = {
            "n_fft": n_fft,  # n_mels * 2,
            "n_mels": n_mels,
            "hop_length": hop_length,
            "win_length": win_length,
            "window_fn": torch.hann_window,
        }
        self.melkwargs |= kwargs

        self.mfcc = MFCC_TorchAudio(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs=self.melkwargs)

    def time_spec(self, input_grid):
        return TransformTimeSpec("create", out_dt=self.hop_length / self.sample_rate)

    def __call__(self, x):
        # TODO @Younes : check if this is correct - add comment as to why squeeze and permute are needed
        return self.mfcc(x).squeeze().permute(1, 0)
