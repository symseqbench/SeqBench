# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Mel spectrogram transform for audio feature extraction.
"""

import logging

from seqbench.transforms.base import TransformTimeSpec

logger = logging.getLogger(__name__)

try:
    from torchaudio.transforms import MelSpectrogram

    HAS_TORCHAUDIO = True
except ImportError:
    logger.warning("LogMel not loaded. torchaudio not installed?")
    HAS_TORCHAUDIO = False


class LogMel:
    """
    Compute Log-Mel features from an input signal.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_mels=40,
        n_fft=512,
        win_length=400,
        hop_length=160,
        f_min=0.0,
        f_max=None,
        power=2.0,
        **kwargs,
    ):
        """

        Parameters
        ----------
        sample_rate : int
            Audio sample rate in Hz.
        n_mels : int
            Number of Mel bands to use.
        n_fft : int
            Number of FFT points.
        win_length : int
            Window length for the STFT, in audio samples.
        hop_length : int
            Hop length for the STFT, in audio samples. The output time-grid
            interval is ``hop_length / sample_rate`` seconds.
        f_min : float
            Minimum frequency of the mel scale.
        f_max : float
            Maximum frequency of the mel scale.
        power : float
            Exponent for the magnitude spectrogram.
        kwargs : dict
            Additional keyword arguments for the MelSpectrogram.
        """
        if not HAS_TORCHAUDIO:
            raise ImportError(
                "torchaudio is required for LogMel. "
                "Install SeqBench with the audio extra: pip install 'seqbench[audio]'."
            )

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.melspec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=power,
            center=kwargs.get("center", True),
            norm=kwargs.get("norm", None),
            mel_scale=kwargs.get("mel_scale", "htk"),
        )
        # self.to_db = AmplitudeToDB(stype="power")  # move this to a separate transform!

    def time_spec(self, input_grid):
        return TransformTimeSpec("create", out_dt=self.hop_length / self.sample_rate)

    def __call__(self, x):
        # Torchaudio returns (channel, n_mels, time) or (n_mels, time); SeqBench
        # uses time on axis 0.
        x = self.melspec(x).squeeze()
        return x.permute(1, 0)
        # used to be:
        # x = self.melspec(x).squeeze()
        # x = self.to_db(x)
        # x = x.permute(1, 0)
