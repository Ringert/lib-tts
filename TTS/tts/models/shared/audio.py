"""Actual waveform and sample rate at the adapter boundary."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class SynthesisAudio:
    waveform: np.ndarray
    sample_rate: int

    def __post_init__(self):
        if (
            not isinstance(self.sample_rate, Integral)
            or isinstance(self.sample_rate, bool)
            or self.sample_rate <= 0
        ):
            raise ValueError("Invalid synthesis sample rate")
        waveform = np.asarray(self.waveform, dtype=np.float32).squeeze()
        if waveform.ndim == 0:
            waveform = waveform.reshape(1)
        if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
            raise ValueError("Invalid synthesis waveform")
        object.__setattr__(self, "waveform", waveform)
