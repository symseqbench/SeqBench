# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Center crop transform for vision data processing.
"""

from seqbench.transforms.base import TransformTimeSpec


class CenterCrop:
    def __init__(self, sensor_size, output_size):
        self.sensor_size = sensor_size
        self.output_size = output_size

    def time_spec(self, input_grid):
        return TransformTimeSpec("preserve")

    def __call__(self, img):
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)

        crop_h, crop_w = self.output_size

        h, w, _ = self.sensor_size

        top = int(round((h - crop_h) / 2.))
        left = int(round((w - crop_w) / 2.))

        return img[..., top:top + crop_h, left:left + crop_w]
