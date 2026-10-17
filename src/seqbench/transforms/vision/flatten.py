# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Flatten transform for reshaping vision data tensors.
"""

from seqbench.transforms.base import TransformTimeSpec


class Flatten:
    def __init__(self, start_dim):
        self.start_dim = start_dim

    def time_spec(self, input_grid):
        return TransformTimeSpec("preserve")

    def __call__(self, img):
        return img.flatten(start_dim=self.start_dim)
