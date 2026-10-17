# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Synthetic Dataset Module for SeqBench

This module provides simple embeddings such as one hot encoding.
"""

import numpy as np
import torch


class OneHot:
    """
    Synthetic one-hot encoded dataset.
    
    This class generates a simple dataset where each sample is a one-hot
    encoded vector. Useful for testing sequence models and data pipelines.
    
    Attributes:
        vocab_size: Number of unique tokens/classes in the vocabulary
        n_steps: Number of timesteps each stimulus occupies on the time grid
        labels: Array of label indices (0 to vocab_size-1)
        stimulus: Dictionary mapping label indices to one-hot vectors
        class_dict: Dictionary mapping class indices to themselves
    """

    def __init__(self, vocab_size, n_steps=1):
        """
        Initialize the one-hot dataset.

        Args:
            vocab_size: Number of unique tokens/classes in the vocabulary
            n_steps: Number of timesteps each stimulus occupies on the time
                grid. The one-hot row is repeated ``n_steps`` times so the
                caller can express a physical stimulus duration on the final
                output grid. Defaults to 1 (a single timestep).
        """

        if n_steps < 1:
            raise ValueError(f"OneHot n_steps must be >= 1, got {n_steps}")
        self.vocab_size = vocab_size
        self.n_steps = n_steps
        self.labels = np.arange(self.vocab_size)
        one_hot = np.eye(self.vocab_size, dtype=int)
        self.stimulus = {k: v for i, (k, v) in enumerate(zip(self.labels, one_hot))}
        x = np.arange(self.vocab_size)
        self.class_dict = {key: key for key in x}

    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: The vocabulary size (number of unique tokens)
        """
        return self.vocab_size

    def __getitem__(self, index):
        """
        Get a one-hot encoded sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve (0 to vocab_size-1)
        
        Returns:
            tuple: (x, y) where:
                - x: One-hot encoded tensor of shape (n_steps, vocab_size)
                - y: Label index (same as input index)
        """

        x = self.stimulus[index]
        x = np.tile(x[None, :], (self.n_steps, 1))

        return torch.from_numpy(x), index
