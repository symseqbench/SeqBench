# SPDX-License-Identifier: MIT AND BSD-3-Clause
# Copyright (c) 2025-present, SeqBench Contributors
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
# SPDX-FileNotice: Modified from the original sparch project version

"""
Dataset loader for SHD (Spiking Heidelberg Digits) and SSC (Spiking Speech Commands) datasets.
"""

import logging

import torch
import numpy as np

from seqbench.dataset.base import BaseDataset

logger = logging.getLogger(__name__)

try:
    import h5py
except ImportError:
    h5py = None


class SpikingDataset(BaseDataset):
    """
    Dataset class for the Spiking Heidelberg Digits (SHD) or
    Spiking Speech Commands (SSC) dataset.

    Arguments
    ---------
    dataset_name : str
        Name of the dataset, either shd or ssc.
    data_folder : str
        Path to folder containing the dataset (h5py file).
    split : str
        Split of the SHD dataset, must be either "train" or "test".
    nb_steps : int
        Number of time steps for the generated spike trains.
    """

    def __init__(self, dataset_name, data_folder, split, nb_steps=140, max_time=1.4, num_bins=1):
        """
        Initialize the spiking dataset.
        
        Args:
            dataset_name: Name of the dataset, either "shd" or "ssc"
            data_folder: Path to folder containing the dataset (h5py file)
            split: Dataset split, must be either "train" or "test"
            nb_steps: Number of time steps for the generated spike trains (default: 140)
            max_time: Maximum time in seconds (default: 1.4)
            num_bins: Number of bins for spatial binning (default: 1)
        """
        if h5py is None:
            raise ImportError(
                "h5py is required for SHD/SSC datasets. "
                "Install SeqBench with the hdf5 extra: pip install 'seqbench[hdf5]'."
            )
        self.nb_steps = nb_steps
        self.num_bins = num_bins
        self.nb_units = 700
        self.nb_units_binned = self.nb_units // self.num_bins
        self.max_time = max_time
        self.time_bins = np.linspace(0, self.max_time, num=self.nb_steps)

        # Read data from h5py file
        filename = f"{data_folder}/{dataset_name}_{split}.h5"
        self.h5py_file = h5py.File(filename, "r")
        self.firing_times = self.h5py_file["spikes"]["times"]
        self.units_fired = self.h5py_file["spikes"]["units"]
        self.labels = np.array(self.h5py_file["labels"], dtype=int)

        super().__init__(path=data_folder, inp_enc=dataset_name, split=split)

    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Get a sample from the spiking dataset.
        
        This method converts spike times and units into a dense tensor representation
        with temporal binning and optional spatial binning.
        
        Args:
            index: Index of the sample to retrieve
        
        Returns:
            tuple: (x, y) where:
                - x: Dense tensor of shape (T, J//Bin) representing binned spike trains
                - y: Label index (int)
        """
        times = np.digitize(self.firing_times[index], self.time_bins)
        units = self.units_fired[index]
        length = max(times)

        x_idx = torch.LongTensor(np.array([times, units]))
        x_val = torch.FloatTensor(np.ones(len(times)))
        x_size = torch.Size([length, self.nb_units])

        x = torch.sparse.FloatTensor(x_idx, x_val, x_size)
        y = self.labels[index]

        x = x.to_dense()
        T = x.shape[0]
        J = self.nb_units
        Bin = self.num_bins

        # Binning
        with torch.no_grad():
            x = x.contiguous().view(T, J // Bin, Bin).sum(-1)

        return x, y


#    def generateBatch(self, batch):
#        xs, ys = zip(*batch)
#        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
#        xlens = torch.tensor([x.shape[0] for x in xs])
#        ys = torch.LongTensor(ys).to(self.device)
#        return xs, xlens, ys


# def load_shd_or_ssc(
#    dataset_name,
#    data_folder,
#    split,
#    batch_size,
#    nb_steps=100,
#    shuffle=True,
#    workers=0,
# ):
#    """
#    This function creates a dataloader for a given split of
#    the SHD or SSC datasets.
#
#    Arguments
#    ---------
#    dataset_name : str
#        Name of the dataset, either shd or ssc.
#    data_folder : str
#        Path to folder containing the Heidelberg Digits dataset.
#    split : str
#        Split of dataset, must be either "train" or "test" for SHD.
#        For SSC, can be "train", "valid" or "test".
#    batch_size : int
#        Number of examples in a single generated batch.
#    shuffle : bool
#        Whether to shuffle examples or not.
#    workers : int
#        Number of workers.
#    """
#    if dataset_name not in ["shd", "ssc"]:
#        raise ValueError(f"Invalid dataset name {dataset_name}")
#
#    if split not in ["train", "valid", "test"]:
#        raise ValueError(f"Invalid split name {split}")
#
#    if dataset_name == "shd" and split == "valid":
#        logging.info("SHD does not have a validation split. Using test split.")
#        split = "test"
#
#    dataset = SpikingDataset(dataset_name, data_folder, split, nb_steps)
#    logging.info(f"Number of examples in {split} set: {len(dataset)}")
#
#    loader = DataLoader(
#        dataset,
#        batch_size=batch_size,
#        collate_fn=dataset.generateBatch,
#        shuffle=shuffle,
#        num_workers=workers,
#        pin_memory=True,
#    )
#
#    return loader
