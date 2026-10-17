# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Tonic Dataset Wrapper Module

This module provides wrappers for various datasets from the tonic library,
making them compatible with SeqBench's BaseDataset interface.
"""

from typing import Dict
import importlib
import inspect

from seqbench.dataset.base import BaseDataset

# Check if tonic is available
try:
    import tonic
    TONIC_AVAILABLE = True
except ImportError:
    TONIC_AVAILABLE = False
    print("Warning: Tonic library not available. Tonic datasets cannot be used.")


class TonicDatasetWrapper(BaseDataset):
    """
    Generic wrapper for tonic datasets that makes them compatible with BaseDataset.
    
    This wrapper allows any tonic dataset to be used with SeqBench by:
    - Extracting class indexes automatically
    - Providing a consistent interface
    - Handling data format conversions if needed
    """
    
    def __init__(
        self,
        dataset_name: str,
        root: str,
        split: str,
        download: bool = False,
        **kwargs
    ):
        """
        Initialize a tonic dataset wrapper.
        
        Args:
            dataset_name: Name of the tonic dataset to import (e.g., "DVSGesture", "NMNIST")
            root: Root directory where dataset is stored
            split: Dataset split ('train' or 'test')
            download: Whether to download the dataset if not present
            **kwargs: Additional arguments to pass to the tonic dataset
        """
        if not TONIC_AVAILABLE:
            raise ImportError(
                "tonic is required for tonic datasets. "
                "Install SeqBench with the tonic extra: pip install 'seqbench[tonic]'."
            )
        
        self.dataset_name = dataset_name
        
        # Dynamically import only the specified dataset
        try:
            tonic_datasets = importlib.import_module("tonic.datasets")
            tonic_dataset_class = getattr(tonic_datasets, dataset_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import tonic dataset '{dataset_name}'. "
                f"Make sure the dataset name is correct and tonic is properly installed."
            ) from e
        
        # Convert split to tonic's expected format
        train = (split == "train" or split == "training")
        
        # Inspect the dataset class signature to only pass supported parameters
        sig = inspect.signature(tonic_dataset_class.__init__)
        valid_params = set(sig.parameters.keys())
        
        # Build kwargs with only valid parameters
        init_kwargs = {}
        
        # Standard parameters that most tonic datasets accept
        if 'save_to' in valid_params:
            init_kwargs['save_to'] = root
        elif 'root' in valid_params:
            init_kwargs['root'] = root
        
        if 'train' in valid_params:
            init_kwargs['train'] = train
        
        if 'download' in valid_params:
            init_kwargs['download'] = download
        
        # Add any additional kwargs that are valid
        for key, value in kwargs.items():
            if key in valid_params:
                init_kwargs[key] = value
        
        # Initialize the tonic dataset
        self.tonic_dataset = tonic_dataset_class(**init_kwargs)

        #TODO Barna: we need to move the following to transforms
        # Get sensor_size from the dataset class
        self.sensor_size = tonic_dataset_class.sensor_size
         
        # Create frame transform
        self.frame_transform = tonic.transforms.ToFrame(
            sensor_size=self.sensor_size,
            time_window= 10 * 1000.0 # 10ms
        )
        
        # Initialize BaseDataset with appropriate parameters
        # Use dataset name as inp_enc identifier
        super().__init__(
            path=root,
            inp_enc=dataset_name.lower(),
            split=split
        )
    
    def __getitem__(self, index: int):
        """Get a sample from the dataset."""
        data, target = self.tonic_dataset[index]

        return data, target
    
    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.tonic_dataset)


# Registry of available tonic datasets (for validation and factory function)
TONIC_DATASET_REGISTRY: Dict[str, str] = {
    "dvsgesture": "DVSGesture",
    "asldvs": "ASLDVS",
    "cifar10dvs": "CIFAR10DVS",
    "ncaltech101": "NCALTECH101",
    "nmnist": "NMNIST",
    "pokerdvs": "POKERDVS",
    "smnist": "SMNIST",
    "dvslip": "DVSLip",
    "shd": "SHD",
    "ssc": "SSC",
    "ntidigits18": "NTIDIGITS18",
}


def create_tonic_dataset(dataset_name: str, root: str, split: str, **kwargs) -> TonicDatasetWrapper:
    """
    Factory function to create a tonic dataset by name.
    
    Args:
        dataset_name: Name of the dataset (case-insensitive, e.g., "dvsgesture", "nmnist")
        root: Root directory where dataset is stored
        split: Dataset split ('train' or 'test')
        **kwargs: Additional arguments to pass to the dataset
        
    Returns:
        TonicDatasetWrapper instance
        
    Raises:
        ValueError: If dataset_name is not in the registry
    """
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower not in TONIC_DATASET_REGISTRY:
        available = ", ".join(TONIC_DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown tonic dataset: {dataset_name}. "
            f"Available datasets: {available}"
        )
    
    # Get the canonical dataset name from registry
    canonical_name = TONIC_DATASET_REGISTRY[dataset_name_lower]
    return TonicDatasetWrapper(
        dataset_name=canonical_name,
        root=root,
        split=split,
        **kwargs
    )
