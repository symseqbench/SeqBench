# SeqBench

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSES/LICENSE)

! **Development Status:** 3 – Alpha: The codebase is still under active development, with potential API modifications.

**SeqBench: *Sequence Benchmarking Framework*** provides an end-to-end workflow for converting symbolic sequences into datasets suitable for training and evaluating sequence learning models. It offers fine control over sequence structure and token representations. Symbols can be associated with images, sounds, vector encodings, or other modalities.

SeqBench integrates with [**SymSeq**](https://github.com/symseqbench/symseq) for sequence generation allowing for full experiment pipelines to be defined through YAML configuration files. A high-level overview of the combined framework and its capabilities can be found in the accompanying [**SymSeqBench paper**](https://arxiv.org/abs/2512.24977).

## Features

- **Flexible mapping** between symbolic sequences and embedded representations
- **Support for dataset-based mapping** of symbols to images, audio, or custom objects
- Support for vector embeddings, including one-hot, random, binary, or custom functions
- **Rich transformation pipeline** for audio, vision, tensor data, and spiking encodings
- **Temporal perturbations** such as gaps, jitter, and timing manipulations
- On-the-fly generation or loading precomputed sequences via SymSeq integration
- Tools to quantify representational and geometric complexity (TODO)
- **Modular design** that supports data augmentation and ablation studies

## Installation

1. Install uv:
```bash
pip install uv
# or
pipx install uv
```

#### Development installation

1. Clone the repository:
```bash
git clone https://github.com/symseqbench/SeqBench.git
cd SeqBench
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate
```

3. Install development dependencies:
```bash
uv pip install -e ".[dev]"
```

4. install symseq (sequence generator)
```bash
git clone https://github.com/symseqbench/symseq.git
cd symseq
uv pip install -e ".[dev]"
cd ..
```

**Note**: Some datasets may require additional dependencies. 
Please refer to the specific dataset documentation for requirements.

## Quick Start

### Creating a Dataset

Create a sequence dataset from a configuration file:

```bash
seqbench validate SeqBench/examples/configs/onehot_raw.yaml
seqbench create SeqBench/examples/configs/onehot_raw.yaml
seqbench inspect-batch SeqBench/examples/configs/onehot_raw.yaml
```

### Visualizing a sample

View a sample from your dataset:

```bash
seqbench show-sample SeqBench/examples/configs/onehot_raw.yaml
```

Or for other datasets:

```bash
seqbench show-sample SeqBench/examples/configs/shd_pre.yaml
```
make sure to download the dataset to directory specified in the config file,
i.e., `seqbench.input_mapping.base_params.base_dataset_path`.

### Example Usage with PyTorch

```python
from seqbench import build_dataloader

dataloader = build_dataloader(
    "SeqBench/examples/configs/onehot_raw.yaml",
    split="train",
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

# Iterate over batches
for batch in dataloader:
    inputs = batch["data"]      # Input sequences
    targets = batch["labels"]   # Target labels
    lengths = batch["lens"]     # Sequence lengths
    # Other keys depend on the task mode and may include:
    # "target_probs", "gap_mask", and "debug_class_seq".
    # Train your model...
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows appropriate coding standards and includes appropriate tests.

## Citation

If you use the ``SeqBench`` library, please cite our [paper](https://arxiv.org/abs/2512.24977).

## License

SeqBench is licensed under the MIT License - see the [LICENSE](LICENSES/LICENSE) file for details.

Some of the dataset files included or referenced by SeqBench are licensed under [third party licenses](LICENSES/third_party).
