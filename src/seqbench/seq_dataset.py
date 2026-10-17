# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Sequence Dataset Module for SeqBench

This module provides PyTorch dataset classes and utilities for handling sequential
data in the SeqBench framework. It includes support for various sequence embeddings,
temporal processing, and various sequence tasks.
"""

# Standard library imports
import os
import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Any

# Third-party imports
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import tqdm

# Local imports
from seqbench.seq_utils.generator import SequenceGenerator
from seqbench.seq_utils.symbol_encoder import SymbolEncoder
from seqbench.dataset_generator import DatasetGenerator, RestrictedTargetProbGenerator
from seqbench.tasks.classify import Classification, StateClassification
from seqbench.tasks.target_builder import TaskTargetBuilder
from seqbench.transforms.base import TimeGrid

# Configure logging
logger = logging.getLogger(__name__)

_AUTO = object()


@dataclass
class _GapProfile:
    """Normalised gap profile used internally by SeqDataset.

    Produced from either a :class:`~seqbench.config.GapProfileCfg` (new path)
    or the legacy Config dict (old path) so that _add_gap_content does not
    need to know which config system it came from.
    """

    start: int = 0
    gap_prob: float = 1.0
    duration: Any = None  # dict {dist: callable, params: dict} or scalar/None
    add_nongramm_gap: bool = False
    noise: Optional[float] = None


@dataclass
class Sample:
    """
    Data structure representing a single sequence sample.

    Attributes:
        class_seq: Sequence of class indices (int array)
        state_seq: Sequence of state symbols (str array)
        target_seq: Target sequence for prediction (int array)
        target_probs: Target probabilities for each timestep (float array)
        length: Length of the sequence (int)
    """

    class_seq: np.ndarray  # int
    state_seq: np.ndarray  # str
    target_seq: np.ndarray  # int
    target_probs: np.ndarray  # float
    length: int


class SeqDataset(Dataset):
    """
    PyTorch Dataset for sequential data processing in SeqBench.

    This dataset class handles various sequence encodings, temporal processing,
    and classification tasks. It supports both training and evaluation modes
    with configurable padding, transformations, and data loading strategies.

    Key Features:
    - Support for multiple input encodings (one-hot, SHD, SSC, etc.)
    - Temporal sequence processing with configurable timesteps
    - Classification and regression target generation
    - Configurable padding and data augmentation
    - Efficient data loading with optional caching

    Attributes:
        base_dataset: Base dataset providing raw sequence data
        is_train: Whether this is a training dataset
        timestep: Temporal resolution for sequence processing
        pad_index: Index used for sequence padding
        dataset_size: Total number of samples in the dataset
    """

    def __init__(
        self,
        config,
        split: Optional[str] = None,
        *,
        generator: Any = None,
        base_dataset: Any = None,
        is_train: Optional[bool] = None,
        dataset_size: Optional[int] = None,
        config_file_path: Optional[str] = None,
        config_snapshot: Optional[dict] = None,
        pad_index: int = -1,
        not_temporal: bool = False,
        load_to_tensor: Optional[callable] = None,
        transform: Optional[callable] = _AUTO,
        initial_time_grid: Optional[TimeGrid] = _AUTO,
        dataset_root: Optional[str] = None,
    ) -> None:
        """
        Initialize the SeqDataset.

        Args:
            config: :class:`~seqbench.config.RunConfig`.
            split: Dataset split name, for example ``"train"`` or ``"test"``.
            generator: Optional sequence generator / trial source override.
            base_dataset: Optional base dataset override.
            is_train: Legacy split boolean. If provided without ``split``,
                ``True`` maps to ``"train"`` and ``False`` maps to ``"test"``.
            dataset_size: Optional split-size override. Defaults to
                ``config.seqbench.split_size(split)``.
            config_file_path: Optional original YAML path copied into generated
                dataset metadata.
            config_snapshot: Optional parsed config snapshot for metadata.
            pad_index: Index to use for padding sequences (default: -1).
            not_temporal: If True, treat data as non-temporal (default: False).
            load_to_tensor: Optional function to convert data to tensors.
            transform: Optional transform override. Omit to build from config.
            initial_time_grid: Optional native time grid override. Omit to
                derive from config.
            dataset_root: Optional generated sequence cache directory override.
        """
        split = self._resolve_split(split, is_train)
        self._init_from_run_cfg(
            config,
            split,
            dataset_size,
            config_file_path,
            config_snapshot,
        )
        base_dataset_provided = base_dataset is not None

        if generator is None:
            if config.symseq is None:
                raise ValueError(
                    "SeqDataset requires a trial source. Provide `generator=` or "
                    "configure a top-level `symseq` section."
                )
            from seqbench.sources import build_symseq_source

            generator = build_symseq_source(config)

        if transform is _AUTO:
            from seqbench.transforms import compose_transforms_from_config

            transform = compose_transforms_from_config(config.seqbench.input_mapping)

        if initial_time_grid is _AUTO:
            from seqbench.dataset import initial_time_grid_from_config

            try:
                initial_time_grid = initial_time_grid_from_config(
                    config.seqbench.input_mapping,
                    final_dt=config.seqbench.time_grid.dt,
                )
            except KeyError:
                if not base_dataset_provided:
                    raise
                initial_time_grid = None

        if base_dataset is None:
            from seqbench.dataset import create_base_dataset_from_config

            kwargs = {}
            if config.seqbench.input_mapping.base == "one_hot":
                kwargs["alphabet_size"] = len(generator.alphabet)
            base_dataset = create_base_dataset_from_config(
                config.seqbench.input_mapping,
                split,
                final_dt=config.seqbench.time_grid.dt,
                **kwargs,
            )

        if dataset_root is None:
            from seqbench.utils import dataset_cache_dir

            dataset_root = dataset_cache_dir(config, split, self.dataset_size)

        self.base_dataset = base_dataset
        self.split = split
        self.is_train = split == "train"
        self.transform = transform
        self.initial_time_grid = initial_time_grid
        self.generator = generator
        self.dataset_root = dataset_root
        self.pad_index = pad_index
        self.not_temporal = not_temporal  # TODO potentially remove and set dynamically

        if load_to_tensor:
            self.load_to_tensor = load_to_tensor
        else:
            self.load_to_tensor = lambda x: x

        if self._seed is None:
            self.rng = np.random.default_rng()
            logger.warning("Results will not be reproducible!")
        else:
            self.rng = np.random.default_rng(seed=self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self._seed)

        expected_grid = TimeGrid(dt=self._final_dt)
        self._time_grid = self._resolve_time_grid(initial_time_grid, expected_grid)
        self._dt = self._time_grid.dt

        base_sample = self.load_to_tensor(self.base_dataset[0][0])
        if self.not_temporal:
            base_sample = base_sample.unsqueeze(0)
        if self.transform:
            base_sample = self.transform(base_sample)
        self.base_shape = self.__get_dynamic_samples_dimensions(base_sample)

        # We create the target_prob_generator even for classification
        # to get num_classes (unreduced-state count) from the transitions
        if self.prob_generator_type == "restricted":
            self.target_prob_generator = RestrictedTargetProbGenerator()
        else:
            raise ValueError(
                f"Did not recognize target_prob_generator {self.prob_generator_type}!"
            )

        # The task module is the single source of truth for the training target.
        # state_id_fn is a bound method: the transition map is populated later
        # (read from generator/file) and resolved lazily at sample-build time.
        self.target_builder = TaskTargetBuilder.from_run_cfg(
            self._run_cfg,
            encoder=SymbolEncoder(self.generator.alphabet),
            pad_index=self.pad_index,
            state_id_fn=self.target_prob_generator.unred_state_to_id,
        )
        self._derive_output_structure()

        if self.read_from_file:
            if self.__dataset_folder_exists() and not self.regenerate:
                self.__prepare_dataset_from_file()
            else:
                print(
                    f"SeqBench: Did not find dataset in {self.dataset_root}. Creating it first!"
                )
                self.__create_dataset()
                print(f"SeqBench: Created new dataset in {self.dataset_root}!")
                self.__prepare_dataset_from_file()
        else:
            print("SeqBench: Generating dataset on the fly!")
            self.gs = self.__create_sequence_generator()
            if hasattr(self.generator, "transitions"):
                self.target_prob_generator.read_transitions_from_generator(self.gs)

    # ------------------------------------------------------------------
    # Private initialisation helpers
    # ------------------------------------------------------------------

    def _resolve_split(self, split, is_train):
        if split is None:
            return "train" if is_train is not False else "test"
        if is_train is not None:
            expected = "train" if is_train else "test"
            if split != expected:
                raise ValueError(
                    f"split={split!r} conflicts with is_train={is_train!r}"
                )
        return split

    def _init_from_run_cfg(
        self,
        run_cfg,
        split,
        dataset_size,
        config_file_path,
        config_snapshot,
    ):
        """Extract all config-derived values from a :class:`~seqbench.config.RunConfig`."""
        from seqbench.utils import build_cache_manifest, parse_gap_duration, to_plain_data

        self._run_cfg = run_cfg
        self.split = split
        seqbench_seed = run_cfg.seqbench.seed if run_cfg.seqbench is not None else None
        self._seed = seqbench_seed if seqbench_seed is not None else run_cfg.run.seed
        self.dataset_size = (
            dataset_size if dataset_size is not None else run_cfg.seqbench.split_size(split)
        )
        self._final_dt = run_cfg.seqbench.time_grid.dt
        self._time_validation = run_cfg.seqbench.time_grid.validation
        self.read_from_file = (
            run_cfg.seqbench.mode == "file"
        )  # TODO self.config['read_from_file']
        self.prob_generator_type = run_cfg.seqbench.prob_generator_type
        self.regenerate = run_cfg.seqbench.storage.force_rebuild
        self._config_file_path = config_file_path
        self._config_snapshot = (
            config_snapshot if config_snapshot is not None else to_plain_data(run_cfg)
        )
        self._cache_manifest = build_cache_manifest(
            run_cfg,
            split,
            self.dataset_size,
            source_config=self._config_snapshot,
            source_config_path=config_file_path,
        )
        self._input_mapping_base = run_cfg.seqbench.input_mapping.base
        self._global_noise = 0

        # Sequence generator params
        self._combine_sequences = run_cfg.seqbench.composition.combine_sequences
        self._combined_seq_len = run_cfg.seqbench.composition.sample_length
        from seqbench.config import resolve_trial_params

        self._trial_params = resolve_trial_params(run_cfg.symseq)

        # Normalise gap profile
        gp_cfg = run_cfg.seqbench.composition.gap_profile
        if gp_cfg is not None:
            duration = dict(gp_cfg.duration)
            if "dist" in duration and isinstance(duration["dist"], str):
                duration["dist"] = parse_gap_duration(duration["dist"])
            self._gap_profile = _GapProfile(
                start=gp_cfg.start,
                gap_prob=1.0,  # not in new schema; always insert gaps
                duration=duration,
                add_nongramm_gap=gp_cfg.add_nongramm_gap,
                noise=None,  # not in new schema
            )
        else:
            self._gap_profile = None

    def _resolve_time_grid(self, initial_time_grid, expected_grid):
        if self.transform is not None:
            return self.transform.resolve_time_grid(
                initial_grid=initial_time_grid,
                expected_final_grid=expected_grid,
                validation=self._time_validation,
            )

        resolved = initial_time_grid or expected_grid
        self._validate_dt(
            "base dataset",
            resolved.dt,
            expected_grid.dt,
            self._time_validation,
        )
        return resolved

    def _validate_dt(self, name, actual, expected, validation):
        if abs(actual - expected) <= 1e-9:
            return
        msg = f"{name} time-grid mismatch: got dt={actual:g}s, expected dt={expected:g}s"
        if validation == "error":
            raise ValueError(msg)
        if validation == "warn":
            warnings.warn(msg, stacklevel=3)

    def _derive_output_structure(self):
        """Decide the output path from the configured task.

        - ``is_per_trial``: one label per sample (whole-sequence Classification).
        - ``_wants_target_probs``: emit the grammar transition distribution —
          only for a per_token, prediction-like task over a grammar source
          (not StateClassification, not per_trial).
        - ``per_token_classify``: the no-target_probs per-token path (derived
          property — True when not per_trial and not _wants_target_probs).
        """
        # With combine_sequences a per_trial task's labels are spread to token
        # boundaries by __concat_targets, so the effective output is per_token.
        self.is_per_trial = (
            self.target_builder.granularity == "per_trial"
            and not self._combine_sequences
        )
        self._wants_target_probs = (
            not self.is_per_trial
            and self.target_builder.granularity == "per_token"
            and hasattr(self.generator, "transitions")
            and not isinstance(self.target_builder.task, (Classification, StateClassification))
        )

    @property
    def per_token_classify(self) -> bool:
        """True for a per-token classification path (no target_probs emitted).

        This is the per-token counterpart to ``is_per_trial``: covers
        StateClassification and any per-token task over a non-grammar source.
        Use instead of the removed ``do_classify`` field.
        """
        return not self.is_per_trial and not self._wants_target_probs

    @property
    def is_classification(self):
        """True when the output is a classification target (per_trial label or
        per-token state ids), i.e. no target_probs. Used to pick the collate."""
        return self.is_per_trial or self.per_token_classify

    @property
    def config(self):
        """Resolved :class:`~seqbench.config.RunConfig` used by this dataset."""
        return self._run_cfg

    @property
    def input_mapping(self):
        """Configured input mapping used to build base data and transforms."""
        return self._run_cfg.seqbench.input_mapping

    @property
    def task_config(self):
        """Configured SeqBench task definition."""
        return self._run_cfg.seqbench.task

    @property
    def time_grid(self):
        """Resolved output time grid after base data and transforms."""
        return self._time_grid

    @property
    def dt(self):
        """Seconds per row in the resolved output time grid."""
        return self._dt

    @property
    def returns_target_probs(self) -> bool:
        """True when ``__getitem__`` includes target probability tensors."""
        return self._wants_target_probs

    @property
    def output_kind(self) -> str:
        """Dataset output mode: prediction, per-token, or per-trial labels."""
        if self.is_per_trial:
            return "per_trial_classification"
        if self.per_token_classify:
            return "per_token_classification"
        return "prediction"

    # ------------------------------------------------------------------
    # Private dataset methods
    # ------------------------------------------------------------------

    def __dataset_folder_exists(self):
        """
        Check if the dataset folder exists and contains required files.

        Returns:
            bool: True if dataset folder exists with all required files, False otherwise.
        """
        if self.dataset_root is None:
            return False
        if not os.path.isdir(self.dataset_root):
            return False
        if not os.path.isfile(os.path.join(self.dataset_root, "manifest.yaml")):
            return False
        if not os.path.isfile(os.path.join(self.dataset_root, "config.yaml")):
            return False
        if hasattr(self.generator, "transitions") and not os.path.isfile(
            os.path.join(self.dataset_root, "transitions")
        ):
            return False
        if not os.path.isfile(os.path.join(self.dataset_root, self.split)):
            return False
        return True

    def __create_dataset(self):
        """
        Create a new dataset by generating sequences and saving them to disk.

        This method creates a DatasetGenerator and generates the dataset files
        (train or test) based on the configuration.
        """
        print(f"SeqBench: Generating {self.split!r} dataset in {self.dataset_root}!")
        self.gs = self.__create_sequence_generator()
        # Populate the transition map before generation so tasks that need it at
        # draw time (e.g. StateClassification) resolve correctly and serialize.
        if hasattr(self.generator, "transitions"):
            self.target_prob_generator.read_transitions_from_generator(self.gs)
        dataset_generator = DatasetGenerator(
            self.gs,
            dataset_size=self.dataset_size,
            output_dir=self.dataset_root,
            config_file_path=self._config_file_path,
            config_snapshot=self._config_snapshot,
            manifest=self._cache_manifest,
            split=self.split,
        )
        dataset_generator.generate()

    def __prepare_dataset_from_file(self):
        """
        Prepare the dataset by loading samples from pre-generated files.

        This method reads transition probabilities from file and buffers all
        samples into memory for fast access.
        """
        print(f"SeqBench: Reading dataset from {self.dataset_root}!")
        if hasattr(self.generator, "transitions"):
            self.target_prob_generator.read_transitions_from_file(
                os.path.join(self.dataset_root, "transitions")
            )
        self.buffered_samples = self.__buffer_samples_from_file()

    def __create_sequence_generator(self):
        """
        Create a sequence generator from the instance configuration values.

        Returns:
            SequenceGenerator: Configured sequence generator instance.
        """
        return SequenceGenerator(
            self.generator,
            combine_sequences=self._combine_sequences,
            combined_seq_len=self._combined_seq_len,
            seed=self._seed,
            trial_params=self._trial_params,
            task_builder=self.target_builder,
        )

    @property
    def num_classes(self):
        """Number of distinct classes in the configured task's target label space.

        Task-driven (see :meth:`TaskTargetBuilder.num_classes`): the encoder vocab
        size for class-id targets (prediction / memory / symbolic classification),
        the grammar's unreduced-state count for StateClassification, or the base
        dataset's class count for base-label classification.
        """
        return self.target_builder.num_classes(
            prob_generator=self.target_prob_generator,
            base_dataset=self.base_dataset,
        )

    def __get_samples_dimensions(self):
        """
        Get the dimensions of samples from the base dataset.

        Returns:
            tuple: Sample dimensions (excluding temporal dimension if not_temporal is False).
        """
        data = self.base_dataset[0][0]
        shape = data.shape
        if self.not_temporal:
            return shape
        else:
            return shape[1:]

    def __get_dynamic_samples_dimensions(self, data):
        """
        Get the dimensions of a data sample dynamically.

        This method is used when sample dimensions may change after transformations.

        Args:
            data: Data tensor to get dimensions from.

        Returns:
            tuple: Sample dimensions (excluding temporal dimension if not_temporal is False).
        """
        shape = data.shape
        if self.not_temporal:
            return shape
        else:
            return shape[1:]

    def _add_gap_content(self, seq_element, state_idx, reshape):
        """
        Add gap content between sequence elements based on gap_profile configuration.

        Args:
            seq_element: Current position in the sequence.
            state_idx: Current state index.
            reshape: Function to reshape data samples.

        Returns:
            List of tensors representing the gap items.
        """
        gap_items = []
        gp = self._gap_profile

        if seq_element < gp.start:
            delay = 0
        else:
            if self.rng.random() > gp.gap_prob:
                delay = 0
            elif isinstance(gp.duration, dict) and "dist" in gp.duration:
                delay = float(gp.duration["dist"](**gp.duration["params"]))
            else:
                delay = float(gp.duration)

        # Single conversion shared by both gap modes: real-time seconds -> rows
        # on the resolved final grid used by the returned sample.
        gap_steps = round(delay / self._dt)

        if gp.add_nongramm_gap:
            preds_idx = np.where(
                self.target_prob_generator.transition_probs[state_idx] == 0
            )[0]
            if gap_steps > 0 and len(preds_idx) != 0:
                # Fill exactly gap_steps rows with non-grammatical stimuli, using
                # their actual emitted lengths and truncating the last one so the
                # filler occupies the same footprint as a zero gap of equal delay.
                accumulated = 0
                while accumulated < gap_steps:
                    idx = self.rng.choice(preds_idx)
                    if self._input_mapping_base == "one_hot":
                        delay_idx = self.base_dataset.class_dict[idx]
                    else:
                        delay_idx = int(
                            self.rng.choice(self.base_dataset.class_dict[idx])
                        )

                    delay_content = reshape(self.base_dataset[delay_idx][0])
                    if self.transform:
                        delay_content = self.transform(delay_content)
                    remaining = gap_steps - accumulated
                    if delay_content.shape[0] > remaining:
                        delay_content = delay_content[:remaining]
                    if delay_content.shape[0] == 0:
                        # Degenerate (empty) stimulus: pad the remainder with
                        # zeros rather than loop forever.
                        gap_items.append(
                            torch.zeros([remaining] + list(self.base_shape))
                        )
                        break
                    gap_items.append(delay_content)
                    accumulated += delay_content.shape[0]
            else:
                # No non-grammatical predecessors (or zero gap): fall back to a
                # zero gap of the same real-time footprint.
                gap_items.append(torch.zeros([gap_steps] + list(self.base_shape)))
        else:
            gap_items.append(torch.zeros([gap_steps] + list(self.base_shape)))

        if gp.noise is not None and not gp.add_nongramm_gap:
            for i, gap_item in enumerate(gap_items):
                gap_item = gap_item.float()
                rand_noise = torch.rand_like(gap_item) * self._global_noise
                gap_items[i] = gap_item + rand_noise

        return gap_items

    def __buffer_samples_from_file(self):
        """
        Load and buffer all samples from pre-generated dataset files.

        Returns:
            list: List of Sample objects loaded from the dataset file.
        """
        data_path = os.path.join(self.dataset_root, self.split)

        buffered_samples = []
        pbar = tqdm.tqdm(total=self.dataset_size, desc="Buffering samples")
        with open(data_path, "r") as file:
            for line in file:
                line = line.replace("\n", "").strip()
                gensample = DatasetGenerator.create_gensample_from_str(line)
                sample = self.gensample_to_sample(gensample)
                buffered_samples.append(sample)
                pbar.update(1)
        pbar.close()
        return buffered_samples

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Total number of samples in the dataset.
        """
        return self.dataset_size

    def __getitem__(self, idx):
        """Build one dataset item from sample ``idx``.

        The configured task owns the target *values* (``sample.target_seq``,
        produced by :class:`~seqbench.tasks.target_builder.TaskTargetBuilder`).
        This method handles the embedding assembly, optional gap insertion, the
        per-element -> per-timestep target broadcast
        (:meth:`_expand_per_token_target`), and gap-mask construction.

        The returned item depends on the task-driven output flags:

        - per_trial (``is_per_trial``) -> ``[data, label, class_seq, gap_mask]``
          via :meth:`__build_per_trial_item`. ``label`` is a single int per
          sample; ``gap_mask`` is 2-D (one row per element, padded).
        - per_token classify (``per_token_classify``) ->
          ``[data, target, class_seq, gap_mask]``. ``class_seq`` is
          ``sample.class_seq``; ``gap_mask`` is 2-D.
        - predict (``_wants_target_probs``) ->
          ``[data, target, class_seq, target_probs, gap_mask]``. ``class_seq`` is
          the unreduced-state ids (debug only); ``target_probs`` is (T, C) grammar
          transition distributions; ``gap_mask`` is 1-D.

        Shapes: ``data`` (T, J); ``target`` (T,); ``target_probs`` (T, C) where C
        is the unreduced-state count. Masked target positions hold ``pad_index``.
        ``class_seq`` is only provided for debugging.
        """
        assert not (torch.is_tensor(idx)), "idx needs to be an integer"

        sample = self.__get_sample(idx)

        if self.not_temporal:
            reshape = lambda x: self.load_to_tensor(x).unsqueeze(0)
        else:
            reshape = lambda x: self.load_to_tensor(x)

        if self.is_per_trial:
            return self.__build_per_trial_item(sample, reshape)

        if self.per_token_classify:
            target_probs_or_placeholder = sample.class_seq
        else:
            target_probs_or_placeholder = sample.target_probs

        data = []
        durations = []
        gap_mask = []
        target_probs = []
        timestamp = 0
        seq_element = 0

        # The per-element target values (sample.target_seq) are consumed after
        # the loop by _expand_per_token_target; here we only accumulate each
        # element's timestep duration so the broadcast can be done in one pass.
        for data_idx, state_idx, target_prob in zip(
            sample.class_seq,
            sample.state_seq,
            target_probs_or_placeholder,
        ):

            if self._input_mapping_base == "one_hot":
                data_idx = self.base_dataset.class_dict[data_idx]
            else:
                data_idx = int(
                    self.rng.choice(self.base_dataset.class_dict[data_idx])
                )  # 1 2 3 0 -> 283 12002 1233 1737

            cur_data_sample = self.base_dataset[data_idx][0]
            cur_data_sample = reshape(cur_data_sample)

            # ---> apply any transforms here
            # TODO BZ: add noise transforms to sample!! 
            if self.transform:
                # cur_data_sample_tmp = self.transform(cur_data_sample)
                # print("Sample shape before transform:", cur_data_sample.shape)
                # print("Sample shape after transform:", cur_data_sample_tmp.shape)
                # cur_data_sample = cur_data_sample_tmp
                cur_data_sample = self.transform(cur_data_sample)
                self.base_shape = self.__get_dynamic_samples_dimensions(cur_data_sample)

            data.append(cur_data_sample)

            if self._gap_profile is not None:
                gap_items = self._add_gap_content(seq_element, state_idx, reshape)
                data.extend(gap_items)
                delay_dur = (
                    sum(item.shape[0] for item in gap_items)
                    if len(gap_items) > 0
                    else 0
                )
            else:
                delay_dur = 0

            # ================================================================================

            data_t = cur_data_sample.shape[0]
            durations.append(data_t + delay_dur)

            if not self.per_token_classify:
                target_prob = torch.tensor(target_prob).unsqueeze(0)
                target_probs.append(target_prob.repeat(data_t + delay_dur, 1))

            if self.per_token_classify:
                gap_mask.append(
                    torch.cat(
                        [torch.zeros(timestamp + data_t - 1), torch.ones(delay_dur + 1)]
                    )
                )
            else:
                gap_mask.append(torch.zeros(data_t - 1))
                gap_mask.append(torch.ones(delay_dur + 1))

            timestamp = timestamp + data_t + delay_dur
            seq_element += 1

        data = torch.cat(data, dim=0)
        target = self._expand_per_token_target(sample.target_seq, durations)

        if not self.per_token_classify:
            target_probs = torch.cat(target_probs, dim=0)

        assert data.shape[0] == target.shape[0]

        if self.per_token_classify:
            gap_mask = torch.nn.utils.rnn.pad_sequence(gap_mask, batch_first=True)
        else:
            gap_mask = torch.cat(gap_mask, dim=0)

        if self.per_token_classify:
            assert data.shape[0] == target.shape[0]
            sample = [data, target, sample.class_seq, gap_mask]
        else:
            assert data.shape[0] == target.shape[0] == target_probs.shape[0]
            class_seq = [
                self.target_prob_generator.unred_state_to_id(s)
                for s in sample.state_seq
            ]
            sample = [data, target, class_seq, target_probs, gap_mask]
        return sample

    def _expand_per_token_target(self, per_element_values, durations):
        """Broadcast a per-element target to per-timestep, held constant per element.

        The task module owns *what* the target is (``per_element_values`` is
        ``sample.target_seq``: a class id per sequence element, or ``pad_index``
        for a masked position). The dataset owns *how long* each element lasts in
        the temporal embedding (``durations[i] = data_t + delay_dur``). This
        repeats each value across its element's timesteps:
        ``torch.cat([torch.ones(dur) * val for val, dur in ...])``.

        Returns a float tensor; the collate functions cast it to long.

        Args:
            per_element_values: per-element target ints (``sample.target_seq``).
            durations: per-element timestep counts (``data_t + delay_dur``),
                in the same element order as ``per_element_values``.

        # TODO: gap timesteps currently inherit the preceding symbol's target (because
        # data_t + delay_dur are folded into a single duration bucket). To allow
        # explicit control, split each bucket into a symbol portion (data_t, keeps val)
        # and a gap portion (delay_dur, filled with pad_index or a configurable value),
        # and expose a gap_target option in _GapProfile / gap_profile config.
        """
        return torch.cat(
            [
                torch.ones(dur) * val
                for val, dur in zip(per_element_values, durations)
            ]
        )

    def __build_per_trial_item(self, sample, reshape):
        """Assemble a whole-sequence classification item: one label per sample.

        The input is built by the same per-token embedding + gap assembly as the
        per-token path, but the target is a single label (``sample.target_seq``
        is a scalar from a per_trial task). Returns ``[data, label, class_seq,
        gap_mask]`` for :meth:`PadSequence.pad_collate_classify_trial`.
        """
        data = []
        gap_mask = []
        timestamp = 0
        seq_element = 0

        for data_idx, state_idx in zip(sample.class_seq, sample.state_seq):
            if self._input_mapping_base == "one_hot":
                data_idx = self.base_dataset.class_dict[data_idx]
            else:
                data_idx = int(self.rng.choice(self.base_dataset.class_dict[data_idx]))

            cur_data_sample = reshape(self.base_dataset[data_idx][0])
            if self.transform:
                cur_data_sample = self.transform(cur_data_sample)
                self.base_shape = self.__get_dynamic_samples_dimensions(cur_data_sample)
            data.append(cur_data_sample)

            if self._gap_profile is not None:
                gap_items = self._add_gap_content(seq_element, state_idx, reshape)
                data.extend(gap_items)
                delay_dur = sum(item.shape[0] for item in gap_items) if gap_items else 0
            else:
                delay_dur = 0

            data_t = cur_data_sample.shape[0]
            gap_mask.append(
                torch.cat(
                    [torch.zeros(timestamp + data_t - 1), torch.ones(delay_dur + 1)]
                )
            )
            timestamp += data_t + delay_dur
            seq_element += 1

        data = torch.cat(data, dim=0)
        gap_mask = torch.nn.utils.rnn.pad_sequence(gap_mask, batch_first=True)
        label = int(sample.target_seq)
        return [data, label, sample.class_seq, gap_mask]

    def __get_sample(self, idx):
        """
        Get a Sample object for the given index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Sample: Sample object containing class_seq, state_seq, target_seq, etc.
        """
        if self.read_from_file:
            return self.__get_sample_from_buffer(idx)
        else:
            gensample = self.gs.generate(idx, compute_length=False)
            return self.gensample_to_sample(gensample)

    def __get_sample_from_buffer(self, idx):
        """
        Get a sample from the buffered samples.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Sample: Sample object from the buffer.
        """
        return self.buffered_samples[idx]

    def gensample_to_sample(self, gensample):
        """
        Convert a GeneratorSample to a Sample object.

        The training target comes entirely from the configured task (via
        :class:`~seqbench.tasks.target_builder.TaskTargetBuilder`). ``target_probs``
        is an orthogonal grammar-transition channel, emitted only for a
        per_token prediction-like task over a grammar source.

        Args:
            gensample: GeneratorSample object from the sequence generator.

        Returns:
            Sample: Sample object with target sequences and probabilities.
        """
        # For Classification(label_source="base"), the task is deferred: it
        # cannot run at draw time because the base-dataset label is only
        # available here (SeqDataset owns base_dataset). Populate the label
        # from the first class in class_seq before calling to_target_seq().
        task = self.target_builder.task
        if (
            getattr(task, "needs_base_dataset", False)
            and not (gensample.targets and self.target_builder.task_id in gensample.targets)
        ):
            class_idx = int(gensample.class_seq[0])
            candidates = self.base_dataset.class_dict[class_idx]
            # class_dict values are lists (real datasets) or scalars (one_hot)
            rep_idx = int(candidates[0]) if hasattr(candidates, "__len__") else int(candidates)
            gensample.label = int(self.base_dataset[rep_idx][1])

        resolved = self.target_builder.to_target_seq(gensample)
        target_seq = resolved.target_seq
        if self._wants_target_probs:
            target_probs = self.target_prob_generator(gensample.state_seq)
        else:
            target_probs = None
        return Sample(
            gensample.class_seq,
            gensample.state_seq,
            target_seq,
            target_probs,
            gensample.length,
        )


# TODO BZ: should probably use the class constructor directly
def make_pad_sequence(dataset, pad_index=-1, debug_class=True):
    """Build the :class:`PadSequence` collate matching a dataset's output kind.

    Derives the right collate from the (task-driven) ``dataset`` flags so call
    sites never hardcode ``do_classify``.
    """
    return PadSequence(
        do_classify=dataset.per_token_classify,
        pad_index=pad_index,
        debug_class=debug_class,
        per_trial=dataset.is_per_trial,
    )


class PadSequence:
    """
    Collate function for padding sequences in batches.

    This class provides padding functionality for sequence data, supporting both
    classification and prediction tasks. It pads sequences to the same length within
    a batch and returns dictionaries with padded tensors and metadata.

    Attributes:
        pad_index: Index value used for padding sequences (default: -1).
        debug_class: Whether to include debug class sequences in output (default: True).
        pad_collate_fn: The collate function to use (set based on do_classify).

    Example:
        >>> pad_fn = PadSequence(do_classify=True, pad_index=-1)
        >>> batch = pad_fn([sample1, sample2, sample3])
        >>> # Returns dict with 'data', 'labels', 'mask', 'lens', 'gap_mask', etc.
    """

    def __init__(self, do_classify=None, pad_index=-1, debug_class=True, per_trial=False):
        """
        Initialize the PadSequence collate function.

        Args:
            do_classify: If True, use the per-token classification collate
                        (per-token labels, no target_probs). If False, use the
                        prediction collate (includes target_probs). Ignored when
                        ``per_trial`` is True.
            pad_index: Index value to use for padding sequences (default: -1).
            debug_class: Whether to include debug class sequences in output (default: True).
            per_trial: If True, use the whole-sequence classification collate
                        (one label per sample).
        """
        self.pad_index = pad_index
        self.debug_class = debug_class
        if per_trial:
            self.pad_collate_fn = self.pad_collate_classify_trial
        elif do_classify:
            self.pad_collate_fn = self.pad_collate_classify
        else:
            self.pad_collate_fn = self.pad_collate_predict

    def __call__(self, batch):
        """
        Collate a batch of samples by padding sequences.

        Args:
            batch: List of samples from SeqDataset.__getitem__.

        Returns:
            dict: Dictionary containing padded tensors and metadata.
        """
        return self.pad_collate_fn(batch)

    def pad_collate_classify(self, batch):
        """
        Collate function for classification tasks.

        Pads sequences in a batch for classification, including data, labels, masks,
        gap masks, and optionally debug class sequences.

        Args:
            batch: List of tuples (data, target, class_seq, gap_mask) from SeqDataset.

        Returns:
            dict: Dictionary containing:
                - 'data': Padded data tensor (batch_size, max_len, features).
                - 'labels': Padded label tensor (batch_size, max_len).
                - 'mask': Padding mask tensor (batch_size, max_len).
                - 'lens': Sequence lengths tensor (batch_size,).
                - 'gap_mask': Padded gap mask tensor (batch_size, max_elem, max_len).
                - 'debug_class_seq': Padded debug class sequence (if debug_class=True).
        """
        data = []
        target = []
        mask = []
        lens = []
        gap_masks = []
        debug_class_seq = []

        for data_b, target_b, class_seq_b, gap_mask_b in batch:
            l = data_b.shape[0]

            data.append(data_b)
            target.append(target_b)
            mask.append(torch.ones(l))
            lens.append(l)
            gap_masks.append(torch.Tensor(gap_mask_b))
            debug_class_seq.append(class_seq_b)

        max_len = max(tensor.size(1) for tensor in gap_masks)
        max_elem = max(tensor.size(0) for tensor in gap_masks)
        padded_gaps = [
            F.pad(tensor, (0, max_len - tensor.size(1), 0, max_elem - tensor.size(0)))
            for tensor in gap_masks
        ]
        gap_mask = torch.stack(padded_gaps)

        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        target = torch.nn.utils.rnn.pad_sequence(
            target, batch_first=True, padding_value=self.pad_index
        )
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
        lens = torch.as_tensor(lens)

        if self.debug_class:
            debug_class_seq = torch.Tensor(np.array(debug_class_seq))
            debug_class_seq = torch.nn.utils.rnn.pad_sequence(
                debug_class_seq, batch_first=True
            )

            return {
                "data": data.float(),
                "labels": target.long(),
                "mask": mask.float(),
                "lens": lens,
                "gap_mask": gap_mask.contiguous(),
                "debug_class_seq": debug_class_seq,
            }
        else:
            return {
                "data": data.float(),
                "labels": target.long(),
                "mask": mask.float(),
                "lens": lens,
                "gap_mask": gap_mask.contiguous(),
            }

    def pad_collate_classify_trial(self, batch):
        """Collate for whole-sequence classification (one label per sample).

        Args:
            batch: List of ``[data, label, class_seq, gap_mask]`` items, where
                ``label`` is a single int per sample.

        Returns:
            dict with ``data`` (B, T, ...), ``labels`` (B,), ``mask`` (B, T),
            ``lens`` (B,), and padded ``gap_mask``.
        """
        data = []
        labels = []
        mask = []
        lens = []
        gap_masks = []

        for data_b, label_b, class_seq_b, gap_mask_b in batch:
            l = data_b.shape[0]
            data.append(data_b)
            labels.append(int(label_b))
            mask.append(torch.ones(l))
            lens.append(l)
            gap_masks.append(torch.Tensor(gap_mask_b))

        max_len = max(tensor.size(1) for tensor in gap_masks)
        max_elem = max(tensor.size(0) for tensor in gap_masks)
        padded_gaps = [
            F.pad(tensor, (0, max_len - tensor.size(1), 0, max_elem - tensor.size(0)))
            for tensor in gap_masks
        ]
        gap_mask = torch.stack(padded_gaps)

        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
        lens = torch.as_tensor(lens)
        labels = torch.as_tensor(labels).long()

        return {
            "data": data.float(),
            "labels": labels,  # (B,) one label per sample
            "mask": mask.float(),
            "lens": lens,
            "gap_mask": gap_mask.contiguous(),
        }

    def pad_collate_predict(self, batch):
        """
        Collate function for prediction tasks.

        Pads sequences in a batch for prediction, including data, labels, target
        probabilities, masks, gap masks, and debug class sequences.

        Args:
            batch: List of tuples (data, target, class_seq, target_probs, gap_mask)
                from SeqDataset.

        Returns:
            dict: Dictionary containing:
                - 'data': Padded data tensor (batch_size, max_len, features).
                - 'labels': Padded label tensor (batch_size, max_len).
                - 'target_probs': Padded target probabilities (batch_size, max_len, num_classes).
                - 'mask': Padding mask tensor (batch_size, max_len).
                - 'lens': Sequence lengths tensor (batch_size,).
                - 'gap_mask': Padded gap mask tensor (batch_size, max_len).
                - 'debug_class_seq': Padded debug class sequence.
        """
        data = []
        target = []
        target_probs = []
        mask = []
        lens = []
        gap_masks = []
        debug_class_seq = []

        for data_b, target_b, class_seq_b, target_probs_b, gap_mask_b in batch:
            l = data_b.shape[0]

            data.append(data_b)
            target.append(target_b)
            target_probs.append(target_probs_b)
            mask.append(torch.ones(l))
            lens.append(l)
            gap_masks.append(gap_mask_b)
            debug_class_seq.append(class_seq_b)

        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        target = torch.nn.utils.rnn.pad_sequence(
            target, batch_first=True, padding_value=self.pad_index
        )
        target_probs = torch.nn.utils.rnn.pad_sequence(target_probs, batch_first=True)
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
        lens = torch.as_tensor(lens)
        gap_masks = torch.nn.utils.rnn.pad_sequence(gap_masks, batch_first=True)
        debug_class_seq = torch.Tensor(np.array(debug_class_seq))
        debug_class_seq = torch.nn.utils.rnn.pad_sequence(
            debug_class_seq, batch_first=True
        )

        return {
            "data": data.float(),
            "labels": target.long(),
            "target_probs": target_probs.float(),
            "mask": mask.float(),
            "lens": lens,
            "gap_mask": gap_masks,
            "debug_class_seq": debug_class_seq,
        }
