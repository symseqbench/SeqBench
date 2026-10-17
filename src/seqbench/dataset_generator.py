# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Dataset Generator Module for SeqBench

This module provides utilities for generating synthetic sequence datasets
with configurable parameters and parallel processing capabilities.

Key Components:
- DatasetGenerator: Main class for generating sequence datasets
- Static utility methods for file I/O operations
"""

# Standard library imports
import os
import json
import yaml
from dataclasses import dataclass
from multiprocessing import Pool, Manager
from typing import Optional, Any, Dict

# Third-party imports
import numpy as np
import tqdm

# Local imports
from seqbench.seq_utils.generator import GeneratorSample
from seqbench.tasks.base import Target


def _json_default(o):
    """JSON encoder fallback for numpy scalars/arrays in Target values."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _targets_to_json(targets: Optional[Dict[str, Target]]) -> str:
    """Serialize a ``{name: Target}`` dict to a JSON string ('' when empty)."""
    if not targets:
        return ""
    obj = {
        name: {
            "values": t.values,
            "mask": t.mask,
            "granularity": t.granularity,
        }
        for name, t in targets.items()
    }
    return json.dumps(obj, default=_json_default)


def _targets_from_json(field: str) -> Optional[Dict[str, Target]]:
    """Parse the serialized targets field back into a ``{name: Target}`` dict."""
    field = field.strip()
    if not field:
        return None
    obj = json.loads(field)
    return {
        name: Target(
            values=d["values"], mask=d["mask"], granularity=d["granularity"]
        )
        for name, d in obj.items()
    }


def _strip_parens(s: str) -> str:
    """Remove parentheses from a state name (e.g. ``"i(1)"`` → ``"i1"``).

    Matches the normalisation applied by
    ``SequenceGenerator.generate_sequence()`` so that state strings in
    ``transition_probs`` / ``unred_state_to_id_map`` are consistent with
    those in ``GeneratorSample.state_seq``.
    """
    return s.replace("(", "").replace(")", "")


@dataclass
class MCState:
    """
    Markov Chain state representation.

    Attributes:
        node: Current state node identifier (str)
        childs: Array of possible child states (str array)
        probs: Array of transition probabilities to child states (float array)
    """

    node: str
    childs: np.ndarray  # str
    probs: np.ndarray  # float


class RestrictedTargetProbGenerator:
    """
    Generator for target probabilities with state restrictions.

    This class manages the generation of target probabilities for sequence
    prediction tasks while maintaining state transition constraints. It supports
    both reduced and unreduced state spaces for efficient probability computation.

    Attributes:
        num_reduced_states: Number of states in reduced state space
        num_unreduced_states: Number of states in unreduced state space
        transitions: State transition matrix
        transition_probs: Transition probability matrix
        red_state_to_id_map: Mapping from reduced states to IDs
        unred_state_to_id_map: Mapping from unreduced states to IDs
        id_to_red_state_map: Mapping from IDs to reduced states
        id_to_unred_state_map: Mapping from IDs to unreduced states
        reduced_states: List of reduced state identifiers
        unreduced_states: List of unreduced state identifiers
    """

    def __init__(self):
        self.num_reduced_states = 0
        self.num_unreduced_states = 0
        self.transitions = None
        self.transition_probs = None
        self.red_state_to_id_map = {}
        self.unred_state_to_id_map = {}
        self.id_to_red_state_map = {}
        self.id_to_unred_state_map = {}
        self.reduced_states = []
        self.unreduced_states = []

    def read_transitions_from_file(self, transitions_filepath):
        print(f"Reading transitions from {transitions_filepath}!")
        with open(transitions_filepath, "r") as file:
            transitions = DatasetGenerator.read_transitions_from_file(file)
        self.__parse(transitions)

    def read_transitions_from_generator(self, generator):
        print(f"Loading transitions from generator!")
        if not hasattr(generator.source, "transitions"):
            raise AttributeError(
                "read_transitions_from_generator requires a grammar source "
                "with a `.transitions` attribute (e.g. ArtificialGrammar)."
            )
        transitions = {}
        for transition in generator.source.transitions:
            s_from = transition[0]
            s_to = transition[1]
            prob = transition[2]

            if s_from not in transitions:
                transitions[s_from] = MCState(s_from, np.array([s_to]), np.array([prob]))
            else:
                transitions[s_from].childs = np.append(transitions[s_from].childs, s_to)
                transitions[s_from].probs = np.append(transitions[s_from].probs, prob)
        self.__parse(transitions)

    def __parse(self, transitions):
        # Keep original transitions for write-back and debugging.
        # Build unreduced_states as normalised (paren-stripped) forms because
        # SequenceGenerator.generate_sequence() strips parens from state strings
        # (e.g. "i(1)" → "i1"), so all downstream lookups use stripped keys.
        self.transitions = transitions
        self.unreduced_states = list(dict.fromkeys(
            _strip_parens(k) for k in transitions
        ))
        self.num_unreduced_states = len(self.unreduced_states)

        self.__parse_states(list(transitions.keys()))
        self.__parse_transitions(transitions)

    def __parse_states(self, original_keys: list) -> None:
        # --- Reduced states (observable symbols) ---
        # Derive from ORIGINAL state names so multi-character bases like "BX"
        # (from "BX(1)", "BX(2)") are preserved correctly by reduce_state.
        # Sort before assigning IDs to align with SymbolEncoder, which assigns
        # indices in sorted alphabet order (EOS=0, symbols=1..N).
        reduced_set: set = {self.reduce_state(s) for s in original_keys}
        for i, s_red in enumerate(sorted(reduced_set), start=1):
            self.reduced_states.append(s_red)
            self.red_state_to_id_map[s_red] = i
        self.num_reduced_states = len(self.reduced_states)

        # --- Unreduced states (normalised forms) ---
        # self.unreduced_states already holds normalised (stripped) keys.
        # Sort for determinism; the ordering only affects classification target IDs.
        for i, s_norm in enumerate(sorted(self.unreduced_states), start=1):
            self.unred_state_to_id_map[s_norm] = i

        # EOS
        self.red_state_to_id_map["#"] = 0
        self.unred_state_to_id_map["#"] = 0
        self.num_reduced_states += 1
        self.num_unreduced_states += 1
        self.reduced_states.append("#")
        self.unreduced_states.append("#")
        self.id_to_red_state_map = {v: k for k, v in self.red_state_to_id_map.items()}
        self.id_to_unred_state_map = {v: k for k, v in self.unred_state_to_id_map.items()}

    def __parse_transitions(self, transitions) -> None:
        transition_probs: dict = {}

        # One probability array per normalised unreduced state (including EOS).
        for s_norm in self.unreduced_states:
            transition_probs[s_norm] = np.zeros(self.num_reduced_states)

        # Fill probabilities.  state_name and child_s are in original form;
        # reduce_state is called on originals for correct multi-char base handling.
        for state_name, mcstate in transitions.items():
            s_from = _strip_parens(state_name)
            for i, child_s in enumerate(mcstate.childs):
                child_id = self.red_state_to_id_map[self.reduce_state(child_s)]
                transition_probs[s_from][child_id] += mcstate.probs[i]

        self.transition_probs = transition_probs

    def __call__(self, state_seq):
        if self.transition_probs is None:
            raise RuntimeError("Transition probabilities have not been initialized")

        target_probs = []
        for state in state_seq:
            if state == "#":
                x = np.zeros(self.num_reduced_states)
                x[0] = 1
                target_probs.append(x)
            else:
                transition_prob = self.transition_probs[state]
                target_probs.append(transition_prob)
        return np.array(target_probs)

    def reduce_state(self, s: str) -> str:
        """Return the observable symbol for state ``s``.

        Strips the parenthesised disambiguation index, if any:
        ``"BX(12)"`` → ``"BX"``, ``"i(1)"`` → ``"i"``, ``"b"`` → ``"b"``.
        Works for single- and multi-character base symbols, and for any number
        of digits in the index.
        """
        idx = s.find('(')
        return s if idx == -1 else s[:idx]

    def print_transitions(self):
        for s_from, state in self.transitions.items():
            print(s_from, state)

    def red_state_to_id(self, red_state):
        return self.red_state_to_id_map[red_state]

    def unred_state_to_id(self, unred_state):
        return self.unred_state_to_id_map[unred_state]

    def id_to_red_state(self, id):
        return self.id_to_red_state_map[id]

    def id_to_unred_state(self, id):
        return self.id_to_unred_state_map[id]

    def get_reduced_states_sorted(self):
        return sorted(self.reduced_states, key=lambda red_state: self.red_state_to_id(red_state))

    def get_unreduced_states_sorted(self):
        return sorted(self.unreduced_states, key=lambda unred_state: self.unred_state_to_id(unred_state))


class DatasetGenerator:
    """
    Generator for creating synthetic sequence datasets.

    This class handles the generation of training and test datasets with
    configurable parameters, parallel processing, and file I/O operations.

    Attributes:
        seq_generator: Sequence generator instance
        dataset_size: Number of samples to generate
        output_dir: Directory to save generated datasets
        config_file_path: Path to configuration file
        generate_train: Whether to generate training data
        generate_test: Whether to generate test data
        append_hash_to_output_dir: Whether to append hash to output directory
    """

    def __init__(
        self,
        seq_generator: Any,
        dataset_size: int = 0,
        output_dir: Optional[str] = None,
        config_file_path: Optional[str] = None,
        config_snapshot: Optional[dict] = None,
        manifest: Optional[dict] = None,
        split: Optional[str] = None,
        generate_train: bool = False,
        generate_test: bool = False,
        append_hash_to_output_dir: bool = False,
    ) -> None:
        """
        Initialize the DatasetGenerator.

        Args:
            seq_generator: Sequence generator instance
            dataset_size: Number of samples to generate
            output_dir: Directory to save generated datasets
            config_file_path: Path to configuration file
            generate_train: Whether to generate training data
            generate_test: Whether to generate test data
            append_hash_to_output_dir: Whether to append hash to output directory
        """
        self.seq_generator = seq_generator
        self.dataset_size = dataset_size
        self.output_dir = output_dir
        self.config_file_path = config_file_path
        self.config_snapshot = config_snapshot
        self.manifest = manifest
        self.split = split
        self.generate_train = generate_train
        self.generate_test = generate_test
        self.append_hash_to_output_dir = append_hash_to_output_dir

    @staticmethod
    def create_gensample_from_str(line: str) -> GeneratorSample:
        """
        Create a GeneratorSample from a string representation.

        Args:
            line: String in format "class_seq::state_seq::length"

        Returns:
            GeneratorSample: Parsed sample object

        Raises:
            ValueError: If the line is malformed or sequences have different lengths
        """
        raw_line = line.rstrip("\n")
        fields = raw_line.split("::", 3)
        if len(fields) not in {3, 4}:
            raise ValueError(
                "GeneratorSample line must have 3 or 4 '::'-separated fields, "
                f"got {len(fields)}: {raw_line!r}"
            )

        try:
            class_seq = fields[0].replace("[", "").replace("]", "")
            class_seq = class_seq.split(",")
            class_seq = [int(d.strip()) for d in class_seq]

            state_seq = fields[1].replace("[", "").replace("]", "")
            state_seq = state_seq.replace("'", "")
            state_seq = state_seq.split(",")
            state_seq = [d.strip() for d in state_seq]

            length = int(fields[2])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed GeneratorSample line: {raw_line!r}") from exc

        if len(class_seq) != len(state_seq):
            raise ValueError("Class and state sequences must have same length")

        # 4th field (targets) is optional: legacy 3-field datasets parse with
        # targets=None and rely on the task builder to recompute derivable ones.
        try:
            targets = _targets_from_json(fields[3]) if len(fields) >= 4 else None
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Malformed GeneratorSample targets field: {raw_line!r}") from exc

        return GeneratorSample(class_seq, state_seq, length, targets=targets)

    @staticmethod
    def read_transitions_from_file(file) -> Dict[str, MCState]:
        """
        Read transition data from a file.

        Args:
            file: File object to read from

        Returns:
            Dictionary mapping state names to MCState objects
        """
        transitions = {}
        for line in file:
            line = line.replace("\n", "").strip()
            line = line.replace("(", "").replace(")", "")
            line = line.replace("'", "")
            line = line.split(",")
            s_from = line[0].strip()
            s_to = line[1].strip()
            prob = float(line[2].strip())

            if s_from not in transitions:
                transitions[s_from] = MCState(s_from, np.array([s_to]), np.array([prob]))
            else:
                transitions[s_from].childs = np.append(transitions[s_from].childs, s_to)
                transitions[s_from].probs = np.append(transitions[s_from].probs, prob)
        return transitions

    @staticmethod
    def write_gensample_to_file(file, gensample: GeneratorSample) -> None:
        """
        Write a GeneratorSample to a file.

        Args:
            file: File object to write to
            gensample: GeneratorSample to write

        Raises:
            ValueError: If class_seq and state_seq have different lengths
        """
        class_seq = gensample.class_seq.tolist()
        state_seq = gensample.state_seq.tolist()

        if len(class_seq) != len(state_seq):
            raise ValueError("Class and state sequences must have same length")

        targets_field = _targets_to_json(getattr(gensample, "targets", None))
        file.write(f"{class_seq}::{state_seq}::{gensample.length}::{targets_field}\n")

    def generate(self) -> None:
        """
        Generate the dataset according to configuration.

        Creates output directory and generates train/test datasets
        as specified in the configuration.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Generating in {self.output_dir}!")
        if self.split is not None:
            self.__generate_for_dataset(self.split, self.dataset_size)
        elif self.generate_train:
            # self.__generate_for_dataset_parallelize('train', self.dataset_size)
            self.__generate_for_dataset("train", self.dataset_size)

        if self.generate_test:
            # self.__generate_for_dataset_parallelize('test', self.dataset_size)
            self.__generate_for_dataset("test", self.dataset_size)

        self.__write_config_to_file()
        self.__write_manifest_to_file()
        if hasattr(self.seq_generator.source, "transitions"):
            self.__write_transitions_to_file()

    def _generate_single(self, args) -> None:
        """
        Generate a single sample (used for parallel processing).

        Args:
            args: Tuple containing (idx, generator, output_file, lock)
        """
        idx, generator, output_file, lock = args
        gensample = generator.generate(idx)
        # Use lock to safely write to shared file
        with lock:
            with open(output_file, "a") as f:
                DatasetGenerator.write_gensample_to_file(f, gensample)

    def __generate_for_dataset(self, filename: str, dataset_size: int) -> None:
        """
        Generate dataset sequentially.

        Args:
            filename: Name of the output file
            dataset_size: Number of samples to generate
        """
        print(f"Generating {filename}!")
        file = open(os.path.join(self.output_dir, filename), "w")
        for i in tqdm.tqdm(range(dataset_size)):
            gensample = self.seq_generator.generate(i)
            DatasetGenerator.write_gensample_to_file(file, gensample)
        file.close()

    def __generate_for_dataset_parallelize(self, filename: str, dataset_size: int) -> None:
        """
        Generate dataset using parallel processing.

        Args:
            filename: Name of the output file
            dataset_size: Number of samples to generate
        """
        print(f"Generating {filename}!")

        output_file = os.path.join(self.output_dir, filename)

        # Create a manager to share the lock between processes
        with Manager() as manager:
            lock = manager.Lock()

            # Clear the file before starting
            with open(output_file, "w") as f:
                pass

            # Create argument tuples for each task
            args_list = [(i, self.seq_generator, output_file, lock) for i in range(dataset_size)]

            # Use multiprocessing to generate and write samples in parallel
            with Pool() as pool:
                list(
                    tqdm.tqdm(
                        pool.imap(self._generate_single, args_list), total=dataset_size, desc="Generating samples"
                    )
                )

    def __write_config_to_file(self) -> None:
        """Write configuration to YAML file."""
        print(f"Writing config!")

        if self.config_file_path is None:
            config = self.config_snapshot or {}
        else:
            with open(self.config_file_path, "r") as f:
                config = yaml.safe_load(f)

        with open(os.path.join(self.output_dir, "config.yaml"), "w") as file:
            file.write(yaml.safe_dump(config))

    def __write_manifest_to_file(self) -> None:
        """Write generated dataset metadata."""
        if self.manifest is None:
            return
        print("Writing manifest!")
        with open(os.path.join(self.output_dir, "manifest.yaml"), "w") as file:
            file.write(yaml.safe_dump(self.manifest, sort_keys=True))

    def __write_transitions_to_file(self) -> None:
        """Write transition data to file."""
        print(f"Writing transitions!")
        with open(os.path.join(self.output_dir, "transitions"), "w") as file:
            for transition in self.seq_generator.source.transitions:
                file.write(f"{transition[0]}, {transition[1]}, {float(transition[2])}\n")
