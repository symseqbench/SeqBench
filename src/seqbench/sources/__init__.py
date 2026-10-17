# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Sources of trials consumed by SeqBench. Each module wraps an external
library (e.g. symseq) and exposes a constructor that returns a TrialSource-
compatible object — anything with ``alphabet``, ``draw_trial()`` and
``draw_batch(n)``.
"""

from seqbench.sources.symseq_source import build_symseq_source

__all__ = ["build_symseq_source"]
