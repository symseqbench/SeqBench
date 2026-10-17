# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Stable symbol → integer mapping used to convert ``Trial.symbols`` (strings)
into the ``class_seq`` integer arrays consumed downstream.

EOS is reserved at index 0; alphabet symbols are assigned indices 1..N in
the order provided by the source's ``alphabet`` list. This preserves the
on-disk dataset format used before the symseq refactor (where EOS=0 and
class indices started at 1).
"""

from __future__ import annotations


class SymbolEncoder:
    EOS_SYMBOL: str = "#"
    EOS_INDEX: int = 0

    def __init__(self, alphabet: list[str]):
        self.alphabet: list[str] = list(alphabet)
        self._sym_to_idx: dict[str, int] = {
            s: i + 1 for i, s in enumerate(self.alphabet)
        }
        if self.EOS_SYMBOL in self._sym_to_idx:
            raise ValueError(
                f"Alphabet must not contain the reserved EOS symbol "
                f"{self.EOS_SYMBOL!r}."
            )
        self._sym_to_idx[self.EOS_SYMBOL] = self.EOS_INDEX

    def encode(self, symbols: list[str]) -> list[int]:
        try:
            return [self._sym_to_idx[s] for s in symbols]
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(
                f"Symbol {missing!r} not in alphabet {self.alphabet}."
            ) from exc

    def __len__(self) -> int:
        return len(self.alphabet) + 1
