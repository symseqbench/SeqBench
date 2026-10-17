# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Metrics utilities for sequence analysis including n-gram frequency calculations.
"""

import numpy as np


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for SeqBench metrics display/DataFrame output. "
            "Install SeqBench with the metrics extra: pip install 'seqbench[metrics]'."
        ) from exc
    return pd


def _require_nltk():
    try:
        import nltk
    except ImportError as exc:
        raise ImportError(
            "nltk is required for SeqBench edit-distance metrics. "
            "Install SeqBench with the metrics extra: pip install 'seqbench[metrics]'."
        ) from exc
    return nltk


def chunk(seq, n):
	"""
	Chunk the sequence into all of its constituent n-grams and determine their frequency in the set
	:param seq: [list or generator] full symbolic sequence (should be as long as possible, particularly for large n)
	:param n: [int] order parameter (chunk "length")
	:return:
	"""
	all_ngrams = [''.join(list(seq)[ii:ii + n]) for ii in range(len(list(seq)))]
	un_ngrams = np.unique(all_ngrams).tolist()

	# check if all possibilities are represented (n-gram frequency):
	count = []
	for ii, nn in enumerate(un_ngrams):
		if len(nn) < n:
			un_ngrams.pop(ii)
		else:
			count.append(len(np.where(np.array(all_ngrams) == nn)[0]))
	return all_ngrams, un_ngrams


def chunk_transitions(seq, n, display=True, return_labels=False):
	"""
	Determine the transition matrix for n-gram sequences
	:param seq: [list or generator] full symbolic sequence (should be as long as possible, particularly for large n)
	:param n: [int] order parameter (chunk "length")
	:param display: [bool] show transition table
	:return M: [array]
	"""
	# TODO - this needs to be optimized
	all_ngrams, un_ngrams = chunk(seq, n)
	M = np.zeros((len(un_ngrams), len(un_ngrams)))
	nGrams = np.array(all_ngrams)
	for ii, i in enumerate(un_ngrams):
		for jj, j in enumerate(un_ngrams):
			M[ii, jj] = float(any(nGrams[np.where(nGrams == i)[0][:-1] + 1] == j))
	if return_labels:
		pd = _require_pandas()
		df = pd.DataFrame(M, columns=un_ngrams, index=un_ngrams)
		if display:
			print(df)
		return df
	if display:
		pd = _require_pandas()
		df = pd.DataFrame(M, columns=un_ngrams, index=un_ngrams)
		print(df)
	return M


def hamming_distance(seq1, seq2):
	"""
	Calculate hamming distance between 2 sequences
	:param seq1:
	:param seq2:
	"""
	return sum(map(str.__ne__, str(seq1), str(seq2)))


def edit_distance(seq1, seq2):
	"""
	Calculate edit distance between 2 sequences. Requires the editdistance package
	:param seq1:
	:param seq2:
	"""
	nltk = _require_nltk()
	return nltk.edit_distance(seq1, seq2)
