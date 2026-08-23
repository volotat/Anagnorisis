"""Everything that loads weights.

The embedder and the descriptor cannot be resident together on an 8 GB card, so
each of these owns a subprocess it can start and kill; the evaluator is small
enough not to care. Nothing here knows what a file is — it takes tensors and
paths and gives back vectors and text.
"""
