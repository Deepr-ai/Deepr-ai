from deeprai.engine.cython.positional_embedding import apply_positional_embedding as cy_apply_positional_embedding
import numpy as np


def embed_position(sequence: np.ndarray) -> np.ndarray:
    """
    Applies positional embedding to a given sequence using Cython implementation.

    Args:
        sequence: A 2D numpy array where the first dimension is sequence length and the second is embedding dimension.

    Returns:
        A 2D numpy array with positional embeddings applied.
    """
    return cy_apply_positional_embedding(sequence)
