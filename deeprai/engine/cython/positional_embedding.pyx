import numpy as np
cimport numpy as cnp

cpdef cnp.ndarray[cnp.float64_t, ndim = 2] apply_positional_embedding(cnp.ndarray[cnp.float64_t, ndim = 2] sequence):
    """
    Applies positional embedding to a given sequence.
    
    Args:
        sequence: A 2D numpy array where the first dimension is sequence length and the second is embedding dimension.
    
    Returns:
        A 2D numpy array with positional embeddings applied.
    """
    cdef int position, i
    cdef int seq_len = sequence.shape[0]
    cdef int embed_dim = sequence.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] position_embedding = np.zeros((seq_len, embed_dim))

    for position in range(seq_len):
        for i in range(0, embed_dim, 2):
            position_embedding[position, i] = np.sin(position / (10000 ** ((2 * i) / embed_dim)))
            if i + 1 < embed_dim:
                position_embedding[position, i + 1] = np.cos(position / (10000 ** ((2 * i + 1) / embed_dim)))

    sequence += position_embedding
    return sequence
