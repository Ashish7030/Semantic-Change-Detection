import numpy as np
from scipy.linalg import orthogonal_procrustes


def get_shared_vocabulary(model1, model2):

    vocab1 = set(
        model1.wv.key_to_index.keys()
    )

    vocab2 = set(
        model2.wv.key_to_index.keys()
    )

    return sorted(
        vocab1.intersection(vocab2)
    )


def build_alignment_matrices(
    model1,
    model2,
    vocabulary,
):

    matrix1 = np.asarray(
        [
            model1.wv[word]
            for word in vocabulary
        ],
        dtype=np.float64,
    )

    matrix2 = np.asarray(
        [
            model2.wv[word]
            for word in vocabulary
        ],
        dtype=np.float64,
    )

    return matrix1, matrix2


def mean_center(matrix):

    return (
        matrix
        - matrix.mean(
            axis=0,
            keepdims=True,
        )
    )


def normalize_rows(matrix):

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0

    return matrix / norms


def align_embeddings(
    model1,
    model2,
    vocabulary,
):
    """
    Align Corpus 2 embedding space
    to Corpus 1 using Orthogonal Procrustes.

    Preprocessing:
        1. Mean-centering
        2. Vector normalization
        3. Orthogonal Procrustes
    """

    matrix1, matrix2 = (
        build_alignment_matrices(
            model1,
            model2,
            vocabulary,
        )
    )

    matrix1 = mean_center(
        matrix1
    )

    matrix2 = mean_center(
        matrix2
    )

    matrix1 = normalize_rows(
        matrix1
    )

    matrix2 = normalize_rows(
        matrix2
    )

    rotation, scale = (
        orthogonal_procrustes(
            matrix2,
            matrix1,
        )
    )

    return rotation, scale


def cosine_distance(
    vector1,
    vector2,
):

    norm1 = np.linalg.norm(
        vector1
    )

    norm2 = np.linalg.norm(
        vector2
    )

    if norm1 == 0 or norm2 == 0:
        return 1.0

    cosine_similarity = (
        np.dot(
            vector1,
            vector2,
        )
        / (norm1 * norm2)
    )

    cosine_similarity = np.clip(
        cosine_similarity,
        -1.0,
        1.0,
    )

    return 1.0 - cosine_similarity