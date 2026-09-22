import numpy as np


def average_embeddings(embeddings):
    """
    Average all contextual embeddings of a target
    within one corpus period.
    """

    if not embeddings:
        return None

    matrix = np.asarray(
        embeddings,
        dtype=np.float64,
    )

    return np.mean(
        matrix,
        axis=0,
    )


def cosine_distance(
    vector1,
    vector2,
):
    """
    Calculate cosine distance.
    """

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return 1.0

    similarity = (
        np.dot(vector1, vector2)
        / (norm1 * norm2)
    )

    similarity = np.clip(
        similarity,
        -1.0,
        1.0,
    )

    return 1.0 - similarity


def semantic_change_score(
    period1_embeddings,
    period2_embeddings,
):
    """
    Semantic change is the cosine distance between the
    averaged contextual representations from the two periods.
    """

    average1 = average_embeddings(
        period1_embeddings
    )

    average2 = average_embeddings(
        period2_embeddings
    )

    if average1 is None or average2 is None:
        return None

    return cosine_distance(
        average1,
        average2,
    )