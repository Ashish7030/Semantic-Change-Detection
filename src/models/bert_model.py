from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAMES = {
    "english": "bert-base-uncased",
    "german": "bert-base-german-cased",
    "latin": "bert-base-multilingual-uncased",
    "swedish": "af-ai-center/bert-base-swedish-uncased",
}


def get_device():
    """
    Select the fastest available device.

    Apple Silicon -> MPS
    NVIDIA GPU    -> CUDA
    Otherwise     -> CPU
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_bert(language: str):
    """
    Load the pretrained language-specific BERT model.
    """

    if language not in MODEL_NAMES:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    model_name = MODEL_NAMES[language]

    print(f"Loading BERT model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModel.from_pretrained(
        model_name
    )

    device = get_device()

    print(f"Using device: {device}")

    model.to(device)
    model.eval()

    return tokenizer, model, device


def find_target_occurrences(
    words: List[str],
    target: str,
) -> List[int]:
    """
    Find occurrences of a SemEval target in a tokenized
    sentence.

    English targets can contain POS suffixes such as
    attack_nn or circle_vb. We therefore try the original
    target as well as the lemma without the POS suffix.
    """

    target_variants = {target}

    suffixes = [
        "_nn",
        "_vb",
        "_jj",
        "_rb",
        "_md",
    ]

    for suffix in suffixes:
        if target.endswith(suffix):
            target_variants.add(
                target[:-len(suffix)]
            )

    positions = []

    for index, word in enumerate(words):
        if word in target_variants:
            positions.append(index)

    return positions


def extract_target_embedding(
    words: List[str],
    target_index: int,
    tokenizer,
    model,
    device,
):
    """
    Extract the contextual representation of one target
    occurrence.

    Representation:
        sum of the final four BERT hidden layers

    If the target is split into multiple WordPieces, the
    representations of those WordPieces are averaged.
    """

    encoded = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=128,
        padding=False,
        return_tensors="pt",
        return_attention_mask=True,
    )

    word_ids = encoded.word_ids(
        batch_index=0
    )

    target_positions = [
        position
        for position, word_id in enumerate(word_ids)
        if word_id == target_index
    ]

    if not target_positions:
        return None

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states

    # Last four BERT layers.
    last_four_layers = hidden_states[-4:]

    # Sum the final four layers.
    representation = (
        last_four_layers[0]
        + last_four_layers[1]
        + last_four_layers[2]
        + last_four_layers[3]
    )

    # Extract the target's WordPiece vectors.
    target_vectors = []

    for position in target_positions:
        vector = representation[
            0,
            position,
            :
        ]

        target_vectors.append(
            vector.detach()
            .cpu()
            .numpy()
        )

    # If the word consists of multiple WordPieces, average their vectors.
    if len(target_vectors) == 1:
        return target_vectors[0]

    return np.mean(
        np.asarray(target_vectors),
        axis=0,
    )


def extract_target_embeddings(
    corpus_path: Path,
    target: str,
    tokenizer,
    model,
    device,
    max_occurrences: Optional[int] = None,
):
    """
    Extract contextual embeddings only from sentences
    containing the target.

    This is much faster than running BERT over every sentence
    in the corpus.
    """

    embeddings = []

    with corpus_path.open(
        "rt",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            words = line.split()

            target_positions = find_target_occurrences(
                words,
                target,
            )

            if not target_positions:
                continue

            for target_index in target_positions:

                embedding = extract_target_embedding(
                    words=words,
                    target_index=target_index,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                )

                if embedding is not None:
                    embeddings.append(
                        embedding
                    )

                if (
                    max_occurrences is not None
                    and len(embeddings)
                    >= max_occurrences
                ):
                    return embeddings

    return embeddings