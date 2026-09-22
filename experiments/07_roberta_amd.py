import os
import sys
import gzip
import re

import numpy as np
import pandas as pd
import torch

from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel


# Configuration

MODEL_NAME = "FacebookAI/xlm-roberta-base"

SEED = 42

# Number of source embeddings processed at once when calculating pairwise cosine distances.
CHUNK_SIZE = 512

# Maximum sequence length used by XLM-RoBERTa.
MAX_LENGTH = 128

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "raw")

os.makedirs(RESULTS_DIR, exist_ok=True)


# Reproducibility

np.random.seed(SEED)
torch.manual_seed(SEED)


# Language configuration

LANGUAGE_CONFIG = {
    "english": {
        "data_dir": os.path.join(
            DATA_DIR,
            "english",
            "semeval2020_ulscd_eng"
        ),
        "corpus1": os.path.join(
            DATA_DIR,
            "english",
            "semeval2020_ulscd_eng",
            "corpus1",
            "lemma",
            "ccoha1.txt.gz"
        ),
        "corpus2": os.path.join(
            DATA_DIR,
            "english",
            "semeval2020_ulscd_eng",
            "corpus2",
            "lemma",
            "ccoha2.txt.gz"
        ),
    },

    "german": {
        "data_dir": os.path.join(
            DATA_DIR,
            "german",
            "semeval2020_ulscd_ger"
        ),
        "corpus1": os.path.join(
            DATA_DIR,
            "german",
            "semeval2020_ulscd_ger",
            "corpus1",
            "lemma",
            "dta.txt.gz"
        ),
        "corpus2": os.path.join(
            DATA_DIR,
            "german",
            "semeval2020_ulscd_ger",
            "corpus2",
            "lemma",
            "bznd.txt.gz"
        ),
    },

    "latin": {
        "data_dir": os.path.join(
            DATA_DIR,
            "latin",
            "semeval2020_ulscd_lat"
        ),
        "corpus1": os.path.join(
            DATA_DIR,
            "latin",
            "semeval2020_ulscd_lat",
            "corpus1",
            "lemma",
            "LatinISE1.txt.gz"
        ),
        "corpus2": os.path.join(
            DATA_DIR,
            "latin",
            "semeval2020_ulscd_lat",
            "corpus2",
            "lemma",
            "LatinISE2.txt.gz"
        ),
    },

    "swedish": {
        "data_dir": os.path.join(
            DATA_DIR,
            "swedish",
            "semeval2020_ulscd_swe"
        ),
        "corpus1": os.path.join(
            DATA_DIR,
            "swedish",
            "semeval2020_ulscd_swe",
            "corpus1",
            "lemma",
            "kubhist2a.txt.gz"
        ),
        "corpus2": os.path.join(
            DATA_DIR,
            "swedish",
            "semeval2020_ulscd_swe",
            "corpus2",
            "lemma",
            "kubhist2b.txt.gz"
        ),
    },
}


# Device

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()


# Model loading

def load_model():
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True
    )

    model.to(DEVICE)
    model.eval()

    print(f"Using device: {DEVICE}")

    return tokenizer, model


# Target handling

def get_target_forms(target, language):
    """
    Returns possible surface forms for a target.

    English targets in the SemEval dataset can contain POS suffixes
    such as _nn, _vb, _jj, _rb and _md.
    """

    if language == "english":
        match = re.match(
            r"^(.+?)_(nn|vb|jj|rb|md)$",
            target
        )

        if match:
            return [match.group(1)]

    return [target]


# Find target occurrences

def sentence_contains_target(sentence, target_forms):
    """
    Check whether a sentence contains the target.
    """

    tokens = sentence.strip().split()

    target_forms_lower = {
        t.lower()
        for t in target_forms
    }

    for token in tokens:
        if token.lower() in target_forms_lower:
            return True

    return False


def find_target_occurrences(corpus_path, target, language):
    """
    Read a gzipped corpus sentence by sentence and return the
    sentences containing the target.

    The corpus is processed line by line to avoid loading the
    entire corpus into memory.
    """

    target_forms = get_target_forms(
        target,
        language
    )

    occurrences = []

    with gzip.open(
        corpus_path,
        "rt",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            if sentence_contains_target(
                line,
                target_forms
            ):
                occurrences.append(line)

    return occurrences


# Extract contextual embedding

def extract_target_embedding(
    sentence,
    target,
    language,
    tokenizer,
    model
):
    """
    Extract the contextual representation of the target word
    from one sentence.

    The final four hidden layers are summed. If the target is
    split into multiple subword tokens, their representations
    are averaged.
    """

    target_forms = get_target_forms(
        target,
        language
    )

    words = sentence.strip().split()

    target_indices = []

    target_forms_lower = {
        t.lower()
        for t in target_forms
    }

    for i, word in enumerate(words):
        if word.lower() in target_forms_lower:
            target_indices.append(i)

    if not target_indices:
        return None

    # Use the first matching occurrence in the sentence.
    target_word_index = target_indices[0]

    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**encoded)

    hidden_states = outputs.hidden_states

    # Sum final four hidden layers.
    representation = (
        hidden_states[-1]
        + hidden_states[-2]
        + hidden_states[-3]
        + hidden_states[-4]
    )

    representation = representation.squeeze(0)

    word_ids = encoded["input_ids"].new_zeros(
        encoded["input_ids"].shape,
        dtype=torch.long
    )

    # tokenizer(...).word_ids() is easier to use directly.
    word_ids_list = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH
    ).word_ids()

    matching_token_positions = [
        i
        for i, word_id in enumerate(word_ids_list)
        if word_id == target_word_index
    ]

    if not matching_token_positions:
        return None

    target_vectors = representation[
        matching_token_positions
    ]

    # Average subword representations.
    target_vector = target_vectors.mean(
        dim=0
    )

    return target_vector.cpu().numpy()


# Extract all contextual embeddings for a target

def extract_embeddings(
    corpus_path,
    target,
    language,
    tokenizer,
    model
):
    """
    Extract contextual representations for all occurrences
    of the target in a corpus.
    """

    sentences = find_target_occurrences(
        corpus_path,
        target,
        language
    )

    embeddings = []

    for sentence in sentences:

        embedding = extract_target_embedding(
            sentence,
            target,
            language,
            tokenizer,
            model
        )

        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        return np.empty(
            (0, 768),
            dtype=np.float32
        )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


# Directional AMD

def directional_amd(
    source_embeddings,
    target_embeddings,
    chunk_size=CHUNK_SIZE
):
    """
    Calculate directional AMD:

        AMD(A -> B)
        = average_a min_b cosine_distance(a,b)

    Pairwise distances are calculated in chunks to avoid
    allocating the full distance matrix.
    """

    if len(source_embeddings) == 0:
        return np.nan

    if len(target_embeddings) == 0:
        return np.nan

    total_distance = 0.0
    total_count = len(source_embeddings)

    for start in range(
        0,
        total_count,
        chunk_size
    ):

        end = min(
            start + chunk_size,
            total_count
        )

        source_chunk = source_embeddings[
            start:end
        ]

        distances = cdist(
            source_chunk,
            target_embeddings,
            metric="cosine"
        )

        minimum_distances = distances.min(
            axis=1
        )

        total_distance += float(
            minimum_distances.sum()
        )

    return total_distance / total_count


# Symmetric AMD

def calculate_amd(
    period1_embeddings,
    period2_embeddings
):
    """
    Calculate symmetric AMD:

        AMD(A,B)
        = [AMD(A -> B) + AMD(B -> A)] / 2
    """

    amd_period1_to_period2 = directional_amd(
        period1_embeddings,
        period2_embeddings
    )

    amd_period2_to_period1 = directional_amd(
        period2_embeddings,
        period1_embeddings
    )

    symmetric_amd = (
        amd_period1_to_period2
        + amd_period2_to_period1
    ) / 2.0

    return (
        amd_period1_to_period2,
        amd_period2_to_period1,
        symmetric_amd
    )


# Load targets and gold annotations

def load_targets(data_dir):
    """
    Load target words from targets.txt.
    """

    targets_path = os.path.join(
        data_dir,
        "targets.txt"
    )

    with open(
        targets_path,
        "r",
        encoding="utf-8"
    ) as f:

        targets = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return targets


def load_gold_annotations(data_dir):
    """
    Load binary and graded gold annotations.
    """

    truth_dir = os.path.join(
        data_dir,
        "truth"
    )

    binary_path = os.path.join(
        truth_dir,
        "binary.txt"
    )

    graded_path = os.path.join(
        truth_dir,
        "graded.txt"
    )

    binary_gold = {}

    with open(
        binary_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) >= 2:
                binary_gold[parts[0]] = float(
                    parts[1]
                )

    graded_gold = {}

    with open(
        graded_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) >= 2:
                graded_gold[parts[0]] = float(
                    parts[1]
                )

    return binary_gold, graded_gold


# Run one language

def run_language(language):
    print("=" * 70)
    print("RoBERTa + Symmetric Average Minimum Distance (AMD)")
    print("=" * 70)

    if language not in LANGUAGE_CONFIG:
        raise ValueError(
            f"Unknown language: {language}"
        )

    config = LANGUAGE_CONFIG[language]

    corpus1 = config["corpus1"]
    corpus2 = config["corpus2"]
    data_dir = config["data_dir"]

    print(f"Language: {language.upper()}")
    print()
    print("Corpus 1:")
    print(corpus1)
    print()
    print("Corpus 2:")
    print(corpus2)
    print()

    targets = load_targets(
        data_dir
    )

    binary_gold, graded_gold = (
        load_gold_annotations(data_dir)
    )

    print(
        f"Number of target words: {len(targets)}"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print()

    tokenizer, model = load_model()

    results = []

    skipped = []

    for index, target in enumerate(
        targets,
        start=1
    ):

        print(
            f"[{index:02d}/{len(targets)}] {target}"
        )

        # Period 1

        print(
            "      Period 1:"
        )

        print(
            "      Extracting occurrences..."
        )

        period1_embeddings = extract_embeddings(
            corpus1,
            target,
            language,
            tokenizer,
            model
        )

        # Period 2

        print(
            "      Period 2:"
        )

        print(
            "      Extracting occurrences..."
        )

        period2_embeddings = extract_embeddings(
            corpus2,
            target,
            language,
            tokenizer,
            model
        )

        n1 = len(period1_embeddings)
        n2 = len(period2_embeddings)

        print(
            f"      period1={n1} period2={n2}"
        )

        # Check whether both periods contain usages

        if n1 == 0 or n2 == 0:

            print(
                "      SKIPPED: missing occurrences "
                "in one or both periods"
            )

            skipped.append(
                (
                    target,
                    "missing occurrences"
                )
            )

            continue

        # Calculate symmetric AMD

        (
            amd_p1_to_p2,
            amd_p2_to_p1,
            score
        ) = calculate_amd(
            period1_embeddings,
            period2_embeddings
        )

        print(
            f"      AMD P1 -> P2 = "
            f"{amd_p1_to_p2:.6f}"
        )

        print(
            f"      AMD P2 -> P1 = "
            f"{amd_p2_to_p1:.6f}"
        )

        print(
            f"      Symmetric AMD = "
            f"{score:.6f}"
        )

        results.append(
            {
                "target": target,
                "score": score,
                "amd_p1_to_p2": amd_p1_to_p2,
                "amd_p2_to_p1": amd_p2_to_p1,
                "binary_gold": binary_gold.get(
                    target,
                    np.nan
                ),
                "graded_gold": graded_gold.get(
                    target,
                    np.nan
                ),
                "period1_occurrences": n1,
                "period2_occurrences": n2
            }
        )

    # Save results

    results_df = pd.DataFrame(
        results
    )

    output_path = os.path.join(
        RESULTS_DIR,
        f"roberta_amd_{language}.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    # Evaluation

    valid_df = results_df.dropna(
        subset=[
            "score",
            "graded_gold"
        ]
    )

    if len(valid_df) >= 2:

        rho, p_value = spearmanr(
            valid_df["score"],
            valid_df["graded_gold"]
        )

    else:

        rho = np.nan
        p_value = np.nan

    # Final output

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Language: {language.upper()}"
    )

    print(
        f"Valid targets: "
        f"{len(valid_df)}/{len(targets)}"
    )

    print(
        f"Skipped targets: "
        f"{len(skipped)}"
    )

    print(
        f"Spearman rho: {rho:.4f}"
    )

    print(
        f"p-value: {p_value:.6f}"
    )

    if skipped:

        print()
        print("Skipped targets:")

        for target, reason in skipped:
            print(
                f"  - {target}: {reason}"
            )

    print()
    print(
        f"Results saved to:"
    )

    print(output_path)

    return results_df, rho, p_value


# Main

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python -m experiments.07_roberta_amd "
            "<language>"
        )

        print()
        print(
            "Available languages:"
        )

        print(
            "  english"
        )

        print(
            "  german"
        )

        print(
            "  latin"
        )

        print(
            "  swedish"
        )

        sys.exit(1)

    language = sys.argv[1].lower()

    run_language(language)


if __name__ == "__main__":
    main()