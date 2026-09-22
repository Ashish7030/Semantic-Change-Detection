import os
import sys
import gzip
import random
import warnings

import numpy as np
import pandas as pd
import torch

from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel


# Configuration

MODEL_NAME = "FacebookAI/xlm-roberta-base"

SEED = 42
MAX_LENGTH = 128

# Used for reproducible sampling when period sizes differ.
RANDOM_SEED = 42

LANGUAGE_TARGETS = {
    "english": {
        "corpus1": "data/english/semeval2020_ulscd_eng/corpus1/lemma/ccoha1.txt.gz",
        "corpus2": "data/english/semeval2020_ulscd_eng/corpus2/lemma/ccoha2.txt.gz",
        "targets": "data/english/semeval2020_ulscd_eng/targets.txt",
        "graded": "data/english/semeval2020_ulscd_eng/truth/graded.txt",
    },

    "german": {
        "corpus1": "data/german/semeval2020_ulscd_ger/corpus1/lemma/dta.txt.gz",
        "corpus2": "data/german/semeval2020_ulscd_ger/corpus2/lemma/bznd.txt.gz",
        "targets": "data/german/semeval2020_ulscd_ger/targets.txt",
        "graded": "data/german/semeval2020_ulscd_ger/truth/graded.txt",
    },

    "latin": {
        "corpus1": "data/latin/semeval2020_ulscd_lat/corpus1/lemma/LatinISE1.txt.gz",
        "corpus2": "data/latin/semeval2020_ulscd_lat/corpus2/lemma/LatinISE2.txt.gz",
        "targets": "data/latin/semeval2020_ulscd_lat/targets.txt",
        "graded": "data/latin/semeval2020_ulscd_lat/truth/graded.txt",
    },

    "swedish": {
        "corpus1": "data/swedish/semeval2020_ulscd_swe/corpus1/lemma/kubhist2a.txt.gz",
        "corpus2": "data/swedish/semeval2020_ulscd_swe/corpus2/lemma/kubhist2b.txt.gz",
        "targets": "data/swedish/semeval2020_ulscd_swe/targets.txt",
        "graded": "data/swedish/semeval2020_ulscd_swe/truth/graded.txt",
    },
}


# Reproducibility

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Device

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Target handling

def normalize_target(target):
    """
    SemEval English targets contain POS suffixes such as:

        attack_nn
        circle_vb
        player_nn

    The lemma appearing in the corpus does not contain this suffix.

    For other languages, the target is returned unchanged.
    """

    suffixes = [
        "_nn",
        "_vb",
        "_jj",
        "_rb",
        "_md",
    ]

    for suffix in suffixes:
        if target.endswith(suffix):
            return target[:-len(suffix)]

    return target


# Load targets

def load_targets(path):
    targets = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            target = line.strip()

            if target:
                targets.append(target)

    return targets


# Load graded gold

def load_graded_gold(path):
    """
    Reads SemEval graded gold annotations.

    Expected format:
        target<TAB>score
    """

    gold = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) >= 2:
                target = parts[0]

                try:
                    score = float(parts[1])
                    gold[target] = score
                except ValueError:
                    continue

    return gold


# Load model

def load_model():
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True
    )

    device = get_device()

    model.to(device)
    model.eval()

    print(f"Using device: {device}")

    return tokenizer, model, device


# Extract target occurrences

def extract_target_embeddings(
    corpus_path,
    target,
    tokenizer,
    model,
    device
):
    """
    Extract one contextual representation for every occurrence
    of the target word in the corpus.

    Representation:
        sum of the final four hidden layers
        followed by averaging the target's subword representations.
    """

    target_word = normalize_target(target)

    embeddings = []

    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            words = line.split()

            # Fast check before tokenization.
            if target_word not in words:
                continue

            try:
                encoded = tokenizer(
                    words,
                    is_split_into_words=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_attention_mask=True,
                )
            except Exception as exc:
                warnings.warn(
                    f"Tokenization failed for target {target}: {exc}"
                )
                continue

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            with torch.no_grad():

                outputs = model(
                    **encoded,
                    output_hidden_states=True
                )

            # Sum the final four hidden layers.
            hidden = torch.stack(
                outputs.hidden_states[-4:]
            ).sum(dim=0)

            word_ids = encoded.pop(
                "input_ids",
                None
            )

            # Obtain word alignment separately because word_ids() is a tokenizer BatchEncoding method.
            batch_word_ids = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=MAX_LENGTH,
            ).word_ids(batch_index=0)

            target_indices = [
                i
                for i, word_id in enumerate(batch_word_ids)
                if word_id is not None
                and word_id < len(words)
                and words[word_id] == target_word
            ]

            if not target_indices:
                continue

            # In most cases target_indices correspond to one target occurrence in the sentence. If multiple occurrences are
            # present, represent each occurrence separately.
            occurrence_word_ids = sorted(
                set(
                    batch_word_ids[i]
                    for i in target_indices
                    if batch_word_ids[i] is not None
                )
            )

            for occurrence_word_id in occurrence_word_ids:

                token_positions = [
                    i
                    for i, wid in enumerate(batch_word_ids)
                    if wid == occurrence_word_id
                ]

                if not token_positions:
                    continue

                token_vectors = hidden[
                    0,
                    token_positions,
                    :
                ]

                # Average subword pieces.
                occurrence_vector = token_vectors.mean(
                    dim=0
                )

                occurrence_vector = (
                    occurrence_vector
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                embeddings.append(occurrence_vector)

    if not embeddings:
        return np.empty(
            (0, model.config.hidden_size),
            dtype=np.float32
        )

    return np.vstack(embeddings)


# Equal-size sampling

def equalize_usage_sets(
    embeddings1,
    embeddings2,
    seed=RANDOM_SEED
):
    """
    Make the two usage sets equal in size.

    n = min(n1, n2)

    If one period contains more usages, sample n usages
    without replacement.

    The sampling is deterministic because a fixed seed is used.
    """

    n1 = len(embeddings1)
    n2 = len(embeddings2)

    if n1 == 0 or n2 == 0:
        return None, None, 0

    n = min(n1, n2)

    rng = np.random.default_rng(seed)

    if n1 > n:
        indices1 = rng.choice(
            n1,
            size=n,
            replace=False
        )
        embeddings1 = embeddings1[
            np.sort(indices1)
        ]

    if n2 > n:
        indices2 = rng.choice(
            n2,
            size=n,
            replace=False
        )
        embeddings2 = embeddings2[
            np.sort(indices2)
        ]

    return embeddings1, embeddings2, n


# Cosine distance matrix

def cosine_distance_matrix(a, b):
    """
    Compute pairwise cosine distance:

        d(x,y) = 1 - cosine_similarity(x,y)

    Both sets are L2-normalized before multiplication.
    """

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    a_norm = np.linalg.norm(
        a,
        axis=1,
        keepdims=True
    )

    b_norm = np.linalg.norm(
        b,
        axis=1,
        keepdims=True
    )

    a = a / np.maximum(a_norm, 1e-12)
    b = b / np.maximum(b_norm, 1e-12)

    similarity = a @ b.T

    np.clip(
        similarity,
        -1.0,
        1.0,
        out=similarity
    )

    distances = 1.0 - similarity

    return distances.astype(np.float32)


# Greedy one-to-one SAMD

def greedy_samd_from_distance_matrix(distances):
    """
    Greedy one-to-one SAMD.

    At every step:
        1. Find the globally smallest remaining distance.
        2. Match that row and column.
        3. Remove both from future matching.

    Continue until all rows/columns have been matched.

    This is the one-to-one greedy matching procedure described
    for SAMD in Goworek & Dubossarsky (2026).
    """

    distances = np.asarray(
        distances,
        dtype=np.float32
    )

    n_rows, n_cols = distances.shape

    if n_rows == 0 or n_cols == 0:
        return float("nan"), 0

    if n_rows != n_cols:
        raise ValueError(
            "SAMD requires equal-sized usage sets. "
            f"Received {n_rows} x {n_cols}."
        )

    n = n_rows

    # Work on a copy because matched entries are masked.
    working = distances.copy()

    matched_distances = []

    for _ in range(n):

        # Find global minimum among all remaining pairs.
        flat_index = np.argmin(working)

        row, col = np.unravel_index(
            flat_index,
            working.shape
        )

        min_distance = float(
            working[row, col]
        )

        matched_distances.append(
            min_distance
        )

        # Remove this row and column from future matching.
        working[row, :] = np.inf
        working[:, col] = np.inf

    samd = float(
        np.mean(matched_distances)
    )

    return samd, n


# SAMD for one target

def compute_samd(
    embeddings1,
    embeddings2,
    seed=RANDOM_SEED
):
    """
    Complete SAMD calculation.

    Steps:
        1. Equalize usage counts.
        2. Compute cosine distances.
        3. Greedy one-to-one matching.
        4. Average matched distances.
    """

    if len(embeddings1) == 0 or len(embeddings2) == 0:
        return float("nan"), 0

    embeddings1, embeddings2, n = (
        equalize_usage_sets(
            embeddings1,
            embeddings2,
            seed=seed
        )
    )

    if n == 0:
        return float("nan"), 0

    distances = cosine_distance_matrix(
        embeddings1,
        embeddings2
    )

    samd_score, matched_pairs = (
        greedy_samd_from_distance_matrix(
            distances
        )
    )

    return samd_score, matched_pairs


# Main language experiment

def run_language(language):

    if language not in LANGUAGE_TARGETS:
        raise ValueError(
            f"Unknown language: {language}"
        )

    paths = LANGUAGE_TARGETS[language]

    # Check files

    for key, path in paths.items():

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {key} file:\n{path}"
            )

    # Load data

    targets = load_targets(
        paths["targets"]
    )

    graded_gold = load_graded_gold(
        paths["graded"]
    )

    tokenizer, model, device = load_model()

    print("=" * 70)
    print("RoBERTa + Symmetric Average Minimum Distance (SAMD)")
    print("=" * 70)
    print(f"Language: {language.upper()}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Targets: {len(targets)}")
    print(f"Sampling seed: {RANDOM_SEED}")
    print("=" * 70)

    results = []

    skipped = []

    # Process targets

    for index, target in enumerate(targets, start=1):

        print(
            f"\n[{index:02d}/{len(targets):02d}] {target}"
        )

        print(
            "      Period 1: extracting occurrences..."
        )

        embeddings1 = extract_target_embeddings(
            paths["corpus1"],
            target,
            tokenizer,
            model,
            device
        )

        print(
            "      Period 2: extracting occurrences..."
        )

        embeddings2 = extract_target_embeddings(
            paths["corpus2"],
            target,
            tokenizer,
            model,
            device
        )

        n1 = len(embeddings1)
        n2 = len(embeddings2)

        print(
            f"      period1={n1} period2={n2}"
        )

        # Both periods must contain at least one occurrence.
        if n1 == 0 or n2 == 0:

            print(
                "      SKIPPED: target missing from one period"
            )

            skipped.append(
                (target, "missing period")
            )

            continue

        n = min(n1, n2)

        print(
            f"      Equalized usage count: {n}"
        )

        if n1 != n2:
            print(
                f"      Sampling without replacement "
                f"(seed={RANDOM_SEED})"
            )

        # SAMD

        samd_score, matched_pairs = compute_samd(
            embeddings1,
            embeddings2,
            seed=RANDOM_SEED
        )

        print(
            f"      SAMD = {samd_score:.6f}"
        )

        print(
            f"      matched pairs = {matched_pairs}"
        )

        gold_score = graded_gold.get(
            target,
            np.nan
        )

        results.append(
            {
                "target": target,
                "score": samd_score,
                "binary_gold": np.nan,
                "graded_gold": gold_score,
                "period1_occurrences": n1,
                "period2_occurrences": n2,
                "sampled_occurrences": n,
                "matched_pairs": matched_pairs,
            }
        )

    # Evaluation

    result_df = pd.DataFrame(results)

    valid_eval = result_df[
        result_df["graded_gold"].notna()
        & result_df["score"].notna()
    ]

    if len(valid_eval) >= 2:

        rho, p_value = spearmanr(
            valid_eval["score"],
            valid_eval["graded_gold"]
        )

    else:

        rho = np.nan
        p_value = np.nan

    # Save results

    output_dir = "results/raw"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        output_dir,
        f"roberta_samd_{language}.csv"
    )

    result_df.to_csv(
        output_path,
        index=False
    )

    # Print final result

    print("\n")
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Language: {language.upper()}"
    )

    print(
        f"Valid targets: {len(result_df)}/{len(targets)}"
    )

    print(
        f"Skipped targets: {len(skipped)}"
    )

    print(
        f"Spearman rho: {rho:.4f}"
    )

    print(
        f"p-value: {p_value:.6f}"
    )

    print(
        f"Results saved to:\n{os.path.abspath(output_path)}"
    )

    if skipped:

        print("\nSkipped targets:")

        for target, reason in skipped:
            print(
                f"  - {target}: {reason}"
            )

    return {
        "language": language,
        "targets": len(targets),
        "valid": len(result_df),
        "skipped": len(skipped),
        "rho": rho,
        "p_value": p_value,
        "results": result_df,
    }


# Run all languages

def run_all():

    summaries = []

    for language in [
        "english",
        "german",
        "latin",
        "swedish",
    ]:

        print("\n\n")

        result = run_language(
            language
        )

        summaries.append(
            {
                "language": language,
                "targets": result["targets"],
                "valid": result["valid"],
                "skipped": result["skipped"],
                "rho": result["rho"],
                "p_value": result["p_value"],
            }
        )

    summary_df = pd.DataFrame(
        summaries
    )

    print("\n")
    print("=" * 70)
    print("SAMD SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    valid_rhos = summary_df[
        summary_df["rho"].notna()
    ]["rho"]

    if len(valid_rhos) > 0:

        macro_average = float(
            valid_rhos.mean()
        )

        print(
            f"\nMacro-average Spearman rho: "
            f"{macro_average:.4f}"
        )

    summary_path = (
        "results/raw/"
        "roberta_samd_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print(
        f"Summary saved to:\n"
        f"{os.path.abspath(summary_path)}"
    )


# Entry point

if __name__ == "__main__":

    set_seed(SEED)

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python -m experiments.08_roberta_samd english\n"
            "  python -m experiments.08_roberta_samd german\n"
            "  python -m experiments.08_roberta_samd latin\n"
            "  python -m experiments.08_roberta_samd swedish\n"
            "  python -m experiments.08_roberta_samd all"
        )

        sys.exit(1)

    language = sys.argv[1].lower()

    if language == "all":
        run_all()
    else:
        run_language(language)