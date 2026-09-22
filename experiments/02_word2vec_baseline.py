from pathlib import Path
import sys
import csv
import argparse

import numpy as np


# Project setup

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))


from src.config import DATA_DIR, LANGUAGES
from src.data_loader import load_targets, load_truth
from src.models.word2vec_model import train_word2vec
from src.methods.procrustes import (
    get_shared_vocabulary,
    align_embeddings,
    cosine_distance,
)


# Exact SemEval-2020 dataset structure

DATASET_NAMES = {
    "english": "semeval2020_ulscd_eng",
    "german": "semeval2020_ulscd_ger",
    "latin": "semeval2020_ulscd_lat",
    "swedish": "semeval2020_ulscd_swe",
}


CORPUS_FILENAMES = {
    "english": {
        "corpus1": "ccoha1.txt.gz",
        "corpus2": "ccoha2.txt.gz",
    },

    "german": {
        "corpus1": "dta.txt.gz",
        "corpus2": "bznd.txt.gz",
    },

    "latin": {
        "corpus1": "LatinISE1.txt.gz",
        "corpus2": "LatinISE2.txt.gz",
    },

    "swedish": {
        "corpus1": "kubhist2a.txt.gz",
        "corpus2": "kubhist2b.txt.gz",
    },
}


# Gold-truth parsing

def parse_binary_truth(lines):
    """
    Parse SemEval binary gold annotations.

    Format:
        target    0/1
    """

    truth = {}

    for line in lines:
        parts = line.split()

        if len(parts) >= 2:
            target = parts[0]
            value = int(parts[1])
            truth[target] = value

    return truth


def parse_graded_truth(lines):
    """
    Parse SemEval graded gold annotations.

    Format:
        target    floating-point-score
    """

    truth = {}

    for line in lines:
        parts = line.split()

        if len(parts) >= 2:
            target = parts[0]
            value = float(parts[1])
            truth[target] = value

    return truth


# Dataset paths

def get_corpus_paths(language):
    """
    Return the exact corpus1 and corpus2 paths
    for a SemEval language.
    """

    if language not in DATASET_NAMES:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    dataset_dir = (
        DATA_DIR
        / language
        / DATASET_NAMES[language]
    )

    corpus1_path = (
        dataset_dir
        / "corpus1"
        / "lemma"
        / CORPUS_FILENAMES[language]["corpus1"]
    )

    corpus2_path = (
        dataset_dir
        / "corpus2"
        / "lemma"
        / CORPUS_FILENAMES[language]["corpus2"]
    )

    return corpus1_path, corpus2_path


# Run one language

def run_language(language):

    print("\n")
    print("=" * 70)
    print("Word2Vec + Orthogonal Procrustes")
    print(f"Language: {language.upper()}")
    print("=" * 70)

    # Locate corpora

    corpus1_path, corpus2_path = get_corpus_paths(
        language
    )

    print("\nCorpus 1:")
    print(corpus1_path)

    print("\nCorpus 2:")
    print(corpus2_path)

    if not corpus1_path.exists():
        raise FileNotFoundError(
            f"\nCorpus 1 not found:\n{corpus1_path}"
        )

    if not corpus2_path.exists():
        raise FileNotFoundError(
            f"\nCorpus 2 not found:\n{corpus2_path}"
        )

    # Load targets and gold data

    targets = load_targets(language)

    binary_lines = load_truth(
        language,
        "binary",
    )

    graded_lines = load_truth(
        language,
        "graded",
    )

    binary_truth = parse_binary_truth(
        binary_lines
    )

    graded_truth = parse_graded_truth(
        graded_lines
    )

    print(
        f"\nNumber of target words: {len(targets)}"
    )

    # Train Word2Vec - Corpus 1

    print(
        "\nTraining Word2Vec on Corpus 1..."
    )

    model1 = train_word2vec(
        corpus1_path,
        vector_size=300,
        window=5,
        min_count=1,
        sg=1,
        negative=5,
        epochs=5,
        seed=42,
    )

    # Train Word2Vec - Corpus 2

    print(
        "\nTraining Word2Vec on Corpus 2..."
    )

    model2 = train_word2vec(
        corpus2_path,
        vector_size=300,
        window=5,
        min_count=1,
        sg=1,
        negative=5,
        epochs=5,
        seed=42,
    )

    print(
        "\nWord2Vec training completed."
    )

    # Shared vocabulary

    shared_vocab = get_shared_vocabulary(
        model1,
        model2,
    )

    print(
        f"\nShared vocabulary size: "
        f"{len(shared_vocab)}"
    )

    if len(shared_vocab) == 0:
        raise RuntimeError(
            "No shared vocabulary found."
        )

    # Orthogonal Procrustes alignment

    print(
        "\nAligning embedding spaces..."
    )

    rotation, scale = align_embeddings(
        model1,
        model2,
        shared_vocab,
    )

    print(
        "Alignment completed."
    )

    # Calculate shared-vocabulary means ONCE

    print(
        "\nCalculating shared-vocabulary means..."
    )

    matrix1 = np.asarray(
        [
            model1.wv[word]
            for word in shared_vocab
        ],
        dtype=np.float64,
    )

    matrix2 = np.asarray(
        [
            model2.wv[word]
            for word in shared_vocab
        ],
        dtype=np.float64,
    )

    mean1 = matrix1.mean(
        axis=0
    )

    mean2 = matrix2.mean(
        axis=0
    )

    print(
        "Mean-centering preparation completed."
    )

    # Calculate semantic-change scores

    results = []

    missing_targets = []

    print(
        "\nCalculating semantic-change scores..."
    )

    for target in targets:

        # Check target availability

        if target not in model1.wv:

            missing_targets.append(
                (target, "corpus1")
            )

            continue

        if target not in model2.wv:

            missing_targets.append(
                (target, "corpus2")
            )

            continue

        # Retrieve target vectors

        vector1 = np.asarray(
            model1.wv[target],
            dtype=np.float64,
        )

        vector2 = np.asarray(
            model2.wv[target],
            dtype=np.float64,
        )

        # Mean-center

        vector1 = vector1 - mean1
        vector2 = vector2 - mean2

        # Normalize target vectors

        norm1 = np.linalg.norm(
            vector1
        )

        norm2 = np.linalg.norm(
            vector2
        )

        if norm1 == 0 or norm2 == 0:

            score = 1.0

        else:

            vector1 = (
                vector1 / norm1
            )

            vector2 = (
                vector2 / norm2
            )

            # Align Corpus 2 vector into Corpus 1 space

            aligned_vector2 = (
                vector2 @ rotation
            )

            # Cosine distance

            score = cosine_distance(
                vector1,
                aligned_vector2,
            )

        # Store result

        results.append(
            {
                "target": target,
                "score": float(score),
                "binary_gold":
                    binary_truth.get(target),
                "graded_gold":
                    graded_truth.get(target),
            }
        )

    # Result summary

    print(
        f"\nSuccessfully calculated scores "
        f"for {len(results)} / "
        f"{len(targets)} targets."
    )

    # Missing targets

    if missing_targets:

        print(
            "\nMissing targets:"
        )

        for target, corpus in missing_targets:

            print(
                f"  {target} "
                f"-> missing from {corpus}"
            )

    # Preview

    print(
        "\nFirst 10 results:"
    )

    for result in results[:10]:

        print(
            f"  {result['target']:20s} "
            f"score={result['score']:.6f} "
            f"binary={result['binary_gold']} "
            f"graded={result['graded_gold']}"
        )

    # Save CSV

    results_dir = (
        PROJECT_ROOT
        / "results"
        / "raw"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        results_dir
        / f"word2vec_{language}.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "score",
                "binary_gold",
                "graded_gold",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nResults saved to:\n{output_file}"
    )


# Main

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Word2Vec + Orthogonal Procrustes "
            "for SemEval-2020 Task 1."
        )
    )

    parser.add_argument(
        "language",
        nargs="?",
        choices=LANGUAGES,
        help="Run one language.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all four languages.",
    )

    args = parser.parse_args()

    if args.all:

        for language in LANGUAGES:
            run_language(language)

    elif args.language:

        run_language(
            args.language
        )

    else:

        parser.error(
            "Specify a language or use --all."
        )


if __name__ == "__main__":
    main()