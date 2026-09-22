from pathlib import Path
import csv

import numpy as np
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LANGUAGES = [
    "english",
    "german",
    "latin",
    "swedish",
]


def load_predictions(path):
    """
    Load target scores and gold annotations from a CSV file.
    """

    predictions = {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            predictions[row["target"]] = {
                "score": float(row["score"]),
                "binary_gold": int(row["binary_gold"]),
                "graded_gold": float(row["graded_gold"]),
            }

    return predictions


def combine_scores(
    word2vec_predictions,
    bert_predictions,
):
    """
    Combine Word2Vec and BERT semantic-change scores.

    Following the ensemble strategy used by Discovery,
    the two scores are multiplied for each target word.
    """

    combined = []

    common_targets = sorted(
        set(word2vec_predictions)
        & set(bert_predictions)
    )

    for target in common_targets:

        word2vec_score = (
            word2vec_predictions[target]["score"]
        )

        bert_score = (
            bert_predictions[target]["score"]
        )

        combined_score = (
            word2vec_score * bert_score
        )

        combined.append(
            {
                "target": target,
                "word2vec_score": word2vec_score,
                "bert_score": bert_score,
                "combined_score": combined_score,
                "binary_gold": (
                    word2vec_predictions[target][
                        "binary_gold"
                    ]
                ),
                "graded_gold": (
                    word2vec_predictions[target][
                        "graded_gold"
                    ]
                ),
            }
        )

    return combined


def calculate_spearman(predictions):

    predicted_scores = [
        item["combined_score"]
        for item in predictions
    ]

    gold_scores = [
        item["graded_gold"]
        for item in predictions
    ]

    correlation, p_value = spearmanr(
        predicted_scores,
        gold_scores,
    )

    return correlation, p_value


def evaluate_language(language):

    word2vec_file = (
        PROJECT_ROOT
        / "results"
        / "raw"
        / f"word2vec_{language}.csv"
    )

    bert_file = (
        PROJECT_ROOT
        / "results"
        / "raw"
        / f"bert_averaging_{language}.csv"
    )

    if not word2vec_file.exists():

        print(
            f"\n{language.upper()}: "
            f"Word2Vec results file not found."
        )

        return None

    if not bert_file.exists():

        print(
            f"\n{language.upper()}: "
            f"BERT results file not found."
        )

        return None

    word2vec_predictions = (
        load_predictions(word2vec_file)
    )

    bert_predictions = (
        load_predictions(bert_file)
    )

    combined_predictions = combine_scores(
        word2vec_predictions,
        bert_predictions,
    )

    spearman, p_value = (
        calculate_spearman(
            combined_predictions
        )
    )

    return {
        "language": language,
        "targets": len(combined_predictions),
        "spearman": spearman,
        "p_value": p_value,
        "predictions": combined_predictions,
    }


def save_combined_results(
    language,
    predictions,
):
    """
    Save the combined target-level scores.
    """

    output_file = (
        PROJECT_ROOT
        / "results"
        / "raw"
        / f"word2vec_bert_{language}.csv"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "target",
                "word2vec_score",
                "bert_score",
                "combined_score",
                "binary_gold",
                "graded_gold",
            ]
        )

        for item in predictions:

            writer.writerow(
                [
                    item["target"],
                    item["word2vec_score"],
                    item["bert_score"],
                    item["combined_score"],
                    item["binary_gold"],
                    item["graded_gold"],
                ]
            )

    return output_file


def main():

    print("=" * 70)
    print(
        "WORD2VEC + BERT "
        "COMBINATION"
    )
    print("=" * 70)

    print(
        "\nCombination method:"
    )

    print(
        "Word2Vec score × "
        "BERT contextual-averaging score"
    )

    results = []

    for language in LANGUAGES:

        result = evaluate_language(
            language
        )

        if result is None:
            continue

        results.append(result)

        output_file = save_combined_results(
            language,
            result["predictions"],
        )

        print(
            f"\n{language.upper()}"
        )

        print(
            f"Targets:   "
            f"{result['targets']}"
        )

        print(
            f"Spearman:  "
            f"{result['spearman']:.4f}"
        )

        print(
            f"p-value:   "
            f"{result['p_value']:.6f}"
        )

        print(
            f"Saved to:  "
            f"{output_file}"
        )

    # Macro-average across the four languages

    if results:

        mean_spearman = np.mean(
            [
                result["spearman"]
                for result in results
            ]
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "MACRO-AVERAGE ACROSS LANGUAGES"
        )

        print(
            "=" * 70
        )

        print(
            f"Mean Spearman: "
            f"{mean_spearman:.4f}"
        )

        print(
            "\nLanguages included:"
        )

        for result in results:

            print(
                f"  {result['language']}: "
                f"{result['spearman']:.4f}"
            )


if __name__ == "__main__":
    main()