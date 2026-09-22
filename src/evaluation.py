from pathlib import Path
import csv

import numpy as np
from scipy.stats import spearmanr


# PROJECT CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LANGUAGES = [
    "english",
    "german",
    "latin",
    "swedish",
]

TOTAL_TARGETS = {
    "english": 37,
    "german": 48,
    "latin": 40,
    "swedish": 31,
}


# LOAD PREDICTIONS

def load_predictions(path):
    """
    Load prediction results from a CSV file.

    Supported formats:

    Standard:
        target
        score
        binary_gold
        graded_gold

    Word2Vec + BERT:
        target
        word2vec_score
        bert_score
        combined_score
        binary_gold
        graded_gold

    The function uses:
        score
    or:
        combined_score

    as the prediction.

    Rows with missing prediction or graded-gold values
    are skipped.
    """

    predictions = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            # Get prediction score

            if (
                "score" in row
                and row["score"] is not None
                and row["score"].strip() != ""
            ):

                score = float(
                    row["score"]
                )

            elif (
                "combined_score" in row
                and row["combined_score"] is not None
                and row["combined_score"].strip() != ""
            ):

                score = float(
                    row["combined_score"]
                )

            else:

                # No prediction score
                continue

            # Get graded gold

            if (
                "graded_gold" not in row
                or row["graded_gold"] is None
                or row["graded_gold"].strip() == ""
            ):

                # No graded gold
                continue

            graded_gold = float(
                row["graded_gold"]
            )

            # Get binary gold

            binary_gold = None

            if (
                "binary_gold" in row
                and row["binary_gold"] is not None
                and row["binary_gold"].strip() != ""
            ):

                binary_gold = int(
                    float(
                        row["binary_gold"]
                    )
                )

            # Store prediction

            predictions.append(
                {
                    "target": row.get(
                        "target",
                        "",
                    ),
                    "score": score,
                    "binary_gold": binary_gold,
                    "graded_gold": graded_gold,
                }
            )

    return predictions


# SPEARMAN

def calculate_spearman(predictions):

    predicted_scores = [
        item["score"]
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


# EVALUATE ONE LANGUAGE

def evaluate_language(
    language,
    filename_prefix,
):

    results_file = (
        PROJECT_ROOT
        / "results"
        / "raw"
        / f"{filename_prefix}_{language}.csv"
    )

    if not results_file.exists():

        print(
            f"\n{language.upper()}: "
            f"results file not found."
        )

        return None

    predictions = load_predictions(
        results_file
    )

    if not predictions:

        print(
            f"\n{language.upper()}: "
            f"no valid predictions found."
        )

        return None

    spearman, p_value = (
        calculate_spearman(
            predictions
        )
    )

    return {
        "language": language,
        "valid_targets": len(predictions),
        "total_targets": TOTAL_TARGETS[language],
        "spearman": spearman,
        "p_value": p_value,
    }


# EVALUATE ONE METHOD

def evaluate_method(
    method_name,
    filename_prefix,
):

    print("\n")
    print("=" * 70)
    print(method_name)
    print("=" * 70)

    results = []

    for language in LANGUAGES:

        result = evaluate_language(
            language,
            filename_prefix,
        )

        if result is None:
            continue

        results.append(result)

        print(
            f"\n{language.upper()}"
        )

        print(
            f"Valid targets: "
            f"{result['valid_targets']}/"
            f"{result['total_targets']}"
        )

        print(
            f"Spearman rho: "
            f"{result['spearman']:.4f}"
        )

        print(
            f"p-value:      "
            f"{result['p_value']:.6f}"
        )

    if results:

        mean_spearman = np.mean(
            [
                result["spearman"]
                for result in results
            ]
        )

        print(
            "\n" + "-" * 70
        )

        print(
            "MACRO-AVERAGE ACROSS LANGUAGES"
        )

        print(
            "-" * 70
        )

        print(
            f"Mean Spearman: "
            f"{mean_spearman:.4f}"
        )

        print(
            "\nLanguage-level correlations:"
        )

        for result in results:

            print(
                f"  {result['language']}: "
                f"{result['spearman']:.4f}"
            )

        print(
            f"\nLanguages evaluated: "
            f"{len(results)}/4"
        )

        return {
            "method": method_name,
            "filename_prefix": filename_prefix,
            "results": results,
            "macro_spearman": mean_spearman,
        }

    return None


# FINAL COMPARISON

def print_final_comparison(all_results):

    print("\n")
    print("=" * 100)
    print("FINAL COMPARISON OF ALL METHODS")
    print("=" * 100)

    print(
        f"{'Method':<35}"
        f"{'English':>10}"
        f"{'German':>10}"
        f"{'Latin':>10}"
        f"{'Swedish':>10}"
        f"{'Macro':>10}"
    )

    print("-" * 100)

    for method_result in all_results:

        language_scores = {
            result["language"]: result["spearman"]
            for result in method_result["results"]
        }

        english = language_scores.get(
            "english",
            np.nan,
        )

        german = language_scores.get(
            "german",
            np.nan,
        )

        latin = language_scores.get(
            "latin",
            np.nan,
        )

        swedish = language_scores.get(
            "swedish",
            np.nan,
        )

        macro = method_result[
            "macro_spearman"
        ]

        print(
            f"{method_result['method']:<35}"
            f"{english:>10.4f}"
            f"{german:>10.4f}"
            f"{latin:>10.4f}"
            f"{swedish:>10.4f}"
            f"{macro:>10.4f}"
        )

    print("=" * 100)


# TARGET COVERAGE

def print_target_coverage(all_results):

    print("\n")
    print("=" * 100)
    print("TARGET COVERAGE")
    print("=" * 100)

    print(
        f"{'Method':<35}"
        f"{'English':>12}"
        f"{'German':>12}"
        f"{'Latin':>12}"
        f"{'Swedish':>12}"
    )

    print("-" * 100)

    for method_result in all_results:

        language_results = {
            result["language"]: result
            for result in method_result["results"]
        }

        coverage = []

        for language in LANGUAGES:

            if language in language_results:

                result = language_results[
                    language
                ]

                value = (
                    f"{result['valid_targets']}/"
                    f"{result['total_targets']}"
                )

            else:

                value = "N/A"

            coverage.append(value)

        print(
            f"{method_result['method']:<35}"
            f"{coverage[0]:>12}"
            f"{coverage[1]:>12}"
            f"{coverage[2]:>12}"
            f"{coverage[3]:>12}"
        )

    print("=" * 100)


# MAIN

def main():

    methods = [

        (
            "Word2Vec + Orthogonal Procrustes",
            "word2vec",
        ),

        (
            "BERT + Contextual Averaging",
            "bert_averaging",
        ),

        (
            "Word2Vec + BERT",
            "word2vec_bert",
        ),

        (
            "RoBERTa + K-Means + JSD",
            "roberta_kmeans_jsd",
        ),

        (
            "RoBERTa + AMD",
            "roberta_amd",
        ),

        (
            "RoBERTa + SAMD",
            "roberta_samd",
        ),
    ]

    all_results = []

    for method_name, filename_prefix in methods:

        result = evaluate_method(
            method_name,
            filename_prefix,
        )

        if result is not None:

            all_results.append(
                result
            )

    if all_results:

        print_final_comparison(
            all_results
        )

        print_target_coverage(
            all_results
        )

    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Methods evaluated: "
        f"{len(all_results)}/6"
    )

    print(
        "No model training or embedding extraction "
        "was performed."
    )


# ENTRY POINT

if __name__ == "__main__":
    main()