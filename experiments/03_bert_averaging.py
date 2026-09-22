import argparse
import csv
import gzip
from pathlib import Path

from src.models.bert_model import (
    MODEL_NAMES,
    extract_target_embeddings,
    load_bert,
)

from src.methods.bert_averaging import (
    semantic_change_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "raw"
)


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


DATASET_NAMES = {
    "english": "semeval2020_ulscd_eng",
    "german": "semeval2020_ulscd_ger",
    "latin": "semeval2020_ulscd_lat",
    "swedish": "semeval2020_ulscd_swe",
}


def get_dataset_dir(language):

    return (
        DATA_DIR
        / language
        / DATASET_NAMES[language]
    )


def get_corpus_paths(language):

    dataset_dir = get_dataset_dir(
        language
    )

    corpus1 = (
        dataset_dir
        / "corpus1"
        / "lemma"
        / CORPUS_FILENAMES[language]["corpus1"]
    )

    corpus2 = (
        dataset_dir
        / "corpus2"
        / "lemma"
        / CORPUS_FILENAMES[language]["corpus2"]
    )

    return corpus1, corpus2


def get_target_file(language):

    return (
        get_dataset_dir(language)
        / "targets.txt"
    )


def get_binary_truth_file(language):

    return (
        get_dataset_dir(language)
        / "truth"
        / "binary.txt"
    )


def get_graded_truth_file(language):

    return (
        get_dataset_dir(language)
        / "truth"
        / "graded.txt"
    )


def read_targets(language):

    targets = []

    with get_target_file(language).open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            target = line.strip()

            if target:
                targets.append(target)

    return targets


def read_truth(path):

    truth = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) >= 2:

                target = parts[0]
                value = parts[1]

                truth[target] = value

    return truth


def decompress_corpus_if_needed(
    gzip_path: Path
):
    """
    Create an uncompressed copy only once.

    The original .gz file is never modified.
    """

    text_path = gzip_path.with_suffix("")

    if text_path.exists():
        return text_path

    print(
        f"\nDecompressing {gzip_path.name}..."
    )

    with gzip.open(
        gzip_path,
        "rt",
        encoding="utf-8",
    ) as source:

        with text_path.open(
            "w",
            encoding="utf-8",
        ) as destination:

            for line in source:
                destination.write(line)

    return text_path


def run_language(language):

    print("\n" + "=" * 70)
    print("BERT + CONTEXTUAL AVERAGING")
    print(
        f"Language: {language.upper()}"
    )
    print("=" * 70)

    corpus1_gz, corpus2_gz = (
        get_corpus_paths(language)
    )

    print("\nCorpus 1:")
    print(corpus1_gz)

    print("\nCorpus 2:")
    print(corpus2_gz)

    if not corpus1_gz.exists():

        raise FileNotFoundError(
            f"Corpus 1 not found:\n{corpus1_gz}"
        )

    if not corpus2_gz.exists():

        raise FileNotFoundError(
            f"Corpus 2 not found:\n{corpus2_gz}"
        )

    targets = read_targets(
        language
    )

    binary_truth = read_truth(
        get_binary_truth_file(language)
    )

    graded_truth = read_truth(
        get_graded_truth_file(language)
    )

    print(
        f"\nNumber of target words: "
        f"{len(targets)}"
    )

    print(
        f"BERT model: "
        f"{MODEL_NAMES[language]}"
    )

    # Decompress only once.

    corpus1 = decompress_corpus_if_needed(
        corpus1_gz
    )

    corpus2 = decompress_corpus_if_needed(
        corpus2_gz
    )

    # Load pretrained BERT.

    tokenizer, model, device = load_bert(
        language
    )

    # Process targets.

    results = []

    print(
        "\nExtracting contextual embeddings..."
    )

    for index, target in enumerate(
        targets,
        start=1,
    ):

        print(
            f"[{index:02d}/{len(targets):02d}] "
            f"{target}"
        )

        period1_embeddings = (
            extract_target_embeddings(
                corpus_path=corpus1,
                target=target,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
        )

        period2_embeddings = (
            extract_target_embeddings(
                corpus_path=corpus2,
                target=target,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
        )

        score = semantic_change_score(
            period1_embeddings,
            period2_embeddings,
        )

        if score is None:

            print(
                "    WARNING: "
                "target missing in one period"
            )

            continue

        result = {
            "target": target,
            "score": score,
            "binary_gold": int(
                binary_truth[target]
            ),
            "graded_gold": float(
                graded_truth[target]
            ),
            "period1_occurrences": len(
                period1_embeddings
            ),
            "period2_occurrences": len(
                period2_embeddings
            ),
        }

        results.append(result)

        print(
            f"    period1="
            f"{len(period1_embeddings):,} "
            f"period2="
            f"{len(period2_embeddings):,} "
            f"score="
            f"{score:.6f}"
        )

    # Save results.

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        RESULTS_DIR
        / f"bert_averaging_{language}.csv"
    )

    with output_file.open(
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
                "period1_occurrences",
                "period2_occurrences",
            ],
        )

        writer.writeheader()

        writer.writerows(results)

    # Summary.

    print("\n" + "=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)

    print(
        f"Successfully calculated scores for "
        f"{len(results)} / {len(targets)} targets."
    )

    print("\nFirst 10 results:")

    for result in results[:10]:

        print(
            f"  {result['target']:<20} "
            f"score={result['score']:.6f} "
            f"binary={result['binary_gold']} "
            f"graded={result['graded_gold']}"
        )

    print("\nResults saved to:")
    print(output_file)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Pretrained BERT + contextual "
            "averaging for SemEval-2020 Task 1"
        )
    )

    parser.add_argument(
        "language",
        choices=[
            "english",
            "german",
            "latin",
            "swedish",
            "all",
        ],
    )

    args = parser.parse_args()

    if args.language == "all":

        for language in [
            "english",
            "german",
            "latin",
            "swedish",
        ]:

            run_language(language)

    else:

        run_language(
            args.language
        )


if __name__ == "__main__":
    main()