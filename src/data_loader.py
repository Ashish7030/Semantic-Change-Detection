from pathlib import Path
import gzip

from src.config import (
    LANGUAGES,
    get_language_dir,
    get_corpus_path,
    get_targets_path,
    get_truth_path,
)


def check_file(path: Path) -> bool:
    """Return True if path exists and is a file."""
    return path.exists() and path.is_file()


def load_targets(language: str) -> list[str]:
    """Load target lemmas from targets.txt."""

    path = get_targets_path(language)

    if not check_file(path):
        raise FileNotFoundError(
            f"targets.txt not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        targets = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return targets


def load_truth(language: str, truth_type: str) -> list[str]:
    """
    Load raw lines from a SemEval truth file.

    We intentionally do NOT parse the columns yet.
    First we want to inspect the actual format.
    """

    path = get_truth_path(language, truth_type)

    if not check_file(path):
        raise FileNotFoundError(
            f"{truth_type}.txt not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return lines


def get_dataset_info(language: str) -> dict:
    """Return important information about one language."""

    dataset_dir = get_language_dir(language)

    corpus1 = get_corpus_path(
        language,
        period=1,
        version="lemma"
    )

    corpus2 = get_corpus_path(
        language,
        period=2,
        version="lemma"
    )

    targets_path = get_targets_path(language)
    binary_path = get_truth_path(language, "binary")
    graded_path = get_truth_path(language, "graded")

    targets = load_targets(language)

    return {
        "language": language,
        "dataset_dir": dataset_dir,
        "corpus1": corpus1,
        "corpus2": corpus2,
        "targets": targets_path,
        "binary_truth": binary_path,
        "graded_truth": graded_path,
        "num_targets": len(targets),
        "corpus1_size_mb": round(
            corpus1.stat().st_size / (1024 ** 2),
            2
        ),
        "corpus2_size_mb": round(
            corpus2.stat().st_size / (1024 ** 2),
            2
        ),
    }


def inspect_corpus(
    language: str,
    period: int,
    num_lines: int = 2
):
    """
    Read only a few lines from a compressed corpus.

    This does NOT decompress the entire corpus.
    """

    path = get_corpus_path(
        language,
        period=period,
        version="lemma"
    )

    print(f"\n{language.upper()} - CORPUS {period}")
    print(f"File: {path}")

    with gzip.open(
        path,
        "rt",
        encoding="utf-8"
    ) as f:

        for i in range(num_lines):

            line = f.readline()

            if not line:
                break

            print(f"Line {i + 1}:")
            print(line[:500].strip())


def inspect_truth(language: str):
    """Print the first few lines of both truth files."""

    print(f"\n{'=' * 70}")
    print(f"{language.upper()} - TRUTH FILES")
    print(f"{'=' * 70}")

    for truth_type in ("binary", "graded"):

        lines = load_truth(
            language,
            truth_type
        )

        print(
            f"\n{truth_type.upper()} "
            f"({len(lines)} lines):"
        )

        for line in lines[:5]:
            print(f"  {line}")


def validate_language(language: str) -> bool:
    """
    Validate that all required SemEval files exist.
    """

    print(f"\n{'=' * 70}")
    print(f"Checking: {language.upper()}")
    print(f"{'=' * 70}")

    try:

        info = get_dataset_info(language)

        print("✓ Dataset directory found")
        print(f"  {info['dataset_dir']}")

        print("✓ Corpus 1 found")
        print(f"  {info['corpus1']}")
        print(
            f"  Compressed size: "
            f"{info['corpus1_size_mb']} MB"
        )

        print("✓ Corpus 2 found")
        print(f"  {info['corpus2']}")
        print(
            f"  Compressed size: "
            f"{info['corpus2_size_mb']} MB"
        )

        print("✓ targets.txt found")
        print(
            f"  Number of targets: "
            f"{info['num_targets']}"
        )

        print("✓ binary.txt found")
        print("✓ graded.txt found")

        print("\nFirst target words:")

        targets = load_targets(language)

        for target in targets[:10]:
            print(f"  {target}")

        return True

    except Exception as e:

        print(f"✗ ERROR: {e}")

        return False