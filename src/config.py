from pathlib import Path


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"


# SemEval-2020 Task 1 datasets

LANGUAGES = {
    "english": "semeval2020_ulscd_eng",
    "german": "semeval2020_ulscd_ger",
    "latin": "semeval2020_ulscd_lat",
    "swedish": "semeval2020_ulscd_swe",
}


# Dataset path functions

def get_language_dir(language: str) -> Path:
    """
    Return the root directory of a language dataset.
    """

    if language not in LANGUAGES:
        raise ValueError(
            f"Unknown language: {language}. "
            f"Available languages: {list(LANGUAGES.keys())}"
        )

    return DATA_DIR / language / LANGUAGES[language]


def get_corpus_path(
    language: str,
    period: int,
    version: str = "lemma"
) -> Path:
    """
    Return the corpus file for a language and time period.

    period 1 = older corpus
    period 2 = newer corpus

    version:
        lemma
        token
    """

    if period not in (1, 2):
        raise ValueError("period must be 1 or 2")

    if version not in ("lemma", "token"):
        raise ValueError("version must be 'lemma' or 'token'")

    corpus_dir = (
        get_language_dir(language)
        / f"corpus{period}"
        / version
    )

    gz_files = list(corpus_dir.glob("*.gz"))

    if not gz_files:
        raise FileNotFoundError(
            f"No .gz corpus file found in: {corpus_dir}"
        )

    return gz_files[0]


def get_targets_path(language: str) -> Path:
    """
    Return targets.txt path.
    """

    return get_language_dir(language) / "targets.txt"


def get_truth_path(
    language: str,
    truth_type: str
) -> Path:
    """
    Return gold truth file.

    truth_type:
        binary  -> Subtask 1
        graded  -> Subtask 2
    """

    if truth_type not in ("binary", "graded"):
        raise ValueError(
            "truth_type must be 'binary' or 'graded'"
        )

    return (
        get_language_dir(language)
        / "truth"
        / f"{truth_type}.txt"
    )