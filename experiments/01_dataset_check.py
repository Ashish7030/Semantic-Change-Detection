from src.config import LANGUAGES
from src.data_loader import (
    validate_language,
    inspect_corpus,
    inspect_truth,
)


def main():
    print("=" * 70)
    print("SemEval-2020 Task 1 Dataset Verification")
    print("=" * 70)

    results = {}

    # 1. Check all four datasets

    for language in LANGUAGES:
        results[language] = validate_language(language)

    # 2. Summary

    print("\n")
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    for language, success in results.items():
        status = "PASS ✓" if success else "FAIL ✗"
        print(f"{language.capitalize():10s}: {status}")

    # 3. Inspect a few corpus lines

    print("\n")
    print("=" * 70)
    print("CORPUS SAMPLE")
    print("=" * 70)

    for language in LANGUAGES:
        inspect_corpus(language, period=1, num_lines=1)
        inspect_corpus(language, period=2, num_lines=1)

    # 4. Inspect truth-file format

    print("\n")
    print("=" * 70)
    print("TRUTH FILE SAMPLES")
    print("=" * 70)

    for language in LANGUAGES:
        inspect_truth(language)


if __name__ == "__main__":
    main()