import os
import sys
import gzip
import csv
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


# CONFIGURATION

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "raw")

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAME = "FacebookAI/xlm-roberta-base"

K = 5
RANDOM_STATE = 42
N_INIT = 10

MAX_LENGTH = 128

# None = process all occurrences
MAX_OCCURRENCES = None

# LANGUAGE CONFIGURATION

LANGUAGE_CONFIG = {
    "english": {
        "data_dir": os.path.join(
            DATA_DIR, "english", "semeval2020_ulscd_eng"
        ),
        "corpus1": os.path.join(
            DATA_DIR,
            "english",
            "semeval2020_ulscd_eng",
            "corpus1",
            "lemma",
            "ccoha1.txt.gz",
        ),
        "corpus2": os.path.join(
            DATA_DIR,
            "english",
            "semeval2020_ulscd_eng",
            "corpus2",
            "lemma",
            "ccoha2.txt.gz",
        ),
    },

    "german": {
        "data_dir": os.path.join(
            DATA_DIR, "german", "semeval2020_ulscd_ger"
        ),
        "corpus1": os.path.join(
            DATA_DIR,
            "german",
            "semeval2020_ulscd_ger",
            "corpus1",
            "lemma",
            "dta.txt.gz",
        ),
        "corpus2": os.path.join(
            DATA_DIR,
            "german",
            "semeval2020_ulscd_ger",
            "corpus2",
            "lemma",
            "bznd.txt.gz",
        ),
    },

    "latin": {
        "data_dir": os.path.join(
            DATA_DIR, "latin", "semeval2020_ulscd_lat"
        ),
        "corpus1": os.path.join(
            DATA_DIR,
            "latin",
            "semeval2020_ulscd_lat",
            "corpus1",
            "lemma",
            "LatinISE1.txt.gz",
        ),
        "corpus2": os.path.join(
            DATA_DIR,
            "latin",
            "semeval2020_ulscd_lat",
            "corpus2",
            "lemma",
            "LatinISE2.txt.gz",
        ),
    },

    "swedish": {
        "data_dir": os.path.join(
            DATA_DIR, "swedish", "semeval2020_ulscd_swe"
        ),
        "corpus1": os.path.join(
            DATA_DIR,
            "swedish",
            "semeval2020_ulscd_swe",
            "corpus1",
            "lemma",
            "kubhist2a.txt.gz",
        ),
        "corpus2": os.path.join(
            DATA_DIR,
            "swedish",
            "semeval2020_ulscd_swe",
            "corpus2",
            "lemma",
            "kubhist2b.txt.gz",
        ),
    },
}


# DEVICE

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# MODEL

def load_model():
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True
    )

    model.to(DEVICE)
    model.eval()

    print(f"Using device: {DEVICE}")

    return tokenizer, model


# TARGET WORD HANDLING

def get_target_forms(target, language):
    """
    Return possible surface forms for matching a target.

    English targets in the SemEval dataset contain POS suffixes,
    e.g. attack_nn, circle_vb.

    The corpus itself contains the lemma without the POS suffix.
    """

    if language == "english":
        if "_" in target:
            lemma, pos = target.rsplit("_", 1)

            # SemEval English POS tags
            valid_pos = {
                "nn",
                "vb",
                "jj",
                "rb",
                "md",
            }

            if pos in valid_pos:
                return {lemma}

        return {target}

    return {target}


# TARGET MATCHING

def find_target_positions(words, target, language):
    """
    Find positions of the target word in a tokenized sentence.
    """

    target_forms = get_target_forms(target, language)

    positions = []

    for i, word in enumerate(words):
        if word in target_forms:
            positions.append(i)

    return positions


# EMBEDDING EXTRACTION

def extract_target_embedding(
    words,
    target_position,
    tokenizer,
    model
):
    """
    Extract one contextual representation for a target occurrence.

    The final four hidden layers are summed and the subword pieces
    belonging to the target word are averaged.
    """

    if not words:
        return None

    try:
        encoded = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        word_ids = encoded.word_ids(batch_index=0)

        if word_ids is None:
            return None

        # Check whether target position survived truncation
        target_token_indices = [
            i
            for i, word_id in enumerate(word_ids)
            if word_id == target_position
        ]

        if not target_token_indices:
            return None

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = model(**encoded)

        hidden_states = outputs.hidden_states

        # Sum the final four hidden layers
        embedding = (
            hidden_states[-1]
            + hidden_states[-2]
            + hidden_states[-3]
            + hidden_states[-4]
        )

        # Average all subword pieces belonging to the target
        target_vectors = embedding[
            0,
            target_token_indices,
            :
        ]

        target_vector = target_vectors.mean(dim=0)

        return target_vector.detach().cpu().numpy().astype(
            np.float32
        )

    except Exception:
        return None


# CORPUS EMBEDDING EXTRACTION

def extract_embeddings_for_target(
    corpus_path,
    target,
    language,
    tokenizer,
    model,
    period_name
):
    """
    Scan a corpus and extract contextual embeddings for all
    occurrences of the target.
    """

    embeddings = []

    print(
        f"      Extracting {period_name} occurrences...",
        flush=True
    )

    try:
        with gzip.open(
            corpus_path,
            "rt",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            for line_number, line in enumerate(f):

                line = line.strip()

                if not line:
                    continue

                words = line.split()

                positions = find_target_positions(
                    words,
                    target,
                    language
                )

                if not positions:
                    continue

                for position in positions:

                    if (
                        MAX_OCCURRENCES is not None
                        and len(embeddings) >= MAX_OCCURRENCES
                    ):
                        break

                    embedding = extract_target_embedding(
                        words,
                        position,
                        tokenizer,
                        model
                    )

                    if embedding is not None:
                        embeddings.append(embedding)

                if (
                    MAX_OCCURRENCES is not None
                    and len(embeddings) >= MAX_OCCURRENCES
                ):
                    break

    except Exception as e:
        print(
            f"      Error reading corpus: {e}"
        )
        return np.empty((0, 0), dtype=np.float32)

    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)

    return np.vstack(embeddings)


# JSD

def calculate_jsd(labels_period1, labels_period2, k):
    """
    Calculate Jensen-Shannon Divergence between the two
    period-specific cluster distributions.

    scipy.spatial.distance.jensenshannon returns the square root
    of JSD, therefore we square the result.
    """

    counts1 = np.bincount(
        labels_period1,
        minlength=k
    ).astype(np.float64)

    counts2 = np.bincount(
        labels_period2,
        minlength=k
    ).astype(np.float64)

    distribution1 = counts1 / counts1.sum()
    distribution2 = counts2 / counts2.sum()

    js_distance = jensenshannon(
        distribution1,
        distribution2
    )

    js_divergence = js_distance ** 2

    return float(js_divergence)

# TARGET SCORE

def calculate_target_score(
    embeddings_period1,
    embeddings_period2,
    k=K
):
    """
    Cluster both periods jointly and calculate JSD between their
    cluster distributions.

    Returns:
        score, reason
    """

    n1 = len(embeddings_period1)
    n2 = len(embeddings_period2)

    total_samples = n1 + n2

    # K-Means cannot create k clusters if there are fewer than k total samples.

    if total_samples < k:
        return None, (
            f"insufficient samples "
            f"(total={total_samples}, required={k})"
        )

    if n1 == 0 or n2 == 0:
        return None, (
            f"missing period "
            f"(period1={n1}, period2={n2})"
        )

    try:
        all_embeddings = np.vstack([
            embeddings_period1,
            embeddings_period2
        ])

        # Safety check
        if len(all_embeddings) < k:
            return None, (
                f"insufficient samples "
                f"(total={len(all_embeddings)}, required={k})"
            )

        # K-Means

        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT
        )

        labels = kmeans.fit_predict(
            all_embeddings
        )

        labels_period1 = labels[:n1]
        labels_period2 = labels[n1:]

        # JSD

        score = calculate_jsd(
            labels_period1,
            labels_period2,
            k
        )

        return score, None

    except Exception as e:
        return None, str(e)


# LOAD TARGETS

def load_targets(language):
    """
    Load target words from targets.txt.
    """

    target_path = os.path.join(
        LANGUAGE_CONFIG[language]["data_dir"],
        "targets.txt"
    )

    targets = []

    with open(
        target_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            target = line.strip()

            if target:
                targets.append(target)

    return targets


# LOAD GOLD DATA

def load_gold_data(language):
    """
    Load binary and graded SemEval gold data.
    """

    truth_dir = os.path.join(
        LANGUAGE_CONFIG[language]["data_dir"],
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
    graded_gold = {}

    # Binary gold

    with open(
        binary_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) >= 2:
                target = parts[0]
                value = float(parts[1])
                binary_gold[target] = value

    # Graded gold

    with open(
        graded_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) >= 2:
                target = parts[0]
                value = float(parts[1])
                graded_gold[target] = value

    return binary_gold, graded_gold


# PROCESS ONE LANGUAGE

def process_language(
    language,
    tokenizer,
    model
):

    config = LANGUAGE_CONFIG[language]

    print("\n" + "=" * 70)
    print("RoBERTa + K-Means + JSD")
    print("=" * 70)

    print(
        f"Language: {language.upper()}"
    )

    print("\nCorpus 1:")
    print(config["corpus1"])

    print("\nCorpus 2:")
    print(config["corpus2"])

    targets = load_targets(language)

    binary_gold, graded_gold = load_gold_data(
        language
    )

    print(
        f"\nNumber of target words: {len(targets)}"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"K-Means clusters: k={K}"
    )

    # Results

    results = []

    valid_count = 0
    skipped_count = 0

    # Process targets

    for index, target in enumerate(
        targets,
        start=1
    ):

        print(
            f"\n[{index:02d}/{len(targets):02d}] {target}"
        )

        # Period 1

        embeddings_period1 = extract_embeddings_for_target(
            config["corpus1"],
            target,
            language,
            tokenizer,
            model,
            "period 1"
        )

        # Period 2

        embeddings_period2 = extract_embeddings_for_target(
            config["corpus2"],
            target,
            language,
            tokenizer,
            model,
            "period 2"
        )

        n1 = len(embeddings_period1)
        n2 = len(embeddings_period2)

        # Calculate score

        score, reason = calculate_target_score(
            embeddings_period1,
            embeddings_period2,
            K
        )

        if score is None:

            skipped_count += 1

            print(
                f"    period1={n1} "
                f"period2={n2}"
            )

            print(
                f"    Skipped: {reason}"
            )

            continue

        valid_count += 1

        print(
            f"    period1={n1} "
            f"period2={n2} "
            f"JSD={score:.6f}"
        )

        results.append({
            "target": target,
            "score": score,
            "binary_gold": binary_gold.get(
                target,
                ""
            ),
            "graded_gold": graded_gold.get(
                target,
                ""
            ),
            "period1_occurrences": n1,
            "period2_occurrences": n2,
        })

    # SAVE RESULTS

    output_path = os.path.join(
        RESULTS_DIR,
        f"roberta_kmeans_jsd_{language}.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "target",
            "score",
            "binary_gold",
            "graded_gold",
            "period1_occurrences",
            "period2_occurrences",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    # EVALUATION

    valid_results = [
        row for row in results
        if row["graded_gold"] != ""
    ]

    if len(valid_results) >= 2:

        predictions = np.array([
            row["score"]
            for row in valid_results
        ])

        gold = np.array([
            float(row["graded_gold"])
            for row in valid_results
        ])

        correlation, p_value = spearmanr(
            predictions,
            gold
        )

        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)

        print(
            f"Language: {language.upper()}"
        )

        print(
            f"Valid targets: {valid_count}/{len(targets)}"
        )

        print(
            f"Skipped targets: {skipped_count}"
        )

        print(
            f"Spearman rho: {correlation:.4f}"
        )

        print(
            f"p-value: {p_value:.6f}"
        )

    else:

        correlation = np.nan
        p_value = np.nan

        print(
            "\nNot enough valid targets for "
            "Spearman correlation."
        )

    print(
        f"\nResults saved to:\n{output_path}"
    )

    return {
        "language": language,
        "targets": len(targets),
        "valid_targets": valid_count,
        "skipped_targets": skipped_count,
        "spearman": correlation,
        "p_value": p_value,
    }


# MAIN

def main():

    print("=" * 70)
    print("RoBERTa + K-Means + JSD")
    print("=" * 70)

    # Language argument

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python -m experiments.06_roberta_kmeans_jsd "
            "english"
        )

        print(
            "\nAvailable languages:"
        )

        print(
            "english"
        )
        print(
            "german"
        )
        print(
            "latin"
        )
        print(
            "swedish"
        )
        print(
            "all"
        )

        sys.exit(1)

    language = sys.argv[1].lower()

    # Load model once

    tokenizer, model = load_model()

    # Process all languages

    if language == "all":

        summary = []

        for lang in [
            "english",
            "german",
            "latin",
            "swedish"
        ]:

            result = process_language(
                lang,
                tokenizer,
                model
            )

            summary.append(result)

        # Macro average

        valid_correlations = [
            result["spearman"]
            for result in summary
            if not np.isnan(
                result["spearman"]
            )
        ]

        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        for result in summary:

            print(
                f"{result['language'].capitalize():10s} "
                f"targets={result['targets']:3d} "
                f"valid={result['valid_targets']:3d} "
                f"skipped={result['skipped_targets']:3d} "
                f"rho={result['spearman']:.4f}"
            )

        if valid_correlations:

            macro_average = np.mean(
                valid_correlations
            )

            print(
                f"\nMacro-average Spearman: "
                f"{macro_average:.4f}"
            )

    else:

        if language not in LANGUAGE_CONFIG:

            print(
                f"Unknown language: {language}"
            )

            print(
                "Use: english, german, latin, "
                "swedish, or all"
            )

            sys.exit(1)

        process_language(
            language,
            tokenizer,
            model
        )


# ENTRY POINT

if __name__ == "__main__":
    main()