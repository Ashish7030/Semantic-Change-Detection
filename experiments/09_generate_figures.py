from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# =============================================================================
# PROJECT PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

FIGURE_DPI = 300


# =============================================================================
# LANGUAGES
# =============================================================================

LANGUAGES = [
    "English",
    "German",
    "Latin",
    "Swedish",
]

LANGUAGE_KEYS = {
    "English": "english",
    "German": "german",
    "Latin": "latin",
    "Swedish": "swedish",
}

TOTAL_TARGETS = {
    "English": 37,
    "German": 48,
    "Latin": 40,
    "Swedish": 31,
}


# =============================================================================
# METHODS
# =============================================================================

METHODS = [
    ("Word2Vec + Procrustes", "word2vec"),
    ("BERT + Averaging", "bert_averaging"),
    ("Word2Vec + BERT", "word2vec_bert"),
    ("RoBERTa + K-Means + JSD", "roberta_kmeans_jsd"),
    ("RoBERTa + AMD", "roberta_amd"),
    ("RoBERTa + SAMD", "roberta_samd"),
]

METHOD_NAMES = [
    method_name
    for method_name, _ in METHODS
]


# =============================================================================
# DIRECTORY SETUP
# =============================================================================

def ensure_output_directory():
    """Create the results directory if necessary."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# RAW RESULT FILE DISCOVERY
# =============================================================================

def find_raw_result_file(prefix, language):
    """
    Find the raw experiment CSV anywhere inside the project.
    """

    language_key = LANGUAGE_KEYS[language]

    filename = f"{prefix}_{language_key}.csv"

    # First check the standard results directory.
    standard_path = RESULTS_DIR / filename

    if standard_path.exists():
        return standard_path

    # Otherwise search the complete project.
    matches = [
        path
        for path in BASE_DIR.rglob(filename)
        if path.is_file()
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print(f"\nMultiple files found for {filename}:")

        for path in matches:
            print(
                f"  {path.relative_to(BASE_DIR)}"
            )

        raise RuntimeError(
            f"Multiple copies of '{filename}' were found. "
            "Please keep only one raw result file."
        )

    # Print available CSVs to make debugging easy.
    print(
        f"\nCould not find required file: {filename}"
    )

    print(
        "\nCSV files currently found in the project:"
    )

    csv_files = sorted(
        path
        for path in BASE_DIR.rglob("*.csv")
        if path.is_file()
    )

    if csv_files:
        for path in csv_files:
            print(
                f"  {path.relative_to(BASE_DIR)}"
            )
    else:
        print("  No CSV files found.")

    raise FileNotFoundError(
        f"\nRequired result file not found: {filename}"
    )


# =============================================================================
# LOAD RESULT FILE
# =============================================================================

def load_result_file(prefix, language):
    """Load one raw experiment result file."""

    path = find_raw_result_file(
        prefix,
        language,
    )

    print(
        f"    {path.relative_to(BASE_DIR)}"
    )

    return pd.read_csv(path)


# =============================================================================
# SCORE COLUMN
# =============================================================================

def get_score_column(df):
    """
    Detect the prediction score column.

    Standard methods use 'score'.
    Word2Vec + BERT uses 'combined_score'.
    """

    if "score" in df.columns:
        return "score"

    if "combined_score" in df.columns:
        return "combined_score"

    raise ValueError(
        "No prediction score column found. "
        "Expected 'score' or 'combined_score'."
    )


# =============================================================================
# STATISTICS
# =============================================================================

def calculate_statistics(df):
    """
    Calculate Spearman rho, p-value, and valid target count.
    """

    score_column = get_score_column(df)

    if "graded_gold" not in df.columns:
        raise ValueError(
            "Column 'graded_gold' is missing."
        )

    valid = df[
        [
            score_column,
            "graded_gold",
        ]
    ].dropna()

    valid_targets = len(valid)

    if valid_targets < 3:
        return np.nan, np.nan, valid_targets

    rho, p_value = spearmanr(
        valid[score_column],
        valid["graded_gold"],
    )

    return (
        float(rho),
        float(p_value),
        int(valid_targets),
    )


# =============================================================================
# COLLECT ALL RESULTS
# =============================================================================

def collect_results():
    """
    Evaluate all six approaches on all four languages.
    """

    records = []

    print("\nReading raw experiment results...")
    print("-" * 70)

    for method_name, prefix in METHODS:

        print(f"\n{method_name}")

        for language in LANGUAGES:

            df = load_result_file(
                prefix,
                language,
            )

            rho, p_value, valid_targets = (
                calculate_statistics(df)
            )

            records.append(
                {
                    "Method": method_name,
                    "Language": language,
                    "rho": rho,
                    "p_value": p_value,
                    "valid_targets": valid_targets,
                }
            )

    return pd.DataFrame(records)


# =============================================================================
# TABLE 1
# =============================================================================

def create_table_1(stats_df):
    """
    Create Table 1:

    Spearman correlation for every approach and language,
    including the macro-average.
    """

    table = stats_df.pivot(
        index="Method",
        columns="Language",
        values="rho",
    )

    table = table.reindex(
        index=METHOD_NAMES,
        columns=LANGUAGES,
    )

    # Macro-average across the four language correlations.
    table["Macro-average"] = table[
        LANGUAGES
    ].mean(axis=1)

    # Format for paper.
    formatted = table.copy()

    for column in formatted.columns:
        formatted[column] = formatted[column].map(
            lambda value:
                "—"
                if pd.isna(value)
                else f"{value:.3f}"
        )

    output_path = (
        RESULTS_DIR /
        "table_1_results.csv"
    )

    formatted.to_csv(
        output_path
    )

    print(
        f"\n✓ Table 1 saved to:\n"
        f"  {output_path}"
    )


# =============================================================================
# TABLE 2
# =============================================================================

def create_table_2(stats_df):
    """
    Create Table 2:

    Number of valid targets / total targets and percentage coverage.
    """

    rows = []

    for method_name in METHOD_NAMES:

        method_data = stats_df[
            stats_df["Method"] == method_name
        ]

        row = {
            "Method": method_name
        }

        for language in LANGUAGES:

            result = method_data[
                method_data["Language"] == language
            ].iloc[0]

            valid_targets = int(
                result["valid_targets"]
            )

            total_targets = TOTAL_TARGETS[
                language
            ]

            percentage = (
                valid_targets /
                total_targets *
                100
            )

            row[language] = (
                f"{valid_targets}/"
                f"{total_targets}"
                f" ({percentage:.1f}%)"
            )

        rows.append(row)

    table = pd.DataFrame(
        rows
    )

    table = table.set_index(
        "Method"
    )

    table = table.reindex(
        index=METHOD_NAMES,
        columns=LANGUAGES,
    )

    output_path = (
        RESULTS_DIR /
        "table_2_target_coverage.csv"
    )

    table.to_csv(
        output_path
    )

    print(
        f"\n✓ Table 2 saved to:\n"
        f"  {output_path}"
    )


# =============================================================================
# FIGURE 1
# =============================================================================

def create_figure_1(stats_df):
    """
    Create Figure 1:

    Language-wise Spearman correlation comparison.
    """

    data = stats_df.pivot(
        index="Language",
        columns="Method",
        values="rho",
    )

    data = data.reindex(
        index=LANGUAGES,
        columns=METHOD_NAMES,
    )

    fig, ax = plt.subplots(
        figsize=(13, 7.5)
    )

    x = np.arange(
        len(LANGUAGES)
    )

    number_of_methods = len(
        METHOD_NAMES
    )

    total_width = 0.72

    bar_width = (
        total_width /
        number_of_methods
    )

    offsets = (
        np.arange(number_of_methods)
        - (number_of_methods - 1) / 2
    ) * bar_width

    # -------------------------------------------------------------------------
    # Bars
    # -------------------------------------------------------------------------

    for i, method in enumerate(
        METHOD_NAMES
    ):

        values = data[
            method
        ].values

        bars = ax.bar(
            x + offsets[i],
            values,
            width=bar_width * 0.92,
            label=method,
        )

        # Add rho value above/below each bar.
        for bar, value in zip(
            bars,
            values,
        ):

            if np.isnan(value):
                continue

            if value >= 0:

                y = value + 0.018
                alignment = "bottom"

            else:

                y = value - 0.018
                alignment = "top"

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                y,
                f"{value:.2f}",
                ha="center",
                va=alignment,
                fontsize=9,
            )

    # -------------------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------------------

    ax.axhline(
        0,
        linewidth=0.9,
    )

    ax.set_title(
        "Spearman Correlation by Language and Approach",
        fontsize=17,
        pad=14,
    )

    ax.set_xlabel(
        "Language",
        fontsize=13,
    )

    ax.set_ylabel(
        "Spearman's ρ",
        fontsize=13,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        LANGUAGES,
        fontsize=12,
    )

    ax.set_ylim(
        -0.20,
        0.85,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        linewidth=0.8,
    )

    ax.set_axisbelow(
        True
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.12,
        ),
        ncol=3,
        frameon=False,
        fontsize=10.5,
    )

    fig.tight_layout()

    output_path = (
        RESULTS_DIR /
        "figure_1_language_method_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\n✓ Figure 1 saved to:\n"
        f"  {output_path}"
    )


# =============================================================================
# P-VALUE FORMATTING
# =============================================================================

def format_p_value(value):
    """
    Format p-values for Figure 2.

    Very small values are not displayed as 0.000.
    """

    if pd.isna(value):
        return "—"

    if value < 0.000001:
        return "<0.000001"

    if value < 0.001:
        return "<0.001"

    return f"{value:.3f}"


# =============================================================================
# FIGURE 2
# =============================================================================

def create_figure_2(stats_df):
    """
    Create Figure 2:

    Heatmap of Spearman correlation p-values.

    Color intensity:
        -log10(p)

    Text:
        actual p-value
    """

    pvalue_matrix = stats_df.pivot(
        index="Method",
        columns="Language",
        values="p_value",
    )

    pvalue_matrix = pvalue_matrix.reindex(
        index=METHOD_NAMES,
        columns=LANGUAGES,
    )

    numeric_matrix = (
        pvalue_matrix.astype(float)
    )

    # Prevent log10(0).
    safe_matrix = numeric_matrix.clip(
        lower=np.finfo(float).tiny
    )

    visual_matrix = (
        -np.log10(
            safe_matrix
        )
    )

    fig, ax = plt.subplots(
        figsize=(12, 7.5)
    )

    image = ax.imshow(
        visual_matrix.values,
        aspect="auto",
        cmap="viridis",
    )

    # -------------------------------------------------------------------------
    # Axis labels
    # -------------------------------------------------------------------------

    ax.set_xticks(
        np.arange(
            len(LANGUAGES)
        )
    )

    ax.set_xticklabels(
        LANGUAGES,
        fontsize=12,
    )

    ax.set_yticks(
        np.arange(
            len(METHOD_NAMES)
        )
    )

    ax.set_yticklabels(
        METHOD_NAMES,
        fontsize=11.5,
    )

    ax.set_xlabel(
        "Language",
        fontsize=13,
    )

    ax.set_ylabel(
        "Approach",
        fontsize=13,
    )

    ax.set_title(
        "Spearman Correlation p-values",
        fontsize=17,
        pad=14,
    )

    # -------------------------------------------------------------------------
    # Cell borders
    # -------------------------------------------------------------------------

    ax.set_xticks(
        np.arange(
            -0.5,
            len(LANGUAGES),
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(METHOD_NAMES),
            1,
        ),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=1.2,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # -------------------------------------------------------------------------
    # p-value annotations
    # -------------------------------------------------------------------------

    for row in range(
        len(METHOD_NAMES)
    ):

        for column in range(
            len(LANGUAGES)
        ):

            value = (
                numeric_matrix.iloc[
                    row,
                    column
                ]
            )

            ax.text(
                column,
                row,
                format_p_value(
                    value
                ),
                ha="center",
                va="center",
                fontsize=10.5,
                color="black",
            )

    # -------------------------------------------------------------------------
    # Colorbar
    # -------------------------------------------------------------------------

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.04,
    )

    colorbar.set_label(
        r"$-\log_{10}(p)$",
        fontsize=12,
    )

    fig.tight_layout()

    output_path = (
        RESULTS_DIR /
        "figure_2_pvalue_heatmap.png"
    )

    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\n✓ Figure 2 saved to:\n"
        f"  {output_path}"
    )


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

def print_summary(stats_df):
    """
    Print the final Spearman correlations.
    """

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINAL SPEARMAN CORRELATIONS"
    )

    print(
        "=" * 90
    )

    print(
        f"{'Approach':<30}"
        f"{'English':>12}"
        f"{'German':>12}"
        f"{'Latin':>12}"
        f"{'Swedish':>12}"
    )

    print(
        "-" * 90
    )

    for method in METHOD_NAMES:

        values = []

        for language in LANGUAGES:

            row = stats_df[
                (
                    stats_df["Method"]
                    == method
                )
                &
                (
                    stats_df["Language"]
                    == language
                )
            ].iloc[0]

            values.append(
                row["rho"]
            )

        print(
            f"{method:<30}"
            f"{values[0]:>12.4f}"
            f"{values[1]:>12.4f}"
            f"{values[2]:>12.4f}"
            f"{values[3]:>12.4f}"
        )

    print(
        "=" * 90
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 70
    )

    print(
        "Generating final LSCD tables and figures"
    )

    print(
        "=" * 70
    )

    print(
        f"\nProject directory:"
    )

    print(
        f"    {BASE_DIR}"
    )

    print(
        f"\nOutput directory:"
    )

    print(
        f"    {RESULTS_DIR}"
    )

    ensure_output_directory()

    # -------------------------------------------------------------------------
    # Collect statistics
    # -------------------------------------------------------------------------

    stats_df = collect_results()

    # -------------------------------------------------------------------------
    # Generate exactly TWO tables
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "-" * 70
    )

    print(
        "Generating final paper tables..."
    )

    create_table_1(
        stats_df
    )

    create_table_2(
        stats_df
    )

    # -------------------------------------------------------------------------
    # Generate exactly TWO figures
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "-" * 70
    )

    print(
        "Generating final paper figures..."
    )

    create_figure_1(
        stats_df
    )

    create_figure_2(
        stats_df
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print_summary(
        stats_df
    )

    # -------------------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\nFinal paper outputs:"
    )

    print(
        "    1. table_1_results.csv"
    )

    print(
        "    2. table_2_target_coverage.csv"
    )

    print(
        "    3. figure_1_language_method_comparison.png"
    )

    print(
        "    4. figure_2_pvalue_heatmap.png"
    )

    print(
        "\nOnly these four artifacts are generated for the paper."
    )


if __name__ == "__main__":
    main()