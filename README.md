# Semantic Change Detection

This repository contains the code and experimental results for my Master's term paper on **Lexical Semantic Change Detection (LSCD)**.

The project investigates how different ways of representing and comparing word usages across different time periods affect the detection of semantic change.

The experiments are conducted on the **SemEval-2020 Task 1** benchmark and cover four languages:

- English
- German
- Latin
- Swedish

The project compares static word embeddings, contextual language models, combined representations, clustering-based approaches, and usage-level comparison methods.

---

## Research Question

The main research question of this project is:

> **How does the representation and comparison strategy affect the detection of lexical semantic change?**

The experiments follow a progression from static representations to contextual and usage-level representations:

```text
Static
   ↓
Contextual
   ↓
Static + Contextual
   ↓
Contextual Clustering
   ↓
Usage-Level Comparison