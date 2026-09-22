import gzip
from pathlib import Path
from gensim.models import Word2Vec


class CorpusIterator:
    """Iterates over a whitespace-tokenized corpus."""

    def __init__(self, path: Path):
        self.path = path

    def __iter__(self):
        with gzip.open(self.path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                yield line.split()


def train_word2vec(
    corpus_path: Path,
    vector_size: int = 300,
    window: int = 5,
    min_count: int = 1,
    sg: int = 1,
    negative: int = 5,
    epochs: int = 5,
    seed: int = 42,
):
    """
    Train a Word2Vec Skip-gram model on one SemEval corpus.
    """

    sentences = CorpusIterator(corpus_path)

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        epochs=epochs,
        seed=seed,
        workers=1,
    )

    return model