from pathlib import Path
import gc
import pickle
import time

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
NORM_DIR = PROJECT_ROOT / "data" / "normalized" / "train"
BLOCKING_DIR = PROJECT_ROOT / "data" / "blocking" / "indexes"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

S2_ROWS = 100_000
S3_ROWS = 100_000
QUERY_ROWS = 2_000

BATCH_SIZE = 32
TOP_KS = [25, 50, 100, 200]

# Keep only the fields that help semantic matching.
# Repeating the business name gives it slightly more weight.
def build_text(df):
    names = df["name_norm"].fillna("").astype(str)
    addresses = df["address_norm"].fillna("").astype(str)
    countries = df["country_norm"].fillna("").astype(str)

    return (
        "business: " + names + " | "
        "business: " + names + " | "
        "address: " + addresses + " | "
        "country: " + countries
    ).tolist()


# ============================================================
# HELPERS
# ============================================================

def load_normalized(path, nrows=None):
    print(f"Loading: {path}")

    usecols = [
        "entity_id",
        "name_norm",
        "address_norm",
        "country_norm",
    ]

    return pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        nrows=nrows,
        dtype=str,
        keep_default_na=False,
    )


def load_validation_queries():
    """
    Load validation S1 records and validation ground truth.
    Only keep entities that actually have at least one match.
    """

    s1_path = NORM_DIR / "train_source1_normalized.tsv"
    gt_path = PROJECT_ROOT / "reports" / "splits" / "validation_ground_truth.tsv"

    s1 = pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    gt = pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    # Ground truth columns:
    # source1_entity_id
    # matched_entity_ids

    gt = gt[gt["matched_entity_ids"].str.len() > 0].copy()

    merged = s1.merge(
        gt,
        left_on="entity_id",
        right_on="source1_entity_id",
        how="inner",
    )

    # Fixed deterministic sample.
    merged = merged.sort_values("entity_id").head(QUERY_ROWS)

    print(f"Validation queries selected: {len(merged):,}")

    return merged


def parse_truth(value):
    if pd.isna(value) or not str(value).strip():
        return set()

    return {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }


def encode_texts(model, texts, batch_size=BATCH_SIZE):
    """
    Encode in small batches and immediately return float32 CPU array.
    """

    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="mps",
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    elapsed = time.time() - start

    print(
        f"Encoded {len(texts):,} records "
        f"in {elapsed:.1f}s "
        f"-> shape={embeddings.shape}"
    )

    return embeddings


# ============================================================
# PHASE 4F BLOCKER LOADING
# ============================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def exact_address_candidates(entity_id, indexes):
    """
    Not used directly for final ANN calculation.
    Kept here so the pilot can later be extended.
    """
    return set()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 4H — SAFE DENSE ANN PILOT")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")

    print("\nSystem:")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available.")

    # --------------------------------------------------------
    # 1. Load corpus
    # --------------------------------------------------------

    print("\n[1/6] Loading corpus...")

    s2_path = NORM_DIR / "train_source2_normalized.tsv"
    s3_path = NORM_DIR / "train_source3_normalized.tsv"

    s2 = load_normalized(s2_path, S2_ROWS)
    s3 = load_normalized(s3_path, S3_ROWS)

    print(f"S2 rows: {len(s2):,}")
    print(f"S3 rows: {len(s3):,}")

    corpus = pd.concat([s2, s3], ignore_index=True)

    corpus_ids = corpus["entity_id"].astype(str).to_numpy()

    print(f"Total ANN corpus: {len(corpus):,}")

    # --------------------------------------------------------
    # 2. Load validation queries
    # --------------------------------------------------------

    print("\n[2/6] Loading validation queries...")

    queries = load_validation_queries()

    query_ids = queries["entity_id"].astype(str).tolist()

    truth = {
        row["entity_id"]: parse_truth(row["matched_entity_ids"])
        for _, row in queries.iterrows()
    }

    # --------------------------------------------------------
    # 3. Load model
    # --------------------------------------------------------

    print("\n[3/6] Loading multilingual embedding model...")

    model_start = time.time()

    model = SentenceTransformer(
        MODEL_NAME,
        device="mps",
    )

    print(
        f"Model loaded in "
        f"{time.time() - model_start:.1f}s"
    )

    # --------------------------------------------------------
    # 4. Encode corpus
    # --------------------------------------------------------

    print("\n[4/6] Encoding ANN corpus...")

    corpus_texts = build_text(corpus)

    corpus_embeddings = encode_texts(
        model,
        corpus_texts,
        batch_size=BATCH_SIZE,
    )

    print(
        f"Corpus embedding memory: "
        f"{corpus_embeddings.nbytes / 1024**2:.1f} MB"
    )

    # We no longer need the dataframe/text list.
    del corpus_texts
    del corpus
    del s2
    del s3
    gc.collect()

    # --------------------------------------------------------
    # 5. Build FAISS index
    # --------------------------------------------------------

    print("\n[5/6] Building FAISS index...")

    dimension = corpus_embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(corpus_embeddings)

    print(f"FAISS dimension: {dimension}")
    print(f"FAISS vectors: {index.ntotal:,}")

    # Corpus embeddings can remain because FAISS IndexFlatIP
    # stores its own copy. Free our Python copy.
    del corpus_embeddings
    gc.collect()

    # --------------------------------------------------------
    # 6. Encode queries and search
    # --------------------------------------------------------

    print("\n[6/6] Encoding validation queries...")

    query_texts = build_text(queries)

    query_embeddings = encode_texts(
        model,
        query_texts,
        batch_size=BATCH_SIZE,
    )

    del query_texts
    gc.collect()

    print("\nSearching ANN index...")

    max_k = max(TOP_KS)

    start = time.time()

    distances, indices = index.search(
        query_embeddings,
        max_k,
    )

    print(
        f"ANN search completed in "
        f"{time.time() - start:.1f}s"
    )

    # --------------------------------------------------------
    # Evaluate ANN recall
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANN RECALL RESULTS")
    print("=" * 70)

    for k in TOP_KS:

        total_truth_links = 0
        recovered_links = 0
        complete_entities = 0

        for i, qid in enumerate(query_ids):

            true_ids = truth[qid]

            if not true_ids:
                continue

            total_truth_links += len(true_ids)

            retrieved = {
                corpus_ids[idx]
                for idx in indices[i, :k]
                if idx >= 0
            }

            recovered = true_ids & retrieved

            recovered_links += len(recovered)

            if true_ids.issubset(retrieved):
                complete_entities += 1

        recall = (
            recovered_links / total_truth_links
            if total_truth_links
            else 0.0
        )

        print(
            f"K={k:3d} | "
            f"Recall={recall:.6%} | "
            f"Recovered={recovered_links:,}/"
            f"{total_truth_links:,} | "
            f"Complete={complete_entities:,}"
        )

    # --------------------------------------------------------
    # Save pilot results
    # --------------------------------------------------------

    output_dir = PROJECT_ROOT / "reports" / "phase4h"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "ann_pilot_results.txt"

    with open(result_path, "w") as f:

        f.write("PHASE 4H — DENSE ANN PILOT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"S2 rows: {S2_ROWS}\n")
        f.write(f"S3 rows: {S3_ROWS}\n")
        f.write(f"Queries: {len(queries)}\n")
        f.write(f"Dimension: {dimension}\n")
        f.write("\n")

        for k in TOP_KS:

            total_truth_links = 0
            recovered_links = 0
            complete_entities = 0

            for i, qid in enumerate(query_ids):

                true_ids = truth[qid]

                if not true_ids:
                    continue

                total_truth_links += len(true_ids)

                retrieved = {
                    corpus_ids[idx]
                    for idx in indices[i, :k]
                    if idx >= 0
                }

                recovered = true_ids & retrieved

                recovered_links += len(recovered)

                if true_ids.issubset(retrieved):
                    complete_entities += 1

            recall = (
                recovered_links / total_truth_links
                if total_truth_links
                else 0.0
            )

            f.write(
                f"K={k}: "
                f"recall={recall:.6%}, "
                f"recovered={recovered_links}, "
                f"truth={total_truth_links}, "
                f"complete={complete_entities}\n"
            )

    print(f"\nResults saved to: {result_path}")

    print("\nPilot completed successfully.")


if __name__ == "__main__":
    main()