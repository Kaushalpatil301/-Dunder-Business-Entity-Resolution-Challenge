import duckdb
import pandas as pd

print("Checking exact recall of candidate_pairs_all_folds_flat.tsv on Fold D...")
split = pd.read_csv("artifacts/entity_split.tsv", sep="\t")
fold_d = set(split[split["fold"] == "D"]["source1_entity_id"])

gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")
gt = gt[gt["matched_entity_ids"].fillna("").str.strip() != ""]
gt = gt.assign(tgt=gt["matched_entity_ids"].str.split(",")).explode("tgt")
gt = gt[["source1_entity_id", "tgt"]].rename(columns={"source1_entity_id": "s1_id", "tgt": "target_id"})
gt_d = gt[gt["s1_id"].isin(fold_d)]
print(f"True pairs in fold D: {len(gt_d):,}")

con = duckdb.connect()
con.register("gt_d", gt_d)

found = con.execute("""
    WITH cand AS (
        SELECT s1_id, target_id 
        FROM read_csv_auto('artifacts/candidate_pairs_all_folds_flat.tsv', delim='\\t')
        WHERE s1_id IN (SELECT s1_id FROM gt_d)
    )
    SELECT COUNT(DISTINCT g.s1_id || '__' || g.target_id)
    FROM gt_d g
    JOIN cand c ON g.s1_id = c.s1_id AND g.target_id = c.target_id
""").fetchone()[0]

print(f"Found pairs: {found:,} / {len(gt_d):,} ({found / len(gt_d):.4%})")
