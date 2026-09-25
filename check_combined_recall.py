import duckdb
import pandas as pd

print("Checking combined recall of CH1-CH5 on Fold D...")
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

channels = ["ch1", "ch2", "ch3", "ch4", "ch5"]
selects = []
for ch in channels:
    selects.append(f"""
        SELECT s1_id, target_id, '{ch}' as ch
        FROM read_csv_auto('artifacts/_tmp_channels/{ch}.tsv', delim='\\t')
        WHERE s1_id IN (SELECT s1_id FROM gt_d)
    """)

union_sql = " UNION ALL ".join(selects)

res = con.execute(f"""
    WITH all_cands AS ({union_sql}),
    found AS (
        SELECT DISTINCT s1_id, target_id FROM all_cands
    )
    SELECT COUNT(*) 
    FROM gt_d g
    JOIN found f ON g.s1_id = f.s1_id AND g.target_id = f.target_id
""").fetchone()[0]

print(f"Total True Pairs Found by CH1-CH5: {res:,} / {len(gt_d):,} ({res / len(gt_d):.4%})")
