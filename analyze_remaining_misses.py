import duckdb
import pandas as pd
import time

print("Analyzing remaining misses after 8 channels...")
split = pd.read_csv("artifacts/entity_split.tsv", sep="\t")
fold_d = set(split[split["fold"] == "D"]["source1_entity_id"])

gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")
gt = gt[gt["matched_entity_ids"].fillna("").str.strip() != ""]
gt = gt.assign(tgt=gt["matched_entity_ids"].str.split(",")).explode("tgt")
gt = gt[["source1_entity_id", "tgt"]].rename(columns={"source1_entity_id": "s1_id", "tgt": "target_id"})
gt_d = gt[gt["s1_id"].isin(fold_d)]

# We already have ch_stripped, ch_pref2, ch_addr from logic or can run on sample
# Let's inspect 10 sample misses that failed ALL 8 channels!
channels = ["ch1", "ch2", "ch3", "ch4", "ch5"]
con = duckdb.connect()
con.register("gt_d", gt_d)
selects = [f"SELECT s1_id, target_id FROM read_csv_auto('artifacts/_tmp_channels/{ch}.tsv', delim='\\t') WHERE s1_id IN (SELECT s1_id FROM gt_d)" for ch in channels]

# Let's see what S1 and TGT records look like for remaining misses
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
s23 = pd.concat([s2, s3], ignore_index=True)

# Test TF-IDF on a sample of remaining misses
# If we do TF-IDF on name with char n-grams (3-4) or word n-grams, how many of remaining misses have cosine similarity >= 0.4?
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz

# Sample 100 misses
gt_sample = gt_d.sample(1000, random_state=42).merge(s1, left_on="s1_id", right_on="entity_id")
gt_sample = gt_sample.merge(s23, left_on="target_id", right_on="entity_id", suffixes=("_s1", "_tgt"))

print("Computing similarity metrics on sample GT pairs:")
fuzz_ratios = []
addr_ratios = []
for _, r in gt_sample.iterrows():
    fuzz_ratios.append(fuzz.token_sort_ratio(str(r["business_name_s1"]), str(r["business_name_tgt"])))
    addr_ratios.append(fuzz.token_sort_ratio(str(r["business_address_s1"]), str(r["business_address_tgt"])))

gt_sample["name_ratio"] = fuzz_ratios
gt_sample["addr_ratio"] = addr_ratios

print(f"Name similarity >= 50: {(gt_sample['name_ratio'] >= 50).mean():.2%}")
print(f"Name similarity >= 70: {(gt_sample['name_ratio'] >= 70).mean():.2%}")
print(f"Addr similarity >= 50: {(gt_sample['addr_ratio'] >= 50).mean():.2%}")
print(f"Either Name>=60 OR Addr>=60: {((gt_sample['name_ratio'] >= 60) | (gt_sample['addr_ratio'] >= 60)).mean():.2%}")
print(f"Either Name>=50 OR Addr>=50: {((gt_sample['name_ratio'] >= 50) | (gt_sample['addr_ratio'] >= 50)).mean():.2%}")
