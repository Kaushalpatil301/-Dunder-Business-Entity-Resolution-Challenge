import duckdb
import pandas as pd
import re
import time

t0 = time.time()
print("Evaluating new candidate channels on Fold D...")

split = pd.read_csv("artifacts/entity_split.tsv", sep="\t")
fold_d = set(split[split["fold"] == "D"]["source1_entity_id"])

gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")
gt = gt[gt["matched_entity_ids"].fillna("").str.strip() != ""]
gt = gt.assign(tgt=gt["matched_entity_ids"].str.split(",")).explode("tgt")
gt = gt[["source1_entity_id", "tgt"]].rename(columns={"source1_entity_id": "s1_id", "tgt": "target_id"})
gt_d = gt[gt["s1_id"].isin(fold_d)]
n_true = len(gt_d)
print(f"True pairs in fold D: {n_true:,}")

# Load Fold D records from S1 and all S2, S3
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
s1_d = s1[s1["entity_id"].isin(fold_d)].copy()
del s1

s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
s23 = pd.concat([s2, s3], ignore_index=True)
del s2, s3

print(f"Loaded: S1_d={len(s1_d):,}, S2S3={len(s23):,}, elapsed={time.time()-t0:.1f}s")

LEGAL = r'\b(llc|pllc|inc|incorporated|ltd|limited|corp|corporation|pvt|private|co|company|services|service|associates|group|holdings|enterprises)\b'
LEET = str.maketrans('0134578@$', 'oleastbas')

def clean_name(s):
    cl = s.fillna('').str.lower().str.replace(r'[^a-z0-9 ]', ' ', regex=True).str.translate(LEET)
    cl = cl.str.replace(LEGAL, ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
    return cl

print("Normalizing names and addresses...")
s1_d["cname"] = clean_name(s1_d["business_name"])
s23["cname"]  = clean_name(s23["business_name"])

s1_d["csort"] = s1_d["cname"].str.split().apply(lambda w: " ".join(sorted(w)) if w else "")
s23["csort"]  = s23["cname"].str.split().apply(lambda w: " ".join(sorted(w)) if w else "")

def pref2(s):
    return s.str.split().apply(lambda w: " ".join(w[:2]) if len(w) >= 2 else (" ".join(w) if len(w) == 1 and len(w[0]) >= 6 else ""))

s1_d["cpref2"] = pref2(s1_d["cname"])
s23["cpref2"]  = pref2(s23["cname"])

def make_addr_key(s):
    cl = s.fillna('').str.lower().str.replace(r'[^a-z0-9 ]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
    ex = cl.str.extract(r'(\d+)\s+([a-z0-9]{3,})', expand=True)
    num = ex[0].fillna('')
    street = ex[1].fillna('')
    return (num + ' ' + street).str.strip()

s1_d["caddr"] = make_addr_key(s1_d["business_address"])
s23["caddr"]  = make_addr_key(s23["business_address"])

print(f"Normalized in {time.time()-t0:.1f}s. Running DuckDB joins...")

con = duckdb.connect()
con.execute("SET memory_limit='12GB'")
con.execute("SET threads=4")
con.register("s1_d", s1_d)
con.register("s23", s23)
con.register("gt_d", gt_d)

# 1. Exact stripped sorted
t_start = time.time()
con.execute("""
    CREATE TEMP TABLE ch_stripped AS
    SELECT a.entity_id AS s1_id, b.entity_id AS target_id
    FROM s1_d a JOIN s23 b ON a.csort = b.csort
    WHERE length(a.csort) >= 4 AND a.entity_id <> b.entity_id
""")
n_cand_stripped = con.execute("SELECT count(*) FROM ch_stripped").fetchone()[0]
tp_stripped = con.execute("SELECT count(*) FROM gt_d g JOIN ch_stripped c ON g.s1_id = c.s1_id AND g.target_id = c.target_id").fetchone()[0]
print(f"Exact Stripped Sorted: {n_cand_stripped:,} cands, {tp_stripped:,} true pairs ({tp_stripped/n_true:.2%}) in {time.time()-t_start:.1f}s")

# 2. Name prefix 2 words (freq <= 50) + country
t_start = time.time()
con.execute("""
    CREATE TEMP TABLE s23_p2_freq AS
    SELECT cpref2, COUNT(*) as cnt FROM s23 WHERE length(cpref2) >= 8 GROUP BY cpref2 HAVING COUNT(*) <= 50;

    CREATE TEMP TABLE ch_pref2 AS
    SELECT a.entity_id AS s1_id, b.entity_id AS target_id
    FROM s1_d a 
    JOIN s23 b ON a.cpref2 = b.cpref2 AND a.country = b.country
    JOIN s23_p2_freq f ON a.cpref2 = f.cpref2
    WHERE length(a.cpref2) >= 8 AND a.entity_id <> b.entity_id
""")
n_cand_pref2 = con.execute("SELECT count(*) FROM ch_pref2").fetchone()[0]
tp_pref2 = con.execute("SELECT count(*) FROM gt_d g JOIN ch_pref2 c ON g.s1_id = c.s1_id AND g.target_id = c.target_id").fetchone()[0]
print(f"Name Prefix2: {n_cand_pref2:,} cands, {tp_pref2:,} true pairs ({tp_pref2/n_true:.2%}) in {time.time()-t_start:.1f}s")

# 3. Addr key (freq <= 50) + country
t_start = time.time()
con.execute("""
    CREATE TEMP TABLE s23_addr_freq AS
    SELECT caddr, COUNT(*) as cnt FROM s23 WHERE length(caddr) >= 6 GROUP BY caddr HAVING COUNT(*) <= 50;

    CREATE TEMP TABLE ch_addr AS
    SELECT a.entity_id AS s1_id, b.entity_id AS target_id
    FROM s1_d a 
    JOIN s23 b ON a.caddr = b.caddr AND a.country = b.country
    JOIN s23_addr_freq f ON a.caddr = f.caddr
    WHERE length(a.caddr) >= 6 AND a.entity_id <> b.entity_id
""")
n_cand_addr = con.execute("SELECT count(*) FROM ch_addr").fetchone()[0]
tp_addr = con.execute("SELECT count(*) FROM gt_d g JOIN ch_addr c ON g.s1_id = c.s1_id AND g.target_id = c.target_id").fetchone()[0]
print(f"Addr Key: {n_cand_addr:,} cands, {tp_addr:,} true pairs ({tp_addr/n_true:.2%}) in {time.time()-t_start:.1f}s")

# Total recall combined with CH1-CH5 on Fold D:
channels = ["ch1", "ch2", "ch3", "ch4", "ch5"]
selects = [f"SELECT s1_id, target_id FROM read_csv_auto('artifacts/_tmp_channels/{ch}.tsv', delim='\\t') WHERE s1_id IN (SELECT s1_id FROM gt_d)" for ch in channels]
selects.append("SELECT s1_id, target_id FROM ch_stripped")
selects.append("SELECT s1_id, target_id FROM ch_pref2")
selects.append("SELECT s1_id, target_id FROM ch_addr")

union_sql = " UNION ALL ".join(selects)

total_tp = con.execute(f"""
    WITH all_cands AS ({union_sql}),
    found AS (
        SELECT DISTINCT s1_id, target_id FROM all_cands
    )
    SELECT COUNT(*) 
    FROM gt_d g
    JOIN found f ON g.s1_id = f.s1_id AND g.target_id = f.target_id
""").fetchone()[0]

print(f"\n==========================================")
print(f"COMBINED RECALL ON FOLD D: {total_tp:,} / {n_true:,} ({total_tp/n_true:.4%})")
print(f"Total time: {time.time()-t0:.1f}s")
