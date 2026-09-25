import os
import pandas as pd
import duckdb

print("Analyzing GT misses on Fold D...")
split = pd.read_csv('artifacts/entity_split.tsv', sep='\t')
fold_d_ids = set(split[split['fold'] == 'D']['source1_entity_id'])
print(f"Fold D S1 IDs: {len(fold_d_ids)}")

gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
gt = gt[gt['matched_entity_ids'].fillna('').str.strip() != '']
gt = gt.assign(tgt=gt['matched_entity_ids'].str.split(',')).explode('tgt')
gt = gt[['source1_entity_id', 'tgt']].rename(columns={'source1_entity_id': 's1_id', 'tgt': 'target_id'})
gt_d = gt[gt['s1_id'].isin(fold_d_ids)]
print(f"Total True Pairs in Fold D: {len(gt_d)}")

# Read existing channels CH1-CH5
ch_files = [f'artifacts/_tmp_channels/ch{i}.tsv' for i in range(1, 6)]

con = duckdb.connect()
con.register('gt_d', gt_d)
select_list = []
for f in ch_files:
    fwd = f.replace('\\', '/')
    select_list.append(f"SELECT s1_id, target_id FROM read_csv_auto('{fwd}', delim='\t') WHERE s1_id IN (SELECT s1_id FROM gt_d)")

unions = ' UNION ALL '.join(select_list)

misses = con.execute(f"""
    WITH found AS (
        SELECT DISTINCT s1_id, target_id FROM ({unions})
    )
    SELECT g.s1_id, g.target_id
    FROM gt_d g
    LEFT JOIN found f ON g.s1_id = f.s1_id AND g.target_id = f.target_id
    WHERE f.s1_id IS NULL
""").df()

print(f"Missed pairs count: {len(misses)} ({len(misses)/len(gt_d)*100:.2f}%)")

# Sample 15 misses and inspect their names and addresses!
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
s23 = pd.concat([s2, s3], ignore_index=True)

sample = misses.head(15).merge(s1, left_on='s1_id', right_on='entity_id')
sample = sample.merge(s23, left_on='target_id', right_on='entity_id', suffixes=('_s1', '_tgt'))
for i, r in sample.iterrows():
    print(f"\n--- Miss #{i+1} ---")
    print(f"S1:  {r['s1_id']} | Name: {r['business_name_s1']!r:35} | Addr: {r['business_address_s1']!r:35} | Country: {r['country_s1']}")
    print(f"TGT: {r['target_id']} | Name: {r['business_name_tgt']!r:35} | Addr: {r['business_address_tgt']!r:35} | Country: {r['country_tgt']}")
