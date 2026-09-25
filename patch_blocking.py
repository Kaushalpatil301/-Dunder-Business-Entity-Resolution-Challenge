"""
patch_blocking_v3.py - fixes CH6 OOM by switching ALL channels to DuckDB COPY TO
(writes directly to TSV on disk, zero Python RAM for results).
Also adds SET temp_directory for CH4 disk spill.
"""
import pathlib

REPO   = pathlib.Path(__file__).resolve().parent
master = REPO / "master_run.py"
content = master.read_text(encoding="utf-8")

# Locate _build_blocking_channels start and end
START_MARKER = "def _build_blocking_channels("
END_MARKER   = "def _dedup_union_from_files("

idx_start = content.find(START_MARKER)
idx_end   = content.find(END_MARKER)
assert idx_start > 0 and idx_end > 0
print(f"Replacing _build_blocking_channels: chars {idx_start}..{idx_end}")

NEW_FUNC = r'''def _build_blocking_channels(s1n, s2s3n, tmp, t0, label="train"):
    """
    6-channel blocking using DuckDB COPY TO (zero Python RAM for results).
    CH1 exact_sorted, CH2 exact_expanded_sorted, CH3 token_rare,
    CH4 token_medium (DuckDB file-db + disk spill), CH5 addr_composite,
    CH6 name_prefix (first 10 alphanum chars, min len 12, + country)
    """
    tmp.mkdir(parents=True, exist_ok=True)
    ch_files = []

    def _addr_key(s):
        ex = s.str.extract(r"(?<!\d)(\d{4,})\s+([a-z]{4,})", expand=True)
        return (ex[0].fillna("") + " " + ex[1].fillna("")).str.strip()

    def _copy_query(con, sql, out_path):
        """Execute query and write directly to TSV via DuckDB COPY TO."""
        fwd = str(out_path).replace("\\", "/")
        con.execute(f"COPY ({sql}) TO '{fwd}' (DELIMITER '\t', HEADER true)")
        rows = pd.read_csv(out_path, sep="\t", usecols=["s1_id"]).shape[0]
        log.info("    %s [%s]: %d pairs  (%.1f min)",
                 out_path.stem, label, rows, (time.time()-t0)/60)
        return out_path

    # ── Main DuckDB connection (exact joins: CH1 CH2 CH5 CH6) ───────────────
    db1 = str(tmp / "_ddb1.db")
    for f in [db1, db1+".wal"]:
        if os.path.exists(f): os.remove(f)

    con = duckdb.connect(database=db1)
    con.execute("SET memory_limit='10GB'")
    con.execute(f"SET threads={max(4,(os.cpu_count() or 8)//2)}")
    con.execute("SET preserve_insertion_order=false")
    con.register("s1n",   s1n)
    con.register("s2s3n", s2s3n)

    log.info("  CH1 exact_sorted ...")
    p1 = tmp / "ch1.tsv"
    _copy_query(con, """
        SELECT a.entity_id AS s1_id, b.entity_id AS target_id,
               'exact_sorted' AS channel
        FROM s1n a JOIN s2s3n b ON a.name_sorted = b.name_sorted
        WHERE length(a.name_sorted)>=3 AND a.entity_id<>b.entity_id
    """, p1); ch_files.append(p1)

    log.info("  CH2 exact_expanded_sorted ...")
    p2 = tmp / "ch2.tsv"
    _copy_query(con, """
        SELECT a.entity_id AS s1_id, b.entity_id AS target_id,
               'exact_expanded_sorted' AS channel
        FROM s1n a JOIN s2s3n b ON a.name_expanded_sorted=b.name_expanded_sorted
        WHERE length(a.name_expanded_sorted)>=3 AND a.entity_id<>b.entity_id
    """, p2); ch_files.append(p2)

    log.info("  CH5 addr_composite ...")
    s1n["addr_key"]   = _addr_key(s1n["addr_alphanum"])
    s2s3n["addr_key"] = _addr_key(s2s3n["addr_alphanum"])
    con.register("s1n",s1n); con.register("s2s3n",s2s3n)
    p5 = tmp / "ch5.tsv"
    _copy_query(con, """
        SELECT a.entity_id AS s1_id, b.entity_id AS target_id,
               'addr_composite' AS channel
        FROM s1n a JOIN s2s3n b ON a.addr_key=b.addr_key AND a.country=b.country
        WHERE length(a.addr_key)>=8 AND a.entity_id<>b.entity_id
    """, p5); ch_files.append(p5)

    log.info("  CH6 name_prefix (10 chars, min len 12, country) ...")
    s1n["name_prefix"]   = s1n["name_alphanum"].str[:10].str.strip()
    s2s3n["name_prefix"] = s2s3n["name_alphanum"].str[:10].str.strip()
    con.register("s1n",s1n); con.register("s2s3n",s2s3n)
    p6 = tmp / "ch6.tsv"
    _copy_query(con, """
        SELECT a.entity_id AS s1_id, b.entity_id AS target_id,
               'name_prefix' AS channel
        FROM s1n a JOIN s2s3n b ON a.name_prefix=b.name_prefix AND a.country=b.country
        WHERE length(a.name_prefix)>=10
          AND length(a.name_alphanum)>=12
          AND a.entity_id<>b.entity_id
    """, p6); ch_files.append(p6)
    con.close()

    for f in [db1, db1+".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

    # ── Token tables ────────────────────────────────────────────────────────
    log.info("  Building token frequency table ...")
    s23_tok = (s2s3n[["entity_id","name_alphanum"]]
               .assign(token=s2s3n["name_alphanum"].str.split())
               .explode("token"))
    s23_tok = s23_tok[s23_tok["token"].str.len()>=MIN_TOK_LEN].copy()
    freq = s23_tok.groupby("token")["entity_id"].nunique()
    tok_rare   = s23_tok[s23_tok["token"].isin(freq[freq<=MAX_RARE_FREQ].index)][["token","entity_id"]].drop_duplicates()
    tok_medium = s23_tok[s23_tok["token"].isin(
        freq[(freq>MAX_RARE_FREQ)&(freq<=MAX_MED_FREQ)].index
    )][["token","entity_id"]].drop_duplicates()
    del s23_tok
    log.info("    tok_rare=%d  tok_medium=%d rows", len(tok_rare), len(tok_medium))

    s1_tok = (s1n[["entity_id","name_alphanum"]]
              .assign(token=s1n["name_alphanum"].str.split())
              .explode("token"))
    s1_tok = s1_tok[s1_tok["token"].str.len()>=MIN_TOK_LEN].copy()

    # CH3: token_rare via DuckDB COPY TO
    log.info("  CH3 token_rare ...")
    db3 = str(tmp / "_ddb3.db")
    for f in [db3, db3+".wal"]:
        if os.path.exists(f): os.remove(f)
    con3 = duckdb.connect(database=db3)
    con3.execute("SET memory_limit='10GB'")
    con3.register("s1_tok",s1_tok); con3.register("tok_rare",tok_rare)
    p3 = tmp / "ch3.tsv"
    _copy_query.__func__ = None  # can't use nested _copy_query directly, inline it:
    fwd3 = str(p3).replace("\\","/")
    con3.execute(f"""
        COPY (
            SELECT DISTINCT s.entity_id AS s1_id, t.entity_id AS target_id,
                   'token_rare' AS channel
            FROM s1_tok s JOIN tok_rare t ON s.token=t.token
            WHERE s.entity_id<>t.entity_id
        ) TO '{fwd3}' (DELIMITER '\t', HEADER true)
    """)
    con3.close()
    for f in [db3, db3+".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass
    rows3 = pd.read_csv(p3, sep="\t", usecols=["s1_id"]).shape[0]
    log.info("    ch3 [%s]: %d pairs  (%.1f min)", label, rows3, (time.time()-t0)/60)
    ch_files.append(p3); del tok_rare

    # CH4: token_medium via DuckDB file-based + COPY TO + disk spill
    log.info("  CH4 token_medium (DuckDB file-db + disk spill) ...")
    db4 = str(tmp / "_ddb4.db")
    for f in [db4, db4+".wal"]:
        if os.path.exists(f): os.remove(f)
    tmp_fwd = str(tmp).replace("\\", "/")
    con4 = duckdb.connect(database=db4)
    con4.execute("SET memory_limit='4GB'")   # force aggressive disk spill
    con4.execute("SET threads=4")
    con4.execute("SET preserve_insertion_order=false")
    con4.execute(f"PRAGMA temp_directory='{tmp_fwd}'")
    con4.register("s1_tok",   s1_tok)
    con4.register("tok_medium", tok_medium)
    p4 = tmp / "ch4.tsv"
    fwd4 = str(p4).replace("\\","/")
    con4.execute(f"""
        COPY (
            SELECT s.entity_id AS s1_id, t.entity_id AS target_id,
                   'token_medium' AS channel
            FROM s1_tok s JOIN tok_medium t ON s.token=t.token
            WHERE s.entity_id<>t.entity_id
            GROUP BY s.entity_id, t.entity_id
            HAVING COUNT(*)>={MIN_MED_SH}
        ) TO '{fwd4}' (DELIMITER '\t', HEADER true)
    """)
    con4.close()
    for f in [db4, db4+".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass
    rows4 = pd.read_csv(p4, sep="\t", usecols=["s1_id"]).shape[0]
    log.info("    ch4 [%s]: %d pairs  (%.1f min)", label, rows4, (time.time()-t0)/60)
    ch_files.append(p4); del tok_medium, s1_tok

    return ch_files

'''

new_content = content[:idx_start] + NEW_FUNC + content[idx_end:]
master.write_text(new_content, encoding="utf-8")
print(f"Patched! chars {len(content)} -> {len(new_content)}")

import ast
try:
    ast.parse(new_content)
    print("Syntax: OK")
except SyntaxError as e:
    print(f"Syntax ERROR at line {e.lineno}: {e.msg}")
    # Show surrounding lines
    lines = new_content.splitlines()
    for i in range(max(0,e.lineno-3), min(len(lines),e.lineno+2)):
        print(f"  {i+1}: {lines[i]}")
