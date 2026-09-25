"""Remove CH6 channel from master_run.py and validate syntax."""
import pathlib, ast, sys

master = pathlib.Path("master_run.py")
c = master.read_text(encoding="utf-8")

# CH6 block starts at the log.info line and ends at the cleanup for-loop
CH6_START = "    log.info('  CH6 name_prefix"
CH6_END   = "    for f in [db1, db1+\".wal\"]:"

idx_ch6  = c.find(CH6_START)
idx_end  = c.find(CH6_END)

print(f"CH6 start: {idx_ch6}, CH6 end: {idx_end}")

if idx_ch6 > 0 and idx_end > idx_ch6:
    c_new = c[:idx_ch6] + c[idx_end:]
    try:
        ast.parse(c_new)
        master.write_text(c_new, encoding="utf-8")
        print(f"CH6 removed. New size: {len(c_new)} chars. Syntax OK.")
    except SyntaxError as e:
        print(f"SYNTAX ERROR: {e}")
        sys.exit(1)
else:
    # Search line-by-line
    lines = c.splitlines()
    ch6_lines = [(i+1, ln) for i, ln in enumerate(lines) if "CH6" in ln]
    print("CH6 lines found:")
    for no, ln in ch6_lines:
        print(f"  {no}: {ln}")
    sys.exit(1)
