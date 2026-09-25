import pathlib

for p in sorted(pathlib.Path("artifacts/_tmp_channels").glob("*.tsv")):
    with open(p, "rb") as f:
        count = sum(1 for _ in f)
    mb = p.stat().st_size / 1e6
    print(f"{p.name:15}: {count:12,d} lines ({mb:8.1f} MB)")
