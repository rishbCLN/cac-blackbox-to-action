"""Audit-only data staging; does not modify project source.

Copies three bundled CSVs to the paths the loaders actually search. The bundled
German CSV uses an unsupported schema, so use UCI's original symbolic file.
"""
from pathlib import Path
import shutil
import urllib.request
import ssl
import certifi
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "cac-blackbox-to-action"
for dataset, filename in [("adult", "adult.csv"), ("heloc", "heloc_dataset_v1 (1).csv"),
                          ("diabetes", "diabetes.csv")]:
    dest = ROOT / "data/raw" / dataset
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "datasets" / filename, dest / "audit_source.csv")

dest = ROOT / "data/raw/german_credit"
dest.mkdir(parents=True, exist_ok=True)
url = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
archive = Path(__file__).resolve().parent / "evidence/german-original.zip"
if not archive.exists():
    with urllib.request.urlopen(url, context=ssl.create_default_context(cafile=certifi.where()), timeout=60) as response:
        archive.write_bytes(response.read())
import zipfile
with zipfile.ZipFile(archive) as z:
    member = next(n for n in z.namelist() if n.endswith("german.data"))
    with z.open(member) as f:
        df = pd.read_csv(f, sep=r"\s+", header=None,
                         names=[f"Attribute{i}" for i in range(1, 21)] + ["class"])
df.to_csv(dest / "audit_source.csv", index=False)
print("Staged bundled Adult, HELOC, diabetes; original UCI German:", df.shape)
