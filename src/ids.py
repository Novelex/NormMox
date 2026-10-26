"""List the FILE_IDs of ABIDE Preprocessed.
Reads data/raw/Phenotypic_V1_0b_preprocessed1.csv and writes data/raw/file_ids.txt, one FILE_ID per line, skipping rows with
'no_filename'. Expected: 1,035 people = 505 autistic (DX_GROUP 1) + 530 controls (DX_GROUP 2), 20 site samples.
Pure Python (csv module).
"""
import csv, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = f"{ROOT}/data/raw/Phenotypic_V1_0b_preprocessed1.csv"
rows = [r for r in csv.DictReader(open(src, newline="")) if r["FILE_ID"] != "no_filename"]
ids = [r["FILE_ID"] for r in rows]
n_asd = sum(r["DX_GROUP"] == "1" for r in rows)
open(f"{ROOT}/data/raw/file_ids.txt", "w").write("\n".join(ids) + "\n")
print(f"people with files: {len(ids)} (expect 1035) | autistic: {n_asd} (expect 505) | controls: {len(ids) - n_asd} (expect 530) "
      f"| sites: {len(set(r['SITE_ID'] for r in rows))} (expect 20)")
if len(ids) != 1035 or len(set(ids)) != 1035:
    sys.exit("STOP: unexpected FILE_ID count")
