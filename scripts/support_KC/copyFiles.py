import pandas as pd
import shutil
from pathlib import Path

# --- paths ---
csv_path = r"E:\TransformerDatasets\NFC_2018\NFC_2018_selected_files.csv"
destination_dir = Path(r"E:\TransformerDatasets\NFC_2018\Full_Year_200RandHrs")

# create destination folder if needed
destination_dir.mkdir(parents=True, exist_ok=True)

# read CSV (header exists)
df = pd.read_csv(csv_path)

# copy files
missing = 0
for file_path in df["filename"]:
    src = Path(file_path)

    if not src.exists():
        print(f"Missing: {src}")
        missing += 1
        continue

    dst = destination_dir / src.name
    shutil.copy2(src, dst)

print(f"Done. Missing files: {missing}")