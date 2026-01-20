# -*- coding: utf-8 -*-
"""
Select xwav files for a year based on detections, using XWAVhdr for exact timestamps.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from AudioStreamDescriptor import XWAVhdr  # import your XWAVhdr class

# -------------------- PARAMETERS --------------------
target_hours = 200
base_folders = ["F:\\", "G:\\", "H:\\"]  # drives to search
year_filter = 2018
input_csv = r"E:\TransformerDatasets\NFC_2018\ID_summary_01Jan2018_to_31Dec2018_NFC.csv"  # path to your detection CSV
output_csv = r"E:\TransformerDatasets\NFC_2018\NFC_2018_selected_files.csv"  # output path

# -------------------- 1. Load detection CSV --------------------
masterTbl = pd.read_csv(input_csv)  # must include 'starttime' column
masterTbl['starttime'] = masterTbl['starttime'].str.replace(r'(\d{2}):(\d{3})$', r'\1.\2', regex=True)
masterTbl['starttime'] = pd.to_datetime( masterTbl['starttime'], format='%b-%d-%Y %H:%M:%S.%f')
year_filter = 2018

print(f"Detections from {year_filter}: {len(masterTbl)} rows")

# -------------------- 2. Build file table --------------------
file_list = []

for base in base_folders:
    for root, dirs, files in os.walk(base):
        xwav_files = [f for f in files if f.endswith(".x.wav")]
        if not xwav_files:
            continue

        # Skip folder if first/last file is outside the target year
        first_hdr = XWAVhdr(os.path.join(root, xwav_files[0]))
        last_hdr  = XWAVhdr(os.path.join(root, xwav_files[-1]))
        if last_hdr.dtimeEnd.year < year_filter or first_hdr.dtimeStart.year > year_filter:
            continue

        for f in xwav_files:
            hdr = XWAVhdr(os.path.join(root, f))
            # Skip files outside target year
            if hdr.dtimeEnd.year < year_filter or hdr.dtimeStart.year > year_filter:
                continue

            file_list.append({
                'filename': hdr.filename,
                'starttime': hdr.dtimeStart,
                'endtime': hdr.dtimeEnd
            })

file_tbl = pd.DataFrame(file_list)
print(f"Found {len(file_tbl)} .x.wav files from {year_filter}")

# -------------------- 3. Map detections to files --------------------
def find_file(row, file_df):
    matched = file_df[(file_df['starttime'] <= row['starttime']) & 
                      (row['starttime'] < file_df['endtime'])]
    if not matched.empty:
        return matched.iloc[0]['filename']
    return np.nan

masterTbl['filename'] = masterTbl.apply(find_file, axis=1, file_df=file_tbl)
masterTbl = masterTbl.dropna(subset=['filename'])
print(f"Detections mapped to files: {len(masterTbl)} rows")

# -------------------- 4. Aggregate detections per file --------------------
agg_tbl = masterTbl.groupby('filename').agg(
    NumDetections=('label_num', 'count'),
    starttime=('starttime', 'min')
).reset_index()
agg_tbl['month'] = agg_tbl['starttime'].dt.month
agg_tbl['weight'] = np.log1p(agg_tbl['NumDetections'])

# -------------------- 5. Month-stratified log-weighted sampling --------------------
hours_per_month = target_hours / 12
minutes_per_month = hours_per_month * 60
selected_list = []

for month in range(1, 13):
    month_files = agg_tbl[agg_tbl['month'] == month].copy()
    cumulative_minutes = 0

    while cumulative_minutes < minutes_per_month and not month_files.empty:
        probs = month_files['weight'] / month_files['weight'].sum()
        chosen_idx = np.random.choice(month_files.index, p=probs)
        chosen_row = month_files.loc[chosen_idx]

        # Estimate duration in minutes from XWAVhdr
        # If your XWAVhdr includes start/end, use it:
        duration_min = (chosen_row['filename'].dtimeEnd - chosen_row['filename'].dtimeStart).total_seconds() / 60 \
                        if hasattr(chosen_row['filename'], 'dtimeEnd') else 37.5

        cumulative_minutes += duration_min
        selected_list.append({
            'filename': chosen_row['filename'],
            'month': month,
            'NumDetections': chosen_row['NumDetections']
        })

        month_files = month_files.drop(chosen_idx)

selected_df = pd.DataFrame(selected_list)
print(f"Selected {len(selected_df)} files")

# -------------------- 6. Save final CSV --------------------
selected_df.to_csv(output_csv, index=False)
print("Saved selected_files.csv")
