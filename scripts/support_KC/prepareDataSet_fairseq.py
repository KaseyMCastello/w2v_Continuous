#!/usr/bin/env python3
"""
Fairseq-style preprocessing for single channel XWAV files (single-channel).

- XWAVs loaded from CSV list and chunked into segment_length xwavs.
- Labels matched to labels csv with time overlap.
- Makes a hd5 file for each audio segment wav.
- Uses XWAVhdr from Marie for all timing
"""

import h5py
import argparse
import string
import numpy as np
import pandas as pd
import soundfile as sf

from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from AudioStreamDescriptor import XWAVhdr


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-length", default=5.0, type=float, help="Segment length in seconds")
    parser.add_argument("--resample-bool", default=False, type=int, help="Resample audio?")
    parser.add_argument("--resample-rate", default=200000, type=int, help="Resample rate (Hz)")
    parser.add_argument("--xwav-csv", required=True, type=str, help="CSV with column 'xwav_path' listing XWAV files")
    parser.add_argument("--labels-csv", required=True, type=str, help="CSV with label_str,starttime,endtime")
    parser.add_argument("--output-folder", required=True, type=str)
    parser.add_argument("--base-name", default="NFC_2018_200Hr", type=str)
    return parser

# -------------------------
# Load XWAV file list
# -------------------------
def load_xwav_list(csv_path):
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError("XWAV CSV must contain column 'filename'")
    paths = [Path(p) for p in df["filename"]]
    return [p for p in paths if p.exists()]

# -------------------------
# XWAV reading + chunking generator
# -------------------------
def read_and_chunk_xwav_generator(file_path, args):
    xwav = XWAVhdr(str(file_path))
    sr = xwav.xhd["SampleRate"]

    # Stream audio (for huge files)
    waveform, srb = sf.read(file_path)

    if waveform.ndim == 2:
        waveform = waveform[:, 0]  # Single channel

    # Resample if needed
    if args.resample_bool and sr != args.resample_rate:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=args.resample_rate)
        sr = args.resample_rate

    segment_samples = int(args.segment_length * sr)
    buffer_list = []
    sample_ptr = 0
    prev_end_time = None
    raw_times_sec = [(dt - xwav.dtimeStart).total_seconds() for dt in xwav.raw['dnumStart']]

    for i, n_bytes in enumerate([int(b) for b in xwav.xhd['byte_length']]):
        raw_audio = waveform[sample_ptr:sample_ptr + n_bytes]
        sample_ptr += n_bytes
        raw_start_sec = raw_times_sec[i]

        # Handle gaps
        if prev_end_time is not None:
            gap = raw_start_sec - prev_end_time
            if gap > 1:
                n_silent_segments = int(np.floor(gap / args.segment_length))
                for g in range(n_silent_segments):
                    chunk_start_time = xwav.dtimeStart + timedelta(seconds=prev_end_time + g * args.segment_length)
                    yield np.zeros(segment_samples, dtype=waveform.dtype), chunk_start_time

        buffer_list.append(raw_audio)
        total_len = sum(len(b) for b in buffer_list)
        seg_start_time = xwav.dtimeStart + timedelta(seconds=raw_start_sec)

        # Yield full segments only
        while total_len >= segment_samples:
            combined = np.concatenate(buffer_list)
            yield combined[:segment_samples], seg_start_time
            combined = combined[segment_samples:]
            buffer_list = [combined] if len(combined) > 0 else []
            seg_start_time += timedelta(seconds=args.segment_length)
            total_len = sum(len(b) for b in buffer_list)

        prev_end_time = raw_start_sec + n_bytes / sr

    # Do not yield partial final segment → dropped

# -------------------------
# Process single file
# -------------------------
def process_file(file_path, args, labels, label_mapping, wav_dir, h5_dir):
    xwav = XWAVhdr(str(file_path))
    sr = xwav.xhd["SampleRate"]
    segment_count = 0

    for seg_data, seg_start in read_and_chunk_xwav_generator(file_path, args):
        segment_count += 1
        base_name = f"{file_path.stem}_{seg_start.strftime('%Y%m%d_%H%M%S')}"
        wav_path = wav_dir / f"{base_name}.wav"
        sf.write(wav_path, seg_data, sr)

        # Match labels
        chunk_start = seg_start
        chunk_end = seg_start + timedelta(seconds=args.segment_length)
        overlaps = labels[(labels.starttime < chunk_end) & (labels.endtime > chunk_start)].copy()

        h5_path = h5_dir / f"{base_name}.h5"
        with h5py.File(h5_path, "w") as f:
            if len(overlaps) > 0:
                overlaps["StartRelative"] = (overlaps.starttime - chunk_start).dt.total_seconds().clip(0, args.segment_length)
                overlaps["EndRelative"] = (overlaps.endtime - chunk_start).dt.total_seconds().clip(0, args.segment_length)
                overlaps["StartFrame"] = (overlaps["StartRelative"] * sr).astype(int)
                overlaps["EndFrame"] = (overlaps["EndRelative"] * sr).astype(int)
                overlaps["Focal"] = True
                f.create_dataset("start_time_lbl", data=overlaps["StartRelative"])
                f.create_dataset("end_time_lbl", data=overlaps["EndRelative"])
                f.create_dataset("start_frame_lbl", data=overlaps["StartFrame"])
                f.create_dataset("end_frame_lbl", data=overlaps["EndFrame"])
                f.create_dataset("lbl", data=np.array(overlaps["label_str"], dtype="S"))
                f.create_dataset("lbl_cat", data=np.array([label_mapping[n] for n in overlaps["label_str"]], dtype=np.int32))
                f.create_dataset("foc", data=overlaps["Focal"].astype(int))
            else:
                # Empty HDF5
                f.create_dataset("start_time_lbl", data=np.array([]))
                f.create_dataset("end_time_lbl", data=np.array([]))
                f.create_dataset("start_frame_lbl", data=np.array([]))
                f.create_dataset("end_frame_lbl", data=np.array([]))
                f.create_dataset("lbl", data=np.array([], dtype="S"))
                f.create_dataset("lbl_cat", data=np.array([], dtype=np.int32))
                f.create_dataset("foc", data=np.array([], dtype=int))

    print(f"Processed {file_path.stem}, {segment_count} segments")

# -------------------------
# Main
# -------------------------
def main(args):
    # Load files
    files_df = pd.read_csv(args.xwav_csv)
    files = [Path(f) for f in files_df['filename'] if Path(f).exists()]
    print(f"Loaded {len(files)} XWAV files from CSV")

    # Load labels
    labels = pd.read_csv(args.labels_csv)
    labels['starttime'] = labels['starttime'].str.replace(r':(\d{3})$', r'.\1', regex=True)
    labels['endtime'] = labels['endtime'].str.replace(r':(\d{3})$', r'.\1', regex=True)
    labels['starttime'] = pd.to_datetime(labels['starttime'], format='%b-%d-%Y %H:%M:%S.%f')
    labels['endtime'] = pd.to_datetime(labels['endtime'], format='%b-%d-%Y %H:%M:%S.%f')

    # Convert labels to categorical for fast mapping
    unique_labels = list(labels['label_str'].unique())
    label_mapping = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    labels['lbl_cat'] = labels['label_str'].map(label_mapping).astype(np.int32)
    print(f"Found {len(unique_labels)} unique labels")
    print("Label mapping (label_str -> index):") 
    for lbl, idx in label_mapping.items(): 
        print(f" {lbl} -> {idx}")

    # Create output folders once
    wav_dir = Path(args.output_folder) / "wav"
    h5_dir = Path(args.output_folder) / "lbl"
    wav_dir.mkdir(parents=True, exist_ok=True)
    h5_dir.mkdir(parents=True, exist_ok=True)

    # Partial function for multithreading
    partial_func = partial(process_file, args=args, labels=labels, label_mapping=label_mapping,
                           wav_dir=wav_dir, h5_dir=h5_dir)

    import multiprocessing
    num_threads = min(max(1, min(multiprocessing.cpu_count()//2, len(files))), 3)
    print(f"Processing with {num_threads} threads...")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(partial_func, f) for f in files]
        for future in futures:
            future.result()

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)