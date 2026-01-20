#!/usr/bin/env python3
import h5py
import numpy as np
from pathlib import Path
import umap
import hdbscan
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import librosa

# -------------------------
# CONFIG
# -------------------------
EMB_DIR = Path("/home/kcastello/Code/animal2vec/outputs/inference_embs/hpf_0.5mstok_pretrain")
LBL_DIR = Path("/home/kcastello/Desktop/w2v_data/NFC_2018_Yr_3sChunk/NFC_2018_Yr_3sChunk")
OUTPUT_DIR = Path("/home/kcastello/Code/w2vPlotHelper/embedding_plots_html/hpf_0.5mstoken_pretrain")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
SUBSAMPLE = 5000  # number of points for click/no-click plots
MAX_MARKER_SHAPES = 20
np.random.seed(RANDOM_SEED)

# -------------------------
# HELPERS
# -------------------------
def extract_season(filepath: str):
    """Infer season from month in filename."""
    try:
        fname = Path(filepath).stem  # just the filename without directories
        # filename like NFC_A_03_180101_044501.x_20180101_044911
        parts = fname.split("_")
        date_str = parts[3]  # e.g., 180101
        month = int(date_str[2:4])
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"
    except Exception:
        return "Unknown"

def get_label_file_for_embedding(emb_file: Path, lbl_dir: Path):
    """Match embedding file to correct label file by timestamp."""
    stem = emb_file.stem
    if ".wav_embeddings" in stem:
        wav_base = stem.split(".wav_embeddings")[0]
        lbl_file = lbl_dir / f"{wav_base}.h5"
        if lbl_file.exists():
            return lbl_file
    return None

def load_embeddings_file(h5_path):
    with h5py.File(h5_path, 'r') as f:
        emb = np.array(f['embedding'], dtype=np.float32)
        times = np.array(f['time'], dtype=np.float32)
        filename = f['filename'][()].decode('utf-8')
    return emb, times, filename

def load_label_file(lbl_path):
    with h5py.File(lbl_path, 'r') as f:
        start_time = np.array(f['start_time_lbl'], dtype=np.float64)
        end_time = np.array(f['end_time_lbl'], dtype=np.float64)
        lbl = np.array(f['lbl'], dtype=str)
    print(f"Loaded {len(lbl)} click labels from {lbl_path}")
    return start_time, end_time, lbl

def assign_labels_to_embeddings(emb_times, lbl_start, lbl_end, lbl_vals):
    """Assign labels only if label interval fully falls within the embedding window."""
    labels = np.array(['NoClick']*len(emb_times), dtype=object)
    for s, e, l in zip(lbl_start, lbl_end, lbl_vals):
        # mark embeddings whose interval contains the label interval
        idx = (s >= emb_times) & (e <= emb_times + 1)
        labels[idx] = l
    return labels

def plotly_3d(x, y, z, color=None, shape=None, title="plot", filename="plot.html", subsample=None):
    if subsample is not None and len(x) > subsample:
        idx = np.random.choice(len(x), size=subsample, replace=False)
        x = x[idx]
        y = y[idx]
        z = z[idx]
        if color is not None: color = color[idx]
        if shape is not None: shape = shape[idx]

    df = pd.DataFrame({"x": x, "y": y, "z": z})
    if color is not None:
        df["color"] = color
    if shape is not None:
        df["shape"] = shape.astype(str)

    if shape is not None and color is not None:
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="color", symbol="shape", title=title)
    elif color is not None:
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="color", title=title)
    else:
        fig = px.scatter_3d(df, x="x", y="y", z="z", title=title)

    fig.write_html(filename)
    print(f"Saved interactive plot: {filename}")

# -------------------------
# LOAD EMBEDDINGS + LABELS
# -------------------------
all_embeddings, all_times, all_click_labels, all_species_labels, all_filenames = [], [], [], [], []

for emb_file in EMB_DIR.glob("*.h5"):
    lbl_file = get_label_file_for_embedding(emb_file, LBL_DIR)
    if lbl_file is None or not lbl_file.exists(): 
        print(f"Skipping {emb_file.name}, no label file found")
        continue

    emb, times, filename = load_embeddings_file(emb_file)
    start_time, end_time, lbl_vals = load_label_file(lbl_file)
    click_labels = assign_labels_to_embeddings(times, start_time, end_time, lbl_vals)
    species_labels = click_labels.copy()
    species_labels[click_labels=='NoClick'] = 'NoClick'

    all_embeddings.append(emb)
    all_times.append(times)
    all_click_labels.append(click_labels)
    all_species_labels.append(species_labels)
    all_filenames.append(np.array([filename]*len(emb)))

# Flatten all for UMAP/HDBSCAN
embeddings_all = np.vstack(all_embeddings)


# -------------------------
# PLOT 1: Combined Waveform+Spectrogram+Embedding Heatmap (first 3 files)
# -------------------------
for i in range(min(3,len(all_embeddings))):
    emb = all_embeddings[i]
    times_emb = all_times[i]
    wav_file = Path(all_filenames[i][0])
    filename_stem = wav_file.stem

    # Load waveform & spectrogram
    y, sr = librosa.load(wav_file, sr=None)
    t_wav = np.arange(len(y))/sr
    S = librosa.stft(y, n_fft=512, hop_length=256)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    t_spec = np.arange(S_db.shape[1])*(256/sr)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.2,0.3,0.5],
        vertical_spacing=0.02,
        subplot_titles=("Waveform","Spectrogram","Embedding Heatmap")
    )

    # Waveform
    fig.add_trace(go.Scatter(x=t_wav, y=y, name="Waveform", line=dict(color="black")), row=1, col=1)
    # Spectrogram
    fig.add_trace(go.Heatmap(z=S_db, x=t_spec, y=np.arange(S_db.shape[0]), colorscale="Viridis",
                             colorbar=dict(title="dB")), row=2, col=1)
    # Embedding heatmap
    fig.add_trace(go.Heatmap(z=emb.T, x=times_emb, colorscale="Viridis",
                             colorbar=dict(title="Embedding Value")), row=3, col=1)

    fig.update_layout(
        height=800, width=1200,
        title=f"Waveform + Spectrogram + Embedding: {filename_stem}"
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)

    out_file = OUTPUT_DIR / f"combined_{filename_stem}.html"
    fig.write_html(out_file)
    print(f"Saved combined plot: {out_file}")

# -------------------------
# UMAP 3D
# -------------------------
reducer = umap.UMAP(n_components=3, random_state=RANDOM_SEED)
emb_3d = reducer.fit_transform(embeddings_all)

# HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
cluster_labels = clusterer.fit_predict(emb_3d)

# -------------------------
# 3D PLOTS
# -------------------------
# Click vs NoClick
clicks_all = np.array([
    "Click" if np.any(lbls != "NoClick") else "NoClick"
    for lbls in all_click_labels
])

species_all = np.array([
    lbls[lbls != "NoClick"][0] if np.any(lbls != "NoClick") else "NoClick"
    for lbls in all_species_labels
])

seasons_all = np.array([
    extract_season(fns[0]) for fns in all_filenames
])

click_binary = (clicks_all != "NoClick").astype(str)
plotly_3d(emb_3d[:,0], emb_3d[:,1], emb_3d[:,2], color=click_binary,
          title="Click vs NoClick", filename=OUTPUT_DIR/"3d_clicks.html")

# Season
plotly_3d(emb_3d[:,0], emb_3d[:,1], emb_3d[:,2], color=seasons_all,
          title="Season", filename=OUTPUT_DIR/"3d_season.html")

# Species
plotly_3d(emb_3d[:,0], emb_3d[:,1], emb_3d[:,2], color=species_all,
          title="Species", filename=OUTPUT_DIR/"3d_species.html")

# Cluster + Click
plotly_3d(emb_3d[:,0], emb_3d[:,1], emb_3d[:,2], color=click_binary,
          shape=cluster_labels % MAX_MARKER_SHAPES,  # max 20 shapes
          title="Cluster + Click", filename=OUTPUT_DIR/"3d_cluster_click.html")

# Cluster + Species
plotly_3d(emb_3d[:,0], emb_3d[:,1], emb_3d[:,2], color=species_all,
          shape=cluster_labels % MAX_MARKER_SHAPES,
          title="Cluster + Species", filename=OUTPUT_DIR/"3d_cluster_species.html")

# Cluster + Season
plotly_3d(emb_3d[:,0], emb_3d[:,1], emb_3d[:,2], color=seasons_all,
          shape=cluster_labels % MAX_MARKER_SHAPES,
          title="Cluster + Season", filename=OUTPUT_DIR/"3d_cluster_season.html")

# -------------------------
# Histogram of HDBSCAN clusters with ground-truth labels
# -------------------------
df_hist = pd.DataFrame({"cluster": cluster_labels, "label": clicks_all})
cluster_ids = np.unique(cluster_labels)
hist_data = []
for c in cluster_ids:
    sub = df_hist[df_hist["cluster"]==c]["label"].value_counts()
    hist_data.append(sub)

df_stack = pd.DataFrame(hist_data).fillna(0)
fig = df_stack.plot(kind='bar', stacked=True, figsize=(12,6), colormap='tab20')
fig.figure.savefig(OUTPUT_DIR/"hdbscan_cluster_hist.png")
print(f"Saved HDBSCAN cluster histogram: {OUTPUT_DIR/'hdbscan_cluster_hist.png'}")

print("All plots saved.")
