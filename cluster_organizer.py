# ==============================================================================
# FILE 3: cluster_organizer.py (UPDATED)
# This is the main processor for clustering and organizing files.
# ==============================================================================
import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from pyannote.audio import Inference
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

import config


def _get_embedding(inference_model, file_path):
    """Extracts a single embedding from an audio file, handling all cases."""
    try:
        waveform, _ = torchaudio.load(file_path)
        if waveform.shape[1] < config.MINIMUM_SAMPLES:
            return None

        # Get raw output from the model
        raw_emb = inference_model(file_path)

        # *** UPDATED LOGIC: Handle both numpy array and SlidingWindowFeature outputs ***
        if isinstance(raw_emb, np.ndarray):
            # For short files, it's a simple numpy array
            emb = raw_emb.squeeze()
        else:
            # For long files, it's a SlidingWindowFeature object.
            # We take the mean of all window embeddings.
            emb = np.mean(raw_emb.data, axis=0)

        if np.isnan(emb).any():
            return None
        return emb
    except Exception as e:
        print(f"\nWarning: Embedding failed for {os.path.basename(file_path)} | Error: {e}")
        return None


def _draw_clustering_results(embeddings, labels):
    """Visualizes the final clustering results using PCA."""
    print("Visualizing clustering results with PCA...")
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="tab20", s=10, alpha=0.7)
    plt.title("Speaker Clustering Visualization (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Speaker ID")
    plt.tight_layout()

    os.makedirs(os.path.dirname(config.CLUSTERING_GRAPH_PATH), exist_ok=True)
    plt.savefig(config.CLUSTERING_GRAPH_PATH, dpi=300)
    print(f"Clustering result graph saved to: {config.CLUSTERING_GRAPH_PATH}")


def run_clustering_and_organization(num_speakers):
    """
    Main function to perform embedding, clustering, and file organization for all files.
    """
    print(f"\n--- STEP 2: Starting Main Clustering Process for {num_speakers} Speakers ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference = Inference(config.MODEL_NAME, device=device, use_auth_token=config.HF_TOKEN)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    try:
        wav_files = [f for f in os.listdir(config.DATA_DIR) if f.endswith(".wav")]
    except FileNotFoundError:
        print(f"ERROR: The directory specified in DATA_DIR was not found: {config.DATA_DIR}")
        return

    print(f"Found {len(wav_files)} total .wav files to process.")

    all_embeddings = []
    valid_file_paths = []

    start_time = time.time()
    for wav_file in tqdm(wav_files, desc="Processing All Embeddings"):
        path = os.path.join(config.DATA_DIR, wav_file)
        emb = _get_embedding(inference, path)
        if emb is not None:
            all_embeddings.append(emb)
            valid_file_paths.append(path)

    if not all_embeddings:
        print("ERROR: No valid embeddings could be extracted. Halting process.")
        return

    all_embeddings = np.array(all_embeddings)

    print(f"\nEmbedding extraction complete. Found {all_embeddings.shape[0]} valid embeddings.")
    print(f"Starting K-Means clustering for {num_speakers} speakers...")

    kmeans = KMeans(n_clusters=num_speakers, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(all_embeddings)

    print("Copying files into speaker-specific folders...")
    for label, path in tqdm(zip(labels, valid_file_paths), total=len(valid_file_paths), desc="Organizing Files"):
        speaker_folder = os.path.join(config.OUTPUT_DIR, f"speaker_{label:03d}")
        os.makedirs(speaker_folder, exist_ok=True)
        shutil.copy(path, speaker_folder)

        base_name = os.path.splitext(os.path.basename(path))[0]
        txt_path = os.path.join(config.DATA_DIR, f"{base_name}.txt")
        if os.path.exists(txt_path):
            shutil.copy(txt_path, speaker_folder)

    _draw_clustering_results(all_embeddings, labels)

    end_time = time.time()
    total_time = end_time - start_time
    print("\nClustering and organization process finished!")
    print(f"Total time taken: {int(total_time // 3600)}h {int((total_time % 3600) // 60)}m {int(total_time % 60)}s")