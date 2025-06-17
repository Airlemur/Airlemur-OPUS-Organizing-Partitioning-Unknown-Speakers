# ==============================================================================
# FILE 2: speaker_estimator.py (UPDATED)
# Automatically estimates the optimal number of speakers from the audio files.
# ==============================================================================
import os
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pyannote.audio import Inference
from tqdm import tqdm
from kneed import KneeLocator

# Import settings from the config file
import config


def _find_optimal_k(embeddings):
    """
    Calculates K-Means inertia for a range of K, finds the knee point,
    and returns the optimal number of clusters.
    """
    print("Calculating inertia values to estimate speaker count...")

    K_range = range(config.ESTIMATOR_MIN_SPEAKERS, config.ESTIMATOR_MAX_SPEAKERS + 1, config.ESTIMATOR_STEP)
    inertias = []

    for k in tqdm(K_range, desc="Calculating Inertia"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)

    print("Finding the elbow point (optimal number of speakers)...")
    try:
        kn = KneeLocator(list(K_range), inertias, curve='convex', direction='decreasing')
        optimal_k = kn.knee
        if not optimal_k:
            print("Warning: Could not automatically find an elbow point. Defaulting to minimum.")
            optimal_k = config.ESTIMATOR_MIN_SPEAKERS
    except Exception as e:
        print(f"Warning: KneeLocator failed ({e}). Defaulting to minimum.")
        optimal_k = config.ESTIMATOR_MIN_SPEAKERS

    plt.figure(figsize=(12, 7))
    plt.plot(K_range, inertias, 'bo-')
    plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='r',
               label=f'Estimated Elbow: K={optimal_k}')
    plt.title("Speaker Count Estimation using the Elbow Method")
    plt.xlabel("Number of Speakers (K)")
    plt.ylabel("Inertia (Total Within-Cluster Sum of Squares)")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(config.ESTIMATOR_GRAPH_PATH), exist_ok=True)
    plt.savefig(config.ESTIMATOR_GRAPH_PATH, dpi=150)
    print(f"Speaker estimation graph saved to: {config.ESTIMATOR_GRAPH_PATH}")

    return optimal_k


def estimate_speakers_from_directory(sample_size=500):
    """
    Main function: Reads audio files, extracts embeddings,
    and returns the estimated optimal number of speakers.
    """
    print("\n--- STEP 1: Estimating Speaker Count ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference = Inference(config.MODEL_NAME, device=device, use_auth_token=config.HF_TOKEN)

    try:
        wav_files = [f for f in os.listdir(config.DATA_DIR) if f.endswith(".wav")]
    except FileNotFoundError:
        print(f"ERROR: The directory specified in DATA_DIR was not found: {config.DATA_DIR}")
        return None

    files_to_process = wav_files[:sample_size]
    print(f"Using a sample of {len(files_to_process)} files for estimation...")

    embeddings = []
    for wav_file in tqdm(files_to_process, desc="Extracting Sample Embeddings"):
        path = os.path.join(config.DATA_DIR, wav_file)
        try:
            waveform, _ = torchaudio.load(path)
            if waveform.shape[1] > config.MINIMUM_SAMPLES:
                # Get raw output from the model
                raw_emb = inference(path)

                # *** UPDATED LOGIC: Handle both output types ***
                if isinstance(raw_emb, np.ndarray):
                    # For short files, it's a simple numpy array
                    emb = raw_emb.squeeze()
                else:
                    # For long files, it's a SlidingWindowFeature object.
                    # We take the mean of all window embeddings to get a single vector.
                    emb = np.mean(raw_emb.data, axis=0)

                if not np.isnan(emb).any():
                    embeddings.append(emb)
        except Exception as e:
            print(f"Warning: Could not process {wav_file}, skipping. Error: {e}")
            continue

    if len(embeddings) < config.ESTIMATOR_MIN_SPEAKERS:
        print("ERROR: Not enough valid embeddings found to perform estimation.")
        return None

    embeddings = np.array(embeddings)

    return _find_optimal_k(embeddings)