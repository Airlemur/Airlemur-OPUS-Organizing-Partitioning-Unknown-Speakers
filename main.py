# ==============================================================================
# FILE 4: main.py (NO CHANGE)
# This is the main entry point for the entire project.
# ==============================================================================
import sys
import config  # Verify config can be imported


def run_project():
    """Orchestrates the entire speaker clustering pipeline."""

    # --- STEP 0: Check Environment ---
    print("--- STEP 0: Checking Environment ---")
    try:
        from library_checker import check_libraries
        if not check_libraries():
            sys.exit(1)
    except ImportError:
        print("ERROR: library_checker.py not found. Make sure all scripts are in the same directory.")
        sys.exit(1)

    # --- STEP 1: Estimate Speaker Count ---
    try:
        from speaker_estimator import estimate_speakers_from_directory
        estimated_speakers = estimate_speakers_from_directory(sample_size=1000)

        if estimated_speakers is None:
            print("Could not estimate the number of speakers. Exiting.")
            sys.exit(1)

        print(f"\n===> Estimated optimal number of speakers: {estimated_speakers} <===")

    except ImportError:
        print("ERROR: speaker_estimator.py not found. Make sure all scripts are in the same directory.")
        sys.exit(1)

    # --- STEP 2: Run Main Clustering and Organization ---
    try:
        from cluster_organizer import run_clustering_and_organization
        run_clustering_and_organization(num_speakers=estimated_speakers)

    except ImportError:
        print("ERROR: cluster_organizer.py not found. Make sure all scripts are in the same directory.")
        sys.exit(1)

    print("\n[SUCCESS] All operations completed successfully!")


if __name__ == '__main__':
    run_project()
