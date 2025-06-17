OPUS (Organizing & Partitioning Unknown Speakers)

OPUS is an audio processing tool that automatically analyzes thousands of unlabeled .wav files, identifies the unique speakers, and organizes each speaker's recordings into separate, dedicated folders.
✨ Key Features

    1- Automatic Speaker Count Estimation: Automatically determines the most likely number of speakers in your dataset using the Elbow Method.

    2- Voice Embedding: Generates high-accuracy voice fingerprints from each audio file using the powerful pyannote/embedding model.

    3- Intelligent Clustering: Groups speakers by clustering the voice embeddings with the K-Means algorithm.

    4- Fully Automated Organization: After analysis, it automatically copies the files into neatly organized folders, such as speaker_001, speaker_002, etc.

▶️ Getting Started

    1- Read the wiki.
     
    2- Configure the Project: Open the project_config.py file and set the HF_TOKEN, DATA_DIR (your audio folder), and OUTPUT_DIR (where the results will be saved).

    3- Run the Analysis: Open your terminal, navigate to the project directory, and execute the main script:

    python main.py

The script will handle the rest, and you will find your organized audio files in the specified output directory.
