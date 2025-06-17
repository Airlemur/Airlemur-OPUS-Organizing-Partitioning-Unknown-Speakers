# ==============================================================================
# FILE 2: library_checker.py
# This script checks if all required libraries are installed in the environment.
# ==============================================================================
import sys


def check_libraries():
    """Checks for the presence and version of all required libraries."""
    print("--- Checking required libraries ---")

    libraries = {
        "torch": "torch",
        "torchaudio": "torchaudio",
        "pyannote.audio": "pyannote.audio",
        "sklearn": "scikit-learn",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "tqdm": "tqdm",
        "kneed": "kneed"
    }

    all_found = True

    for lib_import, lib_install in libraries.items():
        try:
            # Dynamically import the library
            module = __import__(lib_import.split('.')[0])
            version = getattr(module, '__version__', 'N/A')
            print(f"[✓] {lib_import} (version: {version}) found.")
        except ImportError:
            print(f"[X] {lib_import} NOT FOUND. Please install it using: pip install {lib_install}")
            all_found = False

    if all_found:
        print("\n[SUCCESS] All required libraries are installed.")
        return True
    else:
        print("\n[ERROR] Some libraries are missing. Please install them before proceeding.")
        return False


if __name__ == '__main__':
    check_libraries()