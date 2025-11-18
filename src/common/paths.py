from pathlib import Path

def project_root() -> Path:
    # 2 since this is at src/commmon/path.py
    return Path(__file__).resolve().parents[2]

# Set project root
PROJECT_ROOT = project_root()

# Create processing dirs
SAVE_DIR = PROJECT_ROOT / "data" / "processed"
SAVE_DIR.mkdir(parents=True, exist_ok=True)