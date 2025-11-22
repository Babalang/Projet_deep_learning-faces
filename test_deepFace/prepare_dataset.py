import os
import shutil
from pathlib import Path

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def create_folder_structure(base_dir: str = "./imgs_db/train"):
    """Crée la structure de dossiers attendue pour l'entraînement"""
    base = Path(base_dir)
    for emotion in EMOTION_LABELS:
        (base / emotion).mkdir(parents=True, exist_ok=True)
    print(f"✓ Structure créée dans {base_dir}/")
    print("  Placez vos images dans les sous-dossiers correspondants:")
    for emo in EMOTION_LABELS:
        print(f"    - {base_dir}/{emo}/")

if __name__ == "__main__":
    create_folder_structure()