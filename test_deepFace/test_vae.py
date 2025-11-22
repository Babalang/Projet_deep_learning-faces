import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
from glob import glob
from tensorflow.keras.models import load_model
from modules.VAE import (
    preprocess_face_for_vae,
    reconstruct_all_emotions,
    EMOTION_LABELS,
    FiLM
)

def pick_fallback_image(data_root: str = "./imgs_db/train"):
    for emo in EMOTION_LABELS:
        files = glob(os.path.join(data_root, emo, "*"))
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                return f
    raise RuntimeError("Aucune image disponible pour test.")

def visualize_emotion_changes(img_path: str, encoder_path: str, decoder_path: str, out_dir: str = "./vae_results"):
    print(f"Chargement modèles: {encoder_path}, {decoder_path}")
    encoder = load_model(encoder_path, compile=False, custom_objects={"FiLM": FiLM})
    decoder = load_model(decoder_path, compile=False, custom_objects={"FiLM": FiLM})

    if not os.path.isfile(img_path):
        print(f"Image introuvable: {img_path}, fallback dataset.")
        img_path = pick_fallback_image()
        print("Fallback:", img_path)

    img = preprocess_face_for_vae(img_path)
    recon_list = reconstruct_all_emotions(img, encoder, decoder)

    os.makedirs(out_dir, exist_ok=True)
    print("Sauvegarde des reconstructions...")
    for emo, arr in zip(EMOTION_LABELS, recon_list):
        save_path = os.path.join(out_dir, f"{emo}.png")
        cv2.imwrite(save_path, arr)
        print(" ->", save_path)

    tiles = [cv2.resize(r, (96, 96), interpolation=cv2.INTER_NEAREST) for r in recon_list]
    grid = cv2.hconcat(tiles)
    cv2.imwrite(os.path.join(out_dir, "grid.png"), grid)
    print("Grille:", os.path.join(out_dir, "grid.png"))

def main():
    img_path = "./imgs_db/happy.png"  # ou laisser pour fallback
    encoder_path = "./models/vae/cvae_encoder.h5"
    decoder_path = "./models/vae/cvae_decoder.h5"
    visualize_emotion_changes(img_path, encoder_path, decoder_path)

if __name__ == "__main__":
    main()