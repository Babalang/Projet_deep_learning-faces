import os
import cv2
from typing import List, Dict, Any, Tuple

# réduire les logs TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from modules.Encoder import *
from models.demography.Emotion import *
import cv2
import numpy as np
from typing import Any, Dict, List, Union
from tensorflow.keras.models import load_model
from modules.ACGAN import *
from pathlib import Path
BASE = Path(__file__).resolve().parent

def test_analyze_face(img_path: str = BASE/"data/imgs_db/fear.jpg"):
    if not os.path.isfile(img_path):
        print(f"Image not found: {img_path}")
        return

    # appeler analyze_face (utilise detector_backend par défaut 'opencv')
    result = analyze_face(img_path)
    print("Analysis Result:", result)

    image_bgr = cv2.imread(img_path)
    if image_bgr is not None:
        draw_annotations(image_bgr, result)

def draw_annotations(
    img_bgr: "np.ndarray",
    resp_objs: List[Dict[str, Any]],
    box_color: Tuple[int,int,int]=(0,0,0),
    text_color: Tuple[int,int,int]=(255,255,255),
    font_scale: float=0.7,
    thickness: int=2,
    show: bool = True,
    window_name: str = "Result",
    wait_time: int = 0,
    scale: float = 1.6,
) -> "np.ndarray":
    import numpy as np
    out = img_bgr.copy()
    for o in resp_objs:
        fa = o.get("facial_area") or {}
        x, y, w, h = int(fa.get("x", 0)), int(fa.get("y", 0)), int(fa.get("w", 0)), int(fa.get("h", 0))
        if w <= 0 or h <= 0:
            continue
        # rectangle
        cv2.rectangle(out, (x, y), (x + w, y + h), box_color, thickness=2)
        # label lines
        labels = []
        dom_e = o.get("dominant_emotion") or o.get("dominant_emotion", "")
        dom_g = o.get("dominant_gender") or o.get("dominant_gender", "")
        if dom_e:
            labels.append(str(dom_e))
        if dom_g:
            labels.append(str(dom_g))
        label_text = " | ".join(labels) if labels else f"conf:{o.get('face_confidence', 0):.2f}"
        # put text background
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_origin = (x, max(0, y - 6))
        cv2.rectangle(out, (text_origin[0], text_origin[1] - th - 4), (text_origin[0] + tw, text_origin[1]+2), box_color, -1)
        cv2.putText(out, label_text, (text_origin[0], text_origin[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

    if show:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, int(out.shape[1] * scale), int(out.shape[0] * scale))
            cv2.imshow(window_name, out)
            cv2.waitKey(wait_time)
            cv2.destroyWindow(window_name)
        except Exception:
            pass
    return out

def vae_reconstruct(
    img_path: Union[str, np.ndarray, IO[bytes]],
    detector_backend="opencv",
    enforce_detection=True,
    align=True,
    silent=False,
):
    import cv2
    import matplotlib.pyplot as plt

    # 1. On charge les modèles corrects
    encoder_path = BASE / "encoder_model.h5"
    decoder_path = BASE / "decoder_model.h5"
    encoder_weights = BASE / "encoder_weights.h5"
    decoder_weights = BASE / "decoder_weights.h5"
    if not encoder_path.exists() or not decoder_path.exists():
        print(f"[ERR] Models not found: {encoder_path} / {decoder_path}")
        return
    # Try full-model load, fallback to rebuild + load_weights (prefer weights-only files if present)
    try:
        encoder = load_model(str(encoder_path), compile=False)
        decoder = load_model(str(decoder_path), compile=False)
    except Exception as e:
        print(f"[WARN] load_model failed: {e}")
        print("[INFO] Fallback: rebuild architectures and load weights-only.")
        from models.demography.Emotion import load_model_latent, load_model_decoder
        # Rebuild architectures from code
        encoder = load_model_latent(latent_dim=256)
        decoder = load_model_decoder(latent_dim=256)
        # Prefer weights-only files saved with save_weights
        try:
            if encoder_weights.exists():
                encoder.load_weights(str(encoder_weights))
            else:
                encoder.load_weights(str(encoder_path))
            if decoder_weights.exists():
                decoder.load_weights(str(decoder_weights))
            else:
                decoder.load_weights(str(decoder_path))
            print("[INFO] Weights loaded into rebuilt models.")
        except Exception as lw_e:
            print("[ERR] load_weights failed:", lw_e)
            print("Conseil: resauvegarde les poids après entraînement avec model.save_weights(...)")
            raise

    # 2. Extraction visage
    face_obj = extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        grayscale=False,
        align=align,
        max_faces=1,
    )[0]["face"]

    # Conversion en grayscale 48x48x1
    face = cv2.cvtColor((face_obj * 255).astype("uint8"), cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face_norm = face.astype("float32") / 255.0
    face_norm = face_norm[..., np.newaxis]  # (48,48,1)
    x = face_norm[np.newaxis, ...]          # (1,48,48,1)

    # 3. Encode
    mu, logvar = encoder.predict(x)
    epsilon = np.random.randn(*mu.shape)
    z = mu + np.exp(0.5 * logvar) * epsilon

    # 4. Decode pour chaque émotion
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    plt.figure(figsize=(14, 2))
    for i, label in enumerate(range(7)):
        label_arr = np.array([[label]])
        generated = decoder.predict([z, label_arr])[0].squeeze()
        plt.subplot(1, 7, i+1)
        plt.imshow(generated, cmap='gray')
        plt.title(emotion_labels[label], fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    print(" BIENVENUE DANS NOTRE APPLICATION POUR LE TRAITEMENT DES VISAGES AVEC EMOTIONS")
    print("============================================================")
    while True:
        print("Choisissez la fonction à exécuter:")
        print("0. Entrainement")
        print("1. Analyser un visage")
        print("2. Reconstruire un visage avec VAE")
        print("3. Générer un visage avec ACGAN")
        print("4. Quitter")
        choice = input("Entrez le numéro de la fonction à exécuter: ")
        if choice == "0":
            print("Quel entrainement souhaitez vous lancer ?")
            print(" a. Entrainement classificateur d'émotions")
            print(" b. Entrainement VAE pour changement d'émotion")
            print(" c. Entrainement GAN pour génération de visages")
            sub_choice = input("Entrez 'a', 'b' ou 'c': ")
            if sub_choice.lower() == "a":
                train_emotion_model()
            elif sub_choice.lower() == "b":
                vae_train()
            elif sub_choice.lower() == "c":
                display_training()
            else:
                print("Choix invalide.")
        elif choice == "1" or choice.lower() == "analyse":
            img_path = input("Entrez le chemin de l'image à analyser: ")
            if not img_path:
                img_path = "./data/imgs_db/fear.jpg"
            test_analyze_face(img_path)
        elif choice == "2" or choice.lower() == "vae":
            img_path = input("Entrez le chemin de l'image à reconstruire avec VAE: ")
            if not img_path:
                img_path = "./data/imgs_db/neutral.webp"
            vae_reconstruct(img_path)
        elif choice == "3" or choice.lower() == "gan":
            display_gan()
        elif choice == "4" or choice.lower() == "quitter":
            print("Au revoir!")
            break
        else:
            print("Choix invalide.")

if __name__ == "__main__":
    main()
