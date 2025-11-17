import os
import cv2
from typing import List, Dict, Any, Tuple

# réduire les logs TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from modules.Encoder import *
import cv2
import numpy as np
from typing import Any, Dict, List, Union

def test_analyze_face():
    img_path = "./imgs_db/img14.jpg"
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
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    silent: bool = False,
    anti_spoofing: bool = False,
) -> None:
    """
    Encode et décode le premier visage détecté, affiche l'image originale et reconstruite.
    """
    import cv2

    # 1. Encode
    resp = encode(
        img_path=img_path,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        silent=silent,
        anti_spoofing=anti_spoofing,
    )

    if not resp or len(resp) == 0:
        print("Aucun visage détecté.")
        return

    # 2. Decode
    latent = resp[0]["z"]
    reconstructed = decode(latent)

    # 3. Affichage
    # Récupère l'image originale (imagette du visage)
    img_objs = extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        grayscale=False,
        align=align,
        expand_percentage=expand_percentage,
        anti_spoofing=anti_spoofing,
    )
    img_content = img_objs[0]["face"] * 255.0
    img_content = img_content.astype(np.uint8)
    img_gray = cv2.cvtColor(img_content, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (48, 48))

    # Met les deux images côte à côte
    reconstructed = reconstructed.squeeze()
    if reconstructed.shape != img_gray.shape:
        reconstructed = cv2.resize(reconstructed, img_gray.shape[::-1])
    combined = np.hstack([img_gray, reconstructed])

    cv2.imshow("Original (gauche) | Reconstruite (droite)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


def main():
    #train_emotion_model()
    #test_analyze_face()
    vae_reconstruct("./imgs_db/degout.jpg")

if __name__ == "__main__":
    main()
