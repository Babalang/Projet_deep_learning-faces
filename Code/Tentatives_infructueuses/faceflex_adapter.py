"""Adaptateur minimal pour utiliser le VAE PyTorch fourni par `vae_face_flex.py`.

But: ceci est une intégration simple pour tests et inférence CPU.
"""
from pathlib import Path
import numpy as np
from PIL import Image


def pytorch_available():
    try:
        import torch
        return True
    except Exception:
        return False


def load_faceflex_model(device: str = "cpu"):
    """Charge le modèle défini dans `modules/vae_face_flex.py`.

    Returns (model, device) or raises ImportError if torch absent.
    """
    if not pytorch_available():
        raise ImportError("PyTorch n'est pas installé. Installez torch pour utiliser ce backend.")
    import torch
    from modules import vae_face_flex

    model = vae_face_flex.model
    model.eval()
    model.to(device)
    return model, device


def _preprocess_image_torch(path_or_pil, image_size: int = 128):
    """Charge et prépare une image pour le VAE PyTorch (nc=3, taille 128x128 attendu).

    - path_or_pil: chemin ou PIL.Image
    - renvoie un tensor numpy shape (1, 3, H, W) float32 entre 0.0 et 1.0
    """
    if isinstance(path_or_pil, (str, Path)):
        img = Image.open(path_or_pil).convert("RGB")
    else:
        img = path_or_pil.convert("RGB")

    img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    # to CHW
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr


def _postprocess_tensor(tensor_np):
    """tensor_np: numpy array shape (1, C, H, W) in [0,1]
    Retourne PIL.Image
    """
    img = np.clip(tensor_np, 0.0, 1.0)
    img = img.squeeze(0)
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255.0).astype(np.uint8)
    return Image.fromarray(img)


def reconstruct_image_with_faceflex(input_path: str, output_path: str, device: str = "cpu"):
    """Charge l'image `input_path`, la passe dans le VAE PyTorch et sauvegarde le résultat.

    C'est une inférence simple (forward only) utile pour tester l'intégration.
    """
    if not pytorch_available():
        raise ImportError("PyTorch n'est pas installé. Installez torch pour utiliser ce backend.")

    import torch

    model, device = load_faceflex_model(device)
    arr = _preprocess_image_torch(input_path)
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        out, mu, logvar = model(t)

    out_np = out.cpu().numpy()
    pil = _postprocess_tensor(out_np)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path)
    return output_path
