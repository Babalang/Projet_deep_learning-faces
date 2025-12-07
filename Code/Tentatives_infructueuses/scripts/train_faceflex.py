#!/usr/bin/env python3
"""Entrypoint minimal pour entraîner le VAE de `modules/vae_face_flex.py` sur `imgs_db/train`.

Usage:
  python3 scripts/train_faceflex.py --data_dir ./imgs_db/train --epochs 30 --batch_size 16

Notes:
 - Nécessite PyTorch installé (`pip install torch torchvision`).
 - Le script fait un entraînement CPU par défaut, utilise CUDA si disponible et --device cuda.
 - Sauvegarde le fichier de poids sous `models/faceflex_cvae.pth`.
"""
import argparse
import os
from pathlib import Path
from PIL import Image
import numpy as np
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./imgs_db/train", help="Dossier racine d'images organisé par sous-dossiers (labels)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None, help="Device torch (auto detect if not set): cpu or cuda")
    p.add_argument("--save_path", default="./models/faceflex_cvae.pth")
    return p.parse_args()


# Ensure project root is on sys.path so `modules` package can be imported
# when running the script from the `scripts/` folder or from other CWDs.
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class FacesFolderDataset:
    """Dataset simple: parcourt data_dir/<label>/* et retourne images RGB 64x64 normalisées.
    """
    def __init__(self, root_dir, image_size=64):
        self.root = Path(root_dir)
        self.image_size = image_size
        # collect files with labels (assume structure root/<label>/*)
        self.files = []
        self.labels = []
        self.label_names = []
        for sub in sorted(self.root.iterdir()):
            if sub.is_dir():
                self.label_names.append(sub.name)
        self.label_to_idx = {n: i for i, n in enumerate(self.label_names)}
        for sub in sorted(self.root.iterdir()):
            if sub.is_dir():
                lbl_idx = self.label_to_idx[sub.name]
                for f in sub.iterdir():
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        self.files.append(f)
                        self.labels.append(lbl_idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = Image.open(f).convert('RGB')
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        return arr, self.labels[idx]


def collate_fn(batch):
    import torch
    imgs = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    arr = np.stack(imgs, axis=0)
    imgs_t = torch.from_numpy(arr).float()
    labels_t = torch.tensor(labels, dtype=torch.long)
    return imgs_t, labels_t


def train(data_dir, epochs=30, batch_size=16, lr=1e-3, device='cpu', save_path='./models/faceflex_cvae.pth'):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from modules import vae_face_flex

    # device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    ds = FacesFolderDataset(data_dir, image_size=128)
    if len(ds) == 0:
        raise RuntimeError(f"Aucune image trouvée dans {data_dir}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    num_emotions = len(ds.label_names) if hasattr(ds, 'label_names') else 0
    emo_embed_dim = 16 if num_emotions > 0 else 0

    model = vae_face_flex.VAE(nc=3, ngf=64, ndf=64, latent_variable_size=128, n_emotions=num_emotions, emo_embed_dim=emo_embed_dim)
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # reconstruction loss (pixel-wise) + KL
    bce = nn.BCELoss(reduction='sum')

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        recon_sum = 0.0
        kl_sum = 0.0
        n_samples = 0

        for batch_imgs, batch_labels in loader:
            # batch_imgs shape (B, C, H, W)
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(batch_imgs, emo=batch_labels)

            # recon in [0,1] with sigmoid
            # BCE summed over pixels
            recon_loss = bce(recon, batch_imgs)

            # KL per batch
            # mu, logvar shapes assumed (B, latent)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_loss + kl
            loss.backward()
            optimizer.step()

            bsz = batch_imgs.size(0)
            running_loss += loss.item()
            recon_sum += recon_loss.item()
            kl_sum += kl.item()
            n_samples += bsz

        avg_loss = running_loss / n_samples
        avg_recon = recon_sum / n_samples
        avg_kl = kl_sum / n_samples
        print(f"Epoch {epoch}/{epochs}  avg_loss={avg_loss:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f}")

        # save checkpoint each epoch
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)

    print("Training finished. Model saved to", save_path)


def main():
    args = parse_args()
    device = args.device
    if device is None:
        # auto-detect
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device, save_path=args.save_path)


if __name__ == '__main__':
    main()
