#!/usr/bin/env python3
"""Génère les reconstructions pour toutes les émotions et les assemble côte-à-côte.

Usage:
  python3 scripts/generate_all_emotions.py --checkpoint ./models/faceflex_cvae.pth --input imgs_db/happy.png --output out/happy_all_emotions.png
"""
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default=None)
    p.add_argument('--ngf', type=int, default=64)
    p.add_argument('--ndf', type=int, default=64)
    p.add_argument('--latent', type=int, default=128)
    p.add_argument('--n_emotions', type=int, default=7)
    p.add_argument('--emo_embed_dim', type=int, default=16)
    return p.parse_args()


def tensor_to_pil(tensor_np):
    img = np.clip(tensor_np, 0.0, 1.0)
    img = img.squeeze(0)
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255.0).astype(np.uint8)
    return Image.fromarray(img)


def load_image_as_tensor(path, image_size=64):
    img = Image.open(path).convert('RGB')
    img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = np.expand_dims(arr, 0)  # batch
    return arr


def main():
    args = parse_args()

    import torch
    # ensure project root importable
    _pr = Path(__file__).resolve().parents[1]
    if str(_pr) not in sys.path:
        sys.path.insert(0, str(_pr))

    from modules import vae_face_flex

    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device if device != 'cuda' else 'cuda')

    model = vae_face_flex.VAE(nc=3, ngf=args.ngf, ndf=args.ndf, latent_variable_size=args.latent, n_emotions=args.n_emotions, emo_embed_dim=args.emo_embed_dim)
    model.to(dev)

    ckpt = torch.load(args.checkpoint, map_location=dev)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()

    arr = load_image_as_tensor(args.input, image_size=128)
    t = torch.from_numpy(arr).to(dev)

    imgs = []
    with torch.no_grad():
        for emo in range(args.n_emotions):
            emo_tensor = torch.tensor([emo], dtype=torch.long, device=dev)
            out, mu, logvar = model(t, emo=emo_tensor)
            out_np = out.cpu().numpy()
            pil = tensor_to_pil(out_np)
            imgs.append(pil)

    # assemble side-by-side with labels
    w, h = imgs[0].size
    total_w = w * len(imgs)

    # prepare font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # compute label area height
    if font is not None:
        # compute text height robustly across Pillow versions
        label_h = max(font.getmask(n).size[1] for n in label_names) + 8
    else:
        label_h = 24

    canvas = Image.new('RGB', (total_w, h + label_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for i, im in enumerate(imgs):
        x = i * w
        # paste image below the label area
        canvas.paste(im, (x, label_h))
        # draw label centered
        label = label_names[i] if i < len(label_names) else str(i)
        if font is not None:
            text_w, text_h = font.getmask(label).size
        else:
            text_w, text_h = (len(label) * 6, 12)
        tx = x + (w - text_w) // 2
        ty = (label_h - text_h) // 2
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print('Saved labeled mosaic to', args.output)


if __name__ == '__main__':
    main()
