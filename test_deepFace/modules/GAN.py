import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List
import torchvision.utils as vutils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ======================================
# DEVICE SETUP
# ======================================
def get_device():
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

device = get_device()


# ======================================
# DATASET
# ======================================
class FERDataset(Dataset):
    """
    Dataset PyTorch pour FER-2013 (CSV) ou pour dossier structuré par label.
    Retour: (img_tensor, label_int, label_onehot_tensor)
    """
    def __init__(
        self,
        csv_file: Optional[str] = None,
        root_dir: Optional[str] = None,
        from_fer_csv: bool = False,
        image_size: int = 96,
        as_rgb: bool = False,
        transforms_: Optional[transforms.Compose] = None,
        emotion_names: Optional[List[str]] = None
    ):
        assert from_fer_csv ^ (root_dir is not None), "Donner soit csv_file (from_fer_csv=True) soit root_dir."
        self.image_size = image_size
        self.as_rgb = as_rgb

        # default transforms if none provided
        if transforms_ is None:
            if as_rgb:
                norm = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                self.transforms = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    norm
                ])
            else:
                norm = transforms.Normalize(mean=[0.5], std=[0.5])
                self.transforms = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    norm
                ])
        else:
            self.transforms = transforms_

        self.samples = []
        self.emotion_names = emotion_names

        if from_fer_csv:
            assert csv_file is not None
            df = pd.read_csv(csv_file)
            # FER-2013: columns 'emotion' int, 'pixels' string
            if self.emotion_names is None:
                # typical FER-2013 labels 0..6
                self.emotion_names = [str(i) for i in sorted(df['emotion'].unique())]
            for _, row in df.iterrows():
                pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8)
                if pixels.size != 48*48:
                    continue
                img = pixels.reshape(48,48).astype(np.uint8)
                pil = Image.fromarray(img)
                pil = pil.convert('RGB') if as_rgb else pil.convert('L')
                label = int(row['emotion'])
                self.samples.append((pil, label))
        else:
            # root_dir with subfolders per emotion
            root = Path(root_dir)
            folders = sorted([d for d in root.iterdir() if d.is_dir()])
            if self.emotion_names is None:
                self.emotion_names = [f.name for f in folders]
            name_to_idx = {name: idx for idx, name in enumerate(self.emotion_names)}
            for folder in folders:
                label = name_to_idx.get(folder.name, None)
                if label is None:
                    continue
                for f in folder.glob("*"):
                    if not f.is_file():
                        continue
                    if f.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.png'):
                        continue
                    try:
                        # skip zero-byte or corrupted files
                        if f.stat().st_size == 0:
                            continue
                        with Image.open(f) as im:
                            pil = im.convert('RGB') if as_rgb else im.convert('L')
                            self.samples.append((pil.copy(), label))
                    except (UnidentifiedImageError, OSError):
                        # corrupted / unreadable image -> skip
                        continue

        if len(self.samples) == 0:
            raise RuntimeError("Aucun échantillon trouvé dans FERDataset.")
        self.num_classes = len(self.emotion_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pil, label = self.samples[idx]
        img = self.transforms(pil)
        # label one-hot
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        label_vec[label] = 1.0
        return img, label, label_vec

def make_dataloaders(
    dataset: Dataset,
    batch_size: int = 64,
    train_split: float = 0.8,
    num_workers: int = 4,
    seed: int = 42
):
    length = len(dataset)
    train_len = int(length * train_split)
    val_len = length - train_len
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# ======================================
# U-NET GENERATOR
# ======================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])

            x = self.ups[idx + 1](torch.cat((skip, x), dim=1))

        return self.final_conv(x)

class UNetCond(UNet):
    def __init__(self, in_channels=3, out_channels=3, num_classes=7, features=[64, 128, 256, 512]):
        super().__init__(in_channels + num_classes, out_channels, features)
        self.num_classes = num_classes

    def forward(self, x, labels):
        # labels: [B] int tensor
        B, C, H, W = x.shape
        label_onehot = torch.zeros(B, self.num_classes, H, W).to(x.device)
        label_onehot.scatter_(1, labels.view(B, 1, 1, 1).expand(B, 1, H, W), 1)
        x_cond = torch.cat([x, label_onehot], dim=1)
        return super().forward(x_cond)

# ======================================
# DISCRIMINATOR (PATCHGAN)
# ======================================
class Discriminator(nn.Module):
    """
    PatchGAN discriminator (multi-task) :
    - returns patch real/fake logits (rf)
    - returns class logits (cls)
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], num_classes=7):
        super().__init__()

        layers = []
        curr_in = in_channels

        for i, feature in enumerate(features):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(curr_in, feature, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(feature) if i > 0 else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            curr_in = feature
        self.features = nn.Sequential(*layers)
        self.conv_realfake = nn.Conv2d(curr_in, 1, kernel_size=4, padding=1)  # patch real/fake
        self.conv_aux = nn.Conv2d(curr_in, num_classes, kernel_size=4, padding=1)  # class scores (patch -> avg pool)
        self.num_classes = num_classes

        layers.append(nn.Conv2d(curr_in, 1, kernel_size=4, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        f = self.features(x)
        rf = self.conv_realfake(f)
        cls = self.conv_aux(f)
        cls = cls.mean(dim=[2,3])
        return rf, cls


# ======================================
# TRAINING LOOP (GAN)
# ======================================

def _get_num_classes_from_loader(loader):
    ds = loader.dataset
    # handle Subset
    if isinstance(ds, torch.utils.data.Subset):
        return ds.dataset.num_classes
    return ds.num_classes

def train_emotion_gan(G, D, loader, epochs=20, lr=2e-4, lambda_cls=1.0, lambda_rec=10.0, log_interval=50, img_log_interval=200):
    adv_crit = nn.BCEWithLogitsLoss()
    cls_crit = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

    num_classes = _get_num_classes_from_loader(loader)
    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter()  # writes to runs/<timestamp>

    global_step = 0
    for epoch in range(epochs):
        G.train(); D.train()
        running_D = 0.0
        running_G = 0.0

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{epochs}", ncols=120)
        for batch_idx, batch in pbar:
            imgs, src_labels, _ = batch
            imgs = imgs.to(device)
            src_labels = src_labels.to(device).long()
            B = imgs.size(0)
            target_labels = torch.randint(0, num_classes, (B,), device=device, dtype=torch.long)

            # ----- Discriminator -----
            D.zero_grad()
            rf_real, cls_real = D(imgs)
            real_label_map = torch.ones_like(rf_real) * 0.9
            loss_D_real = adv_crit(rf_real, real_label_map)
            loss_cls_real = cls_crit(cls_real, src_labels)
            acc_real = (cls_real.argmax(dim=1) == src_labels).float().mean().item()
            with torch.no_grad():
                fake = G(imgs, target_labels)
            rf_fake, _ = D(fake)
            fake_label_map = torch.zeros_like(rf_fake)
            loss_D_fake = adv_crit(rf_fake, fake_label_map)
            loss_D = 0.5 * (loss_D_real + loss_D_fake) + lambda_cls * loss_cls_real
            loss_D.backward()
            optD.step()
            writer.add_scalar('Acc/D_cls_real', acc_real, global_step)
            # ----- Generator -----
            G.zero_grad()
            fake = G(imgs, target_labels)
            rf_fake, cls_fake = D(fake)
            loss_G_adv = adv_crit(rf_fake, real_label_map)
            loss_G_cls = cls_crit(cls_fake, target_labels)
            rec = G(fake, src_labels)
            loss_rec = l1(rec, imgs)
            loss_G = loss_G_adv + lambda_cls * loss_G_cls + lambda_rec * loss_rec
            loss_G.backward()
            optG.step()

            running_D += loss_D.item()
            running_G += loss_G.item()

            # TensorBoard scalars
            writer.add_scalar('Loss/D', loss_D.item(), global_step)
            writer.add_scalar('Loss/G', loss_G.item(), global_step)
            writer.add_scalar('Loss/G_adv', loss_G_adv.item(), global_step)
            writer.add_scalar('Loss/G_cls', loss_G_cls.item(), global_step)
            writer.add_scalar('Loss/G_rec', loss_rec.item(), global_step)

            # log images occasionally (first 8 images of batch)
            if global_step % img_log_interval == 0:
                with torch.no_grad():
                    # denormalize for visual clarity if using Normalize(mean=0.5,std=0.5)
                    def denorm(x): return x * 0.5 + 0.5
                    imgs_grid = vutils.make_grid(denorm(imgs[:8].cpu()), nrow=4, normalize=False)
                    fake_grid = vutils.make_grid(denorm(fake[:8].cpu()), nrow=4, normalize=False)
                    rec_grid = vutils.make_grid(denorm(rec[:8].cpu()), nrow=4, normalize=False)
                    writer.add_image('train/input', imgs_grid, global_step)
                    writer.add_image('train/fake', fake_grid, global_step)
                    writer.add_image('train/recon', rec_grid, global_step)

            # update progress bar text every log_interval
            if batch_idx % log_interval == 0:
                avg_D = running_D / (batch_idx + 1)
                avg_G = running_G / (batch_idx + 1)
                pbar.set_postfix({'D_loss': f"{avg_D:.4f}", 'G_loss': f"{avg_G:.4f}", 'step': global_step})

            global_step += 1

        # end epoch
        avg_D = running_D / len(loader)
        avg_G = running_G / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | D_loss={avg_D:.4f} | G_loss={avg_G:.4f}")

        save_path = os.path.join("checkpoints", f"generator_epoch_{epoch+1}.pth")
        torch.save(G.state_dict(), save_path)
        writer.add_scalar('Epoch/D_avg', avg_D, epoch+1)
        writer.add_scalar('Epoch/G_avg', avg_G, epoch+1)
        print(f"✓ Saved generator to: {save_path}")

    writer.close()


# ======================================
# VISUALIZATION
# ======================================
def visualize_results(generator, dataset, num_samples=5):
    generator.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    num_classes = dataset.num_classes
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    # helper to map index -> name
    def label_name(idx):
        if hasattr(dataset, 'emotion_names') and dataset.emotion_names is not None:
            try:
                return dataset.emotion_names[int(idx)]
            except Exception:
                return str(idx)
        return str(idx)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, label_int, _ = dataset[idx]
            inp = img.unsqueeze(0).to(device)
            # choose a target emotion different from source if possible
            target = (int(label_int) + 1) % num_classes
            out = generator(inp, torch.tensor([target], device=device))
            out = out.cpu().squeeze(0)

            inp_img = img.permute(1,2,0).numpy()
            out_img = out.permute(1,2,0).numpy()

            inp_img = np.clip(inp_img, 0, 1)
            out_img = np.clip(out_img, 0, 1)

            axes[i,0].imshow(inp_img.squeeze() if inp_img.shape[2]==1 else inp_img)
            axes[i,0].set_title(f"Input: {label_name(label_int)}")
            axes[i,0].axis("off")

            axes[i,1].imshow(out_img.squeeze() if out_img.shape[2]==1 else out_img)
            axes[i,1].set_title(f"Edited -> {label_name(target)}")
            axes[i,1].axis("off")

            axes[i,2].axis("off")

    plt.tight_layout()
    plt.savefig("gan_editing_results.png", dpi=150)
    plt.show()


 
# ======================================
# MAIN SCRIPT
# ======================================
def main():
    DATA_DIR = "test_deepFace/imgs_db/train"
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 2e-4
    TRAIN_SPLIT = 0.8

    full_dataset = FERDataset(
        root_dir=DATA_DIR,
        from_fer_csv=False,
        image_size=48,
        as_rgb=True
    )

    train_loader, val_loader = make_dataloaders(full_dataset, batch_size=BATCH_SIZE, train_split=TRAIN_SPLIT, num_workers=4)


    in_channels = 3 if full_dataset.as_rgb else 1
    generator = UNetCond(in_channels=in_channels, out_channels=in_channels, num_classes=full_dataset.num_classes).to(device)
    discriminator = Discriminator(in_channels=in_channels, num_classes=full_dataset.num_classes).to(device)


    print(f"Generator params: {sum(p.numel() for p in generator.parameters())}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters())}")

    print("\n=== TRAINING GAN ===")
    train_emotion_gan(generator, discriminator, train_loader, epochs=EPOCHS, lr=LR, lambda_cls=5.0, lambda_rec=5.0)

    print("\n=== VISUALIZATION ===")
    # load last checkpoint
    last_ckpt = os.path.join("checkpoints", f"generator_epoch_{EPOCHS}.pth")
    if os.path.exists(last_ckpt):
        generator.load_state_dict(torch.load(last_ckpt, map_location=device))
    visualize_results(generator, full_dataset, num_samples=3)


if __name__ == "__main__":
    main()
