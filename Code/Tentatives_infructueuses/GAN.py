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
import torch.nn.functional as F

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
# DATASET (inchangé)
# ======================================
class FERDataset(Dataset):
    def __init__(
        self,
        csv_file: Optional[str] = None,
        root_dir: Optional[str] = None,
        from_fer_csv: bool = False,
        image_size: int = 128,  # augmenté à 128
        as_rgb: bool = False,
        transforms_: Optional[transforms.Compose] = None,
        emotion_names: Optional[List[str]] = None
    ):
        assert from_fer_csv ^ (root_dir is not None), "Donner soit csv_file (from_fer_csv=True) soit root_dir."
        self.image_size = image_size
        self.as_rgb = as_rgb

        if transforms_ is None:
            if as_rgb:
                norm = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                self.transforms = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),  # augmentation
                    transforms.ToTensor(),
                    norm
                ])
            else:
                norm = transforms.Normalize(mean=[0.5], std=[0.5])
                self.transforms = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
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
            if self.emotion_names is None:
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
                    if f.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                        continue
                    try:
                        if f.stat().st_size == 0:
                            continue
                        with Image.open(f) as im:
                            pil = im.convert('RGB') if as_rgb else im.convert('L')
                            self.samples.append((pil.copy(), label))
                    except (UnidentifiedImageError, OSError):
                        continue

        if len(self.samples) == 0:
            raise RuntimeError("Aucun échantillon trouvé dans FERDataset.")
        self.num_classes = len(self.emotion_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pil, label = self.samples[idx]
        img = self.transforms(pil)
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
# ATTENTION MODULE (self-attention pour features)
# ======================================
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(B, C, H * W)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

# ======================================
# FILM LAYER (modulation émotion)
# ======================================
class FiLM(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.gamma_fc = nn.Linear(num_classes, num_features)
        self.beta_fc = nn.Linear(num_classes, num_features)

    def forward(self, x, cond):
        # x: [B, C, H, W], cond: [B, num_classes]
        gamma = self.gamma_fc(cond).unsqueeze(2).unsqueeze(3)
        beta = self.beta_fc(cond).unsqueeze(2).unsqueeze(3)
        return x * (1 + gamma) + beta

# ======================================
# U-NET GENERATOR (amélioré avec FiLM + attention)
# ======================================
class ResBlock(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.film = FiLM(channels, num_classes)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x, cond):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.film(x, cond)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_classes=7, features=[64,128,256,512]):
        super().__init__()
        self.num_classes = num_classes
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 7, padding=3),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(features[0], features[1], 4, stride=2, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 4, stride=2, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 4, stride=2, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck avec ResBlocks conditionnés
        self.bottleneck = nn.Sequential(
            ResBlock(features[3], num_classes),
            ResBlock(features[3], num_classes),
            SelfAttention(features[3])
        )
        
        # Decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(features[3], features[2], 4, stride=2, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(features[2]*2, features[1], 4, stride=2, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(features[1]*2, features[0], 4, stride=2, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(features[0]*2, out_channels, 7, padding=3),
            nn.Tanh()
        )
        
        # FiLM layers pour decoder
        self.film_dec4 = FiLM(features[2], num_classes)
        self.film_dec3 = FiLM(features[1], num_classes)
        self.film_dec2 = FiLM(features[0], num_classes)

    def forward(self, x, labels):
        # labels: [B] int tensor -> one-hot
        B = x.size(0)
        cond = F.one_hot(labels, self.num_classes).float()
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck avec conditioning
        b = e4
        for layer in self.bottleneck:
            if isinstance(layer, ResBlock):
                b = layer(b, cond)
            else:
                b = layer(b)
        
        # Decoder avec skip + FiLM
        d4 = self.dec4(b)
        d4 = self.film_dec4(d4, cond)
        d4 = torch.cat([d4, e3], dim=1)
        
        d3 = self.dec3(d4)
        d3 = self.film_dec3(d3, cond)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.dec2(d3)
        d2 = self.film_dec2(d2, cond)
        d2 = torch.cat([d2, e1], dim=1)
        
        return self.final(d2)

# ======================================
# DISCRIMINATOR (multi-scale + classification)
# ======================================
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super().__init__()
        
        def discriminator_block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        # Main discriminator
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )
        
        # Real/fake head
        self.adv_layer = nn.Sequential(
            nn.Conv2d(512, 1, 4, padding=1)
        )
        
        # Classification head
        self.cls_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.model(x)
        validity = self.adv_layer(features)
        cls_logits = self.cls_layer(features)
        return validity, cls_logits

# ======================================
# PERCEPTUAL LOSS (VGG features)
# ======================================
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return F.l1_loss(x_vgg, y_vgg)

# ======================================
# TRAINING LOOP (amélioré)
# ======================================
def train_emotion_gan(G, D, loader, epochs=50, lr=1e-4, lambda_cls=10.0, lambda_rec=5.0, lambda_perc=2.0, log_interval=50, img_log_interval=200):
    adv_crit = nn.MSELoss()  # LSGAN (plus stable que BCE)
    cls_crit = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()
    perc_loss = PerceptualLoss().to(device)
    
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))
    
    num_classes = _get_num_classes_from_loader(loader)
    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter()

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
            validity_real, cls_real = D(imgs)
            loss_D_real = adv_crit(validity_real, torch.ones_like(validity_real))
            loss_cls_real = cls_crit(cls_real, src_labels)
            
            with torch.no_grad():
                fake = G(imgs, target_labels)
            validity_fake, _ = D(fake.detach())
            loss_D_fake = adv_crit(validity_fake, torch.zeros_like(validity_fake))
            
            loss_D = 0.5 * (loss_D_real + loss_D_fake) + lambda_cls * loss_cls_real
            loss_D.backward()
            optD.step()

            # ----- Generator (every 2 D steps) -----
            if batch_idx % 1 == 0:
                G.zero_grad()
                fake = G(imgs, target_labels)
                validity_fake, cls_fake = D(fake)
                
                loss_G_adv = adv_crit(validity_fake, torch.ones_like(validity_fake))
                loss_G_cls = cls_crit(cls_fake, target_labels)
                
                # Cycle consistency
                rec = G(fake, src_labels)
                loss_rec = l1(rec, imgs)
                
                # Perceptual loss
                if imgs.size(1) == 3:
                    loss_perc = perc_loss(fake, imgs)
                else:
                    loss_perc = torch.tensor(0.0, device=device)
                
                # Identity loss (si target = source, output doit = input)
                same_emo_fake = G(imgs, src_labels)
                loss_identity = l1(same_emo_fake, imgs)
                
                loss_G = loss_G_adv + lambda_cls * loss_G_cls + lambda_rec * loss_rec + lambda_perc * loss_perc + 5.0 * loss_identity
                loss_G.backward()
                optG.step()

                running_G += loss_G.item()

            running_D += loss_D.item()

            # Logging
            writer.add_scalar('Loss/D', loss_D.item(), global_step)
            writer.add_scalar('Loss/G', loss_G.item(), global_step)
            writer.add_scalar('Loss/G_cls', loss_G_cls.item(), global_step)
            writer.add_scalar('Loss/G_rec', loss_rec.item(), global_step)

            if global_step % img_log_interval == 0:
                with torch.no_grad():
                    def denorm(x): return x * 0.5 + 0.5
                    imgs_grid = vutils.make_grid(denorm(imgs[:8].cpu()), nrow=4, normalize=False)
                    fake_grid = vutils.make_grid(denorm(fake[:8].cpu()), nrow=4, normalize=False)
                    rec_grid = vutils.make_grid(denorm(rec[:8].cpu()), nrow=4, normalize=False)
                    writer.add_image('train/input', imgs_grid, global_step)
                    writer.add_image('train/fake', fake_grid, global_step)
                    writer.add_image('train/recon', rec_grid, global_step)

            if batch_idx % log_interval == 0:
                avg_D = running_D / (batch_idx + 1)
                avg_G = running_G / max(1, (batch_idx + 1) // 1)
                pbar.set_postfix({'D_loss': f"{avg_D:.4f}", 'G_loss': f"{avg_G:.4f}"})

            global_step += 1

        # Save checkpoint
        save_path = os.path.join("checkpoints", f"generator_epoch_{epoch+1}.pth")
        torch.save(G.state_dict(), save_path)
        print(f"✓ Saved: {save_path}")

    writer.close()

def _get_num_classes_from_loader(loader):
    ds = loader.dataset
    if isinstance(ds, torch.utils.data.Subset):
        return ds.dataset.num_classes
    return ds.num_classes

# ======================================
# VISUALIZATION
# ======================================
def visualize_results(generator, dataset, num_samples=5):
    generator.eval()
    fig, axes = plt.subplots(num_samples, dataset.num_classes+1, figsize=(3*(dataset.num_classes+1), 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, label_int, _ = dataset[idx]
            inp = img.unsqueeze(0).to(device)
            
            # Original
            inp_img = img.permute(1,2,0).cpu().numpy()
            inp_img = (inp_img * 0.5 + 0.5).clip(0,1)
            axes[i,0].imshow(inp_img.squeeze() if inp_img.shape[2]==1 else inp_img)
            axes[i,0].set_title(f"Input: {dataset.emotion_names[label_int]}")
            axes[i,0].axis("off")
            
            # Generate all emotions
            for target in range(dataset.num_classes):
                out = generator(inp, torch.tensor([target], device=device))
                out_img = out.cpu().squeeze(0).permute(1,2,0).numpy()
                out_img = (out_img * 0.5 + 0.5).clip(0,1)
                axes[i,target+1].imshow(out_img.squeeze() if out_img.shape[2]==1 else out_img)
                axes[i,target+1].set_title(dataset.emotion_names[target])
                axes[i,target+1].axis("off")

    plt.tight_layout()
    plt.savefig("gan_all_emotions.png", dpi=150)
    plt.show()

# ======================================
# MAIN
# ======================================
def main():
    DATA_DIR = "test_deepFace/imgs_db/train"
    BATCH_SIZE = 16  # réduit pour 128x128
    EPOCHS = 50
    LR = 2e-4
    TRAIN_SPLIT = 0.8

    full_dataset = FERDataset(
        root_dir=DATA_DIR,
        from_fer_csv=False,
        image_size=96,  # augmenté
        as_rgb=True
    )

    train_loader, val_loader = make_dataloaders(full_dataset, batch_size=BATCH_SIZE, train_split=TRAIN_SPLIT, num_workers=4)

    in_channels = 3 if full_dataset.as_rgb else 1
    generator = UNetGenerator(in_channels=in_channels, out_channels=in_channels, num_classes=full_dataset.num_classes, features=[32,64,128,256]).to(device)
    discriminator = MultiScaleDiscriminator(in_channels=in_channels, num_classes=full_dataset.num_classes).to(device)

    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    print("\n=== TRAINING GAN ===")
    train_emotion_gan(generator, discriminator, train_loader, epochs=EPOCHS, lr=LR, lambda_cls=15.0, lambda_rec=2.0, lambda_perc=3.0)

    print("\n=== VISUALIZATION ===")
    last_ckpt = os.path.join("checkpoints", f"generator_epoch_{EPOCHS}.pth")
    if os.path.exists(last_ckpt):
        generator.load_state_dict(torch.load(last_ckpt, map_location=device))
    visualize_results(generator, full_dataset, num_samples=5)


if __name__ == "__main__":
    main()