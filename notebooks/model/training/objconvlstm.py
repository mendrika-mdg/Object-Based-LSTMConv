import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import random

# set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# compute fss
def compute_fss(preds, targets, window=9):
    pool = nn.AvgPool2d(window, 1, window // 2)
    p = pool(preds)
    t = pool(targets)
    mse = torch.mean((p - t) ** 2)
    ref = torch.mean(p ** 2) + torch.mean(t ** 2)
    return (1 - mse / (ref + 1e-8)).clamp(0.0, 1.0)

# hybrid loss
class MultiScaleHybridLoss(nn.Module):
    def __init__(self, windows=(5, 9, 17), weights=(0.25, 0.5, 0.25),
                 pos_weight=25.0, alpha=0.7):
        super().__init__()
        self.pools = nn.ModuleList([nn.AvgPool2d(k, 1, k // 2) for k in windows])
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.alpha = alpha
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        fss_terms = []
        for w, pool in zip(self.weights, self.pools):
            p = pool(probs)
            t = pool(targets)
            mse = F.mse_loss(p, t)
            ref = (p.pow(2).mean() + t.pow(2).mean()).clamp_min(1e-8)
            fss_terms.append(w * (mse / ref))
        fss_loss = torch.stack(fss_terms).sum()
        bce_loss = self.bce(logits, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * fss_loss

# dataset
class ShardDataset(Dataset):
    def __init__(self, shard_dir):
        files = sorted([f for f in os.listdir(shard_dir) if f.endswith(".pt")])
        if not files:
            raise RuntimeError(f"No shards found in {shard_dir}")
        X, G, Y = [], [], []
        for f in files:
            d = torch.load(os.path.join(shard_dir, f), map_location="cpu")
            X.append(d["X"])
            G.append(d["G"])
            Y.append(d["Y"])
        self.X = torch.cat(X)
        self.G = torch.cat(G)
        self.Y = torch.cat(Y)
        print(f"Loaded {len(self.X)} samples from {len(files)} shards")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i].float(), self.G[i].float(), self.Y[i].float()

# decoder
class SmoothDecoder(nn.Module):
    def __init__(self, in_channels=64, out_hw=(512, 512), dropout=0.15):
        super().__init__()
        self.out_hw = out_hw
        layers = []
        ch = [64, 64, 32, 16, 8]
        c_in = 64
        for c in ch:
            layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            layers.append(nn.Conv2d(c_in, c, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(dropout))
            c_in = c
        self.net = nn.Sequential(*layers)
        self.final = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        x = self.net(x)
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        return self.final(x)

# core2map model
class Core2MapModel(pl.LightningModule):
    def __init__(self, lstm_hidden=512, lstm_layers=1, dropout=0.15,
                 pos_weight=25.0, alpha=0.7, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # lstm encoder
        self.lstm = nn.LSTM(
            input_size=650,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # linear projection to 32x32 latent
        self.map_proj = nn.Linear(lstm_hidden + 4, 64 * 32 * 32)

        # decoder
        self.decoder = SmoothDecoder(in_channels=64, out_hw=(512, 512))

        # loss
        self.criterion = MultiScaleHybridLoss(
            windows=(5, 9, 17),
            weights=(0.25, 0.5, 0.25),
            pos_weight=pos_weight,
            alpha=alpha
        )

        # metrics
        self.val_auc = BinaryAUROC()

        # learning rate
        self.lr = lr

    def forward(self, x, g):
        b = x.size(0)
        x = x.view(b, 5, -1)
        out, (h, _) = self.lstm(x)
        h_last = h[-1]
        hg = torch.cat([h_last, g], dim=1)
        latent = self.map_proj(hg).view(b, 64, 32, 32)
        logits = self.decoder(latent)
        return logits

    def training_step(self, batch, _):
        x, g, y = (t.to(self.device) for t in batch)
        logits = self(x, g)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        x, g, y = (t.to(self.device) for t in batch)
        preds = torch.sigmoid(self(x, g))
        for w in [5, 9, 17]:
            self.log(f"val_fss_{w}", compute_fss(preds, y, w),
                     on_epoch=True, prog_bar=(w == 9), sync_dist=True)
        self.val_auc.update(preds.flatten(), y.flatten().int())

    def on_validation_epoch_end(self):
        self.log("val_auc", self.val_auc.compute(), prog_bar=True, sync_dist=True)
        self.val_auc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# main
def main():
    torch.set_float32_matmul_precision("high")

    lead_time = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    set_seed(seed)
    pl.seed_everything(seed, workers=True)

    base_dir = f"/work/scratch-nopw2/mendrika/OB/preprocessed/t{lead_time}"
    train_dir = f"{base_dir}/train_t{lead_time}"
    val_dir   = f"{base_dir}/val_t{lead_time}"

    ckpt_dir  = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/OB/checkpoints/lstm_core2map/t{lead_time}/seed{seed}"
    os.makedirs(ckpt_dir, exist_ok=True)

    train_ds = ShardDataset(train_dir)
    val_ds   = ShardDataset(val_dir)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl   = DataLoader(val_ds, batch_size=32, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

    model = Core2MapModel()

    logger = WandbLogger(project="western_sahel_lstm_core2map",
                         name=f"t{lead_time}_seed{seed}")

    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="gpu", devices=4, strategy="ddp",
        precision="bf16-mixed",
        logger=logger,
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        callbacks=[
            ModelCheckpoint(
                dirpath=ckpt_dir, filename="best",
                monitor="val_fss_9", mode="max", save_top_k=1
            ),
            EarlyStopping(
                monitor="val_fss_9", mode="max",
                patience=5, min_delta=0.001
            )
        ]
    )

    trainer.fit(model, train_dl, val_dl)
    print(f"Training complete for seed {seed}")

if __name__ == "__main__":
    main()
