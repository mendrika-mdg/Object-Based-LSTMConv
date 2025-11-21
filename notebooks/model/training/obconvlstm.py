import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import warnings
warnings.filterwarnings("ignore", message="It is recommended to use `self.log`")


# --------------------------------------------------------
# Utility
# --------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_fss(preds, targets, window=9):
    pool = nn.AvgPool2d(window, 1, window // 2)
    p = pool(preds)
    t = pool(targets)
    mse = torch.mean((p - t) ** 2)
    ref = torch.mean(p**2) + torch.mean(t**2)
    return (1 - mse / (ref + 1e-8)).clamp(0.0, 1.0)


# --------------------------------------------------------
# BCE + Multi-Scale FSS Hybrid
# --------------------------------------------------------
class MultiScaleHybridLoss(nn.Module):
    def __init__(self, windows=(5, 9, 17), weights=(0.25, 0.5, 0.25),
                 pos_weight=25.0, alpha=0.7):
        super().__init__()
        self.pools = nn.ModuleList([nn.AvgPool2d(k, 1, k // 2) for k in windows])
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        terms = []
        for w, pool in zip(self.weights, self.pools):
            p = pool(probs)
            t = pool(targets)
            mse = F.mse_loss(p, t)
            ref = (p.pow(2).mean() + t.pow(2).mean()).clamp_min(1e-8)
            terms.append(w * (mse / ref))
        fss_loss = torch.stack(terms).sum()
        bce_loss = self.bce(logits, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * fss_loss


# --------------------------------------------------------
# Infinite Iterable Dataset (DDP-safe)
# --------------------------------------------------------
class ShardDataset(IterableDataset):
    def __init__(self, shard_dir, split_by_rank=True, split_by_worker=True):
        super().__init__()
        self.shard_dir = shard_dir
        self.split_by_rank = split_by_rank
        self.split_by_worker = split_by_worker
        self.files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".pt"))
        if not self.files:
            raise RuntimeError("No shards found")

    def _rankinfo(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def __iter__(self):

        rank, world = self._rankinfo()
        worker = torch.utils.data.get_worker_info()

        files = list(self.files)

        if self.split_by_rank and world > 1:
            n = len(files)
            size = max(1, n // world)
            files = files[rank * size:(rank + 1) * size]

        if worker and self.split_by_worker:
            files = files[worker.id::worker.num_workers]

        while True:
            random.shuffle(files)
            for f in files:
                d = torch.load(os.path.join(self.shard_dir, f), map_location="cpu")
                X, Y = d["X"], d["Y"]
                for i in range(X.shape[0]):
                    yield X[i].float(), Y[i].float()

class SmoothDecoder(nn.Module):
    def __init__(self, in_channels=64, out_hw=(512, 512), dropout=0.15):
        super().__init__()
        self.out_hw = out_hw
        layers = []
        ch = [64, 64, 32, 16, 8]
        c_in = in_channels

        for c in ch:
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(c_in, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
            ]
            c_in = c

        self.net = nn.Sequential(*layers)
        self.final = nn.Conv2d(c_in, 1, 1)

    def forward(self, x):
        x = self.net(x)
        return self.final(
            F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        )


# --------------------------------------------------------
# LSTM Core2Map Model
# --------------------------------------------------------
class OB2MapModel(pl.LightningModule):
    def __init__(self, lstm_hidden=256, lstm_layers=1, dropout=0.15,
                 pos_weight=25.0, alpha=0.7, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            650, lstm_hidden, lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.map_proj = nn.Linear(lstm_hidden, 64 * 16 * 16)
        self.decoder = SmoothDecoder(64, (512, 512))

        self.criterion = MultiScaleHybridLoss(pos_weight=pos_weight, alpha=alpha)
        self.val_auc = BinaryAUROC()

    def forward(self, x):
        b = x.size(0)
        x = x.view(b, 5, -1)
        _, (h, _) = self.lstm(x)
        z = self.map_proj(h[-1]).view(b, 64, 16, 16)
        return self.decoder(z)

    def training_step(self, batch, _):
        x, y = (t.to(self.device) for t in batch)
        y = y.unsqueeze(1)
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, _):
        x, y = (t.to(self.device) for t in batch)
        y = y.unsqueeze(1)

        preds = torch.sigmoid(self(x))

        # ---- Log multi-scale FSS ----
        for w in [3, 5, 9]:
            fss_val = compute_fss(preds, y, window=w)
            self.log(
                f"val_fss_{w}",
                fss_val,
                on_epoch=True,
                prog_bar=(w == 9),
                sync_dist=True
            )

        # ---- Update AUC metric ----
        self.val_auc.update(preds.flatten(), y.flatten().int())

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        self.log("val_auc", auc, prog_bar=True, sync_dist=True)
        self.val_auc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    torch.set_float32_matmul_precision("high")

    lead = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    set_seed(seed)
    pl.seed_everything(seed, workers=True)

    base = f"/work/scratch-nopw2/mendrika/OB/preprocessed/t{lead}"
    train_dir = f"{base}/train_t{lead}"
    val_dir = f"{base}/val_t{lead}"

    ckpt = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/OB/checkpoints/lstm_core2map_hybrid/corrected/t{lead}/seed{seed}"
    os.makedirs(ckpt, exist_ok=True)

    train_dl = DataLoader(
        ShardDataset(train_dir, True, True),
        batch_size=32,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )

    # IMPORTANT: batch_size=1 avoids DDP deadlocks
    val_dl = DataLoader(
        ShardDataset(val_dir, split_by_rank=False, split_by_worker=True),
        batch_size=1,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )

    model = OB2MapModel()

    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        precision="bf16-mixed",
        logger=WandbLogger(project="western_sahel_scale_corrected",
                           name=f"t{lead}_seed{seed}"),
        log_every_n_steps=5,
        limit_val_batches=300,
        limit_train_batches=3000,
        callbacks=[
            ModelCheckpoint(
                dirpath=ckpt,
                filename="best-core2map",
                monitor="val_auc",       
                mode="max",
                save_top_k=1
            ),
            EarlyStopping(
                monitor="val_auc",      
                mode="max",
                patience=5,
                min_delta=0.001
            )
        ]
    )

    trainer.fit(model, train_dl, val_dl)
    print(f"Training complete for seed {seed}")


if __name__ == "__main__":
    main()
