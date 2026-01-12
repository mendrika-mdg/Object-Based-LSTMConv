import os
import sys
import glob
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import random


IDX_MONTH_SIN = 0
IDX_MONTH_COS = 1
IDX_TOD_SIN   = 2
IDX_TOD_COS   = 3
IDX_LAT       = 4
IDX_LON       = 5
IDX_LAT_MIN   = 6
IDX_LAT_MAX   = 7
IDX_LON_MIN   = 8
IDX_LON_MAX   = 9
IDX_TIR       = 10
IDX_SIZE      = 11
IDX_MASK      = 12


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
    ref = torch.mean(p ** 2) + torch.mean(t ** 2)
    return (1 - mse / (ref + 1e-8)).clamp(0.0, 1.0)


class SpatiallyEnhancedLoss(nn.Module):
    def __init__(self, window_size=15, pos_weight=25.0, alpha=0.3):
        super().__init__()
        self.pool = nn.AvgPool2d(window_size, 1, window_size // 2)
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = self.bce(logits, targets)
        fss = F.mse_loss(self.pool(probs), self.pool(targets))
        return self.alpha * bce + (1 - self.alpha) * fss


class ShardDataset(IterableDataset):
    def __init__(self, shard_dir, split_by_rank=True, split_by_worker=True):
        super().__init__()
        self.shard_dir = shard_dir
        self.split_by_rank = split_by_rank
        self.split_by_worker = split_by_worker

        self.files = sorted(
            glob.glob(os.path.join(shard_dir, "**", "*.pt"), recursive=True)
        )
        if not self.files:
            raise RuntimeError(f"No shard files found under {shard_dir}")

    def _rankinfo(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def __iter__(self):
        files = list(self.files)

        rank, world = self._rankinfo()
        worker = torch.utils.data.get_worker_info()

        if self.split_by_rank and world > 1:
            files = files[rank::world]

        if worker and self.split_by_worker:
            files = files[worker.id::worker.num_workers]

        while True:
            random.shuffle(files)
            for f in files:
                try:
                    d = torch.load(f, map_location="cpu")
                except Exception:
                    print(f"Skipping corrupted shard: {f}")
                    continue

                X, Y = d["X"], d["Y"]
                for i in range(X.shape[0]):
                    yield X[i].float(), Y[i].unsqueeze(0).float()


class BilinearSplatRasteriser(nn.Module):
    def __init__(self, latent_hw=64, lat_range=(-35.0, 24.0), lon_range=(-18.0, 51.0), eps=1e-6):
        super().__init__()
        self.latent_hw = int(latent_hw)
        self.lat_min, self.lat_max = float(lat_range[0]), float(lat_range[1])
        self.lon_min, self.lon_max = float(lon_range[0]), float(lon_range[1])
        self.eps = float(eps)

    def forward(self, emb, lat, lon, valid):
        b, n, d = emb.shape
        h = self.latent_hw
        w = self.latent_hw

        lat = lat.clamp(self.lat_min, self.lat_max)
        lon = lon.clamp(self.lon_min, self.lon_max)

        x = (lon - self.lon_min) / (self.lon_max - self.lon_min + self.eps) * (w - 1)
        y = (self.lat_max - lat) / (self.lat_max - self.lat_min + self.eps) * (h - 1)

        x0 = torch.floor(x).long().clamp(0, w - 1)
        y0 = torch.floor(y).long().clamp(0, h - 1)
        x1 = (x0 + 1).clamp(0, w - 1)
        y1 = (y0 + 1).clamp(0, h - 1)

        wx = (x - x0.float()).clamp(0.0, 1.0)
        wy = (y - y0.float()).clamp(0.0, 1.0)

        w00 = (1 - wx) * (1 - wy)
        w10 = wx * (1 - wy)
        w01 = (1 - wx) * wy
        w11 = wx * wy

        valid = valid.float()
        w00 = w00 * valid
        w10 = w10 * valid
        w01 = w01 * valid
        w11 = w11 * valid

        out = emb.new_zeros((b, d, h * w))
        norm = emb.new_zeros((b, 1, h * w))

        def scatter(weight, xi, yi):
            idx = (yi * w + xi).view(b, n)
            val = (emb * weight.unsqueeze(-1)).permute(0, 2, 1).contiguous()
            out.scatter_add_(2, idx.unsqueeze(1).expand(-1, d, -1), val)
            norm.scatter_add_(2, idx.unsqueeze(1), weight.unsqueeze(1))

        scatter(w00, x0, y0)
        scatter(w10, x1, y0)
        scatter(w01, x0, y1)
        scatter(w11, x1, y1)

        out = out / (norm + self.eps)
        out = out.view(b, d, h, w)
        return out


class SimpleDecoder(nn.Module):
    def __init__(self, embed_dim, out_hw=(512, 512), dropout_p=0.2):
        super().__init__()
        self.out_hw = out_hw
        ch = [embed_dim, 512, 256, 128, 64, 32]
        layers = []
        for c1, c2 in zip(ch[:-1], ch[1:]):
            layers += [
                nn.ConvTranspose2d(c1, c2, 4, 2, 1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_p),
            ]
        self.up = nn.Sequential(*layers)
        self.final = nn.Conv2d(ch[-1], 1, 1)

    def forward(self, x):
        x = self.up(x)
        if x.shape[-2:] != self.out_hw:
            x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        return self.final(x)


class Core2MapModel(pl.LightningModule):
    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        lr=1e-4,
        dropout_p=0.2,
        pos_weight=25.0,
        alpha=0.3,
        latent_hw=64,
        lat_range=(-35.0, 24.0),
        lon_range=(-18.0, 51.0),
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_proj = nn.Sequential(
            nn.Linear(13, embed_dim),
            nn.Dropout(dropout_p),
        )

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout_p,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)

        self.raster = BilinearSplatRasteriser(
            latent_hw=latent_hw,
            lat_range=lat_range,
            lon_range=lon_range,
        )

        self.decoder = SimpleDecoder(
            embed_dim,
            out_hw=(2015, 2187),
            dropout_p=dropout_p,
        )

        self.criterion = SpatiallyEnhancedLoss(
            window_size=15,
            pos_weight=pos_weight,
            alpha=alpha,
        )

        self.val_auc = BinaryAUROC()

    def forward(self, x):
        b, t, n, f = x.shape

        x0 = x[:, -1, :, :]

        valid = x0[:, :, IDX_MASK] > 0

        tokens = self.in_proj(x0)
        tokens = self.transformer(tokens, src_key_padding_mask=~valid)

        lat = x0[:, :, IDX_LAT]
        lon = x0[:, :, IDX_LON]

        z = self.raster(tokens, lat, lon, valid)
        return self.decoder(z)

    def training_step(self, batch, _):
        x, y = (t.to(self.device) for t in batch)
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        x, y = (t.to(self.device) for t in batch)
        preds = torch.sigmoid(self(x))

        preds_m = F.interpolate(
            preds,
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        )

        y_m = F.adaptive_max_pool2d(
            y,
            output_size=(512, 512),
        )

        for w in [9, 15, 25]:
            self.log(
                f"val_fss_{w}",
                compute_fss(preds_m, y_m, w),
                on_epoch=True,
                prog_bar=(w == 15),
                sync_dist=True,
            )

        self.val_auc.update(
            preds_m.flatten(),
            y_m.flatten().int(),
        )

    def on_validation_epoch_end(self):
        self.log("val_auc", self.val_auc.compute(), prog_bar=True, sync_dist=True)
        self.val_auc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


def main():
    torch.set_float32_matmul_precision("high")

    lead = int(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4

    pl.seed_everything(seed, workers=True)

    base = f"/gws/ssde/j25b/swift/mendrika/pancast/shards/t{lead:03d}min"
    train_dir = f"{base}/train"
    val_dir = f"{base}/val"

    ckpt = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/pancast_64_rasterised/checkpoints/t{lead:03d}min/seed{seed}"
    os.makedirs(ckpt, exist_ok=True)

    train_dl = DataLoader(
        ShardDataset(train_dir, True, True),
        batch_size=4,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    val_dl = DataLoader(
        ShardDataset(val_dir, split_by_rank=True, split_by_worker=True),
        batch_size=1,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    model = Core2MapModel(
        lr=lr,
        latent_hw=64,
        lat_range=(-35.0, 24.0),
        lon_range=(-18.0, 51.0),
    )

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        precision="16-mixed",
        logger=WandbLogger(
            project="dev-pancast-64-raster",
            name=f"t{lead}_seed{seed}",
        ),
        log_every_n_steps=5,
        limit_val_batches=300,
        limit_train_batches=3000,
        callbacks=[
            ModelCheckpoint(
                dirpath=ckpt,
                filename="best-pancast",
                monitor="val_fss_15",
                mode="max",
                save_top_k=1,
            ),
            EarlyStopping(
                monitor="val_fss_15",
                mode="max",
                patience=5,
                min_delta=0.001,
            ),
        ],
    )

    trainer.fit(model, train_dl, val_dl)
    print(f"Training complete for seed {seed}")


if __name__ == "__main__":
    main()
