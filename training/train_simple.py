"""
Simple training script for LILITH - single-station prediction.

This is a simplified training loop that trains the model on individual station
sequences without the graph structure. Good for initial model development.
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.processing.ghcn_processor import GHCNProcessor


class WeatherDataset(Dataset):
    """Simple weather dataset for single-station training."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, meta: np.ndarray):
        """
        Args:
            X: Input sequences [N, input_days, features]
            Y: Target sequences [N, target_days, features]
            meta: Station metadata [N, 4] (lat, lon, elev, day_of_year)
        """
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.meta = torch.from_numpy(meta)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.meta[idx]


class SimpleLILITH(nn.Module):
    """
    Simplified LILITH for single-station sequence prediction.
    Uses a straightforward encoder-decoder Transformer architecture.
    """

    def __init__(
        self,
        input_features: int = 3,
        output_features: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
        max_forecast: int = 90
    ):
        super().__init__()

        self.d_model = d_model
        self.max_forecast = max_forecast
        self.output_features = output_features

        # Input projection
        self.input_proj = nn.Linear(input_features, d_model)

        # Station metadata embedding
        self.meta_embed = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )

        # Positional encoding
        self.register_buffer('pos_encoding', self._create_pe(500, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, output_features)
        )

        # Initialize weights
        self._init_weights()

    def _create_pe(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        meta: torch.Tensor,
        target_len: int
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence [batch, seq_len, features]
            meta: Station metadata [batch, 4]
            target_len: Number of days to forecast
        Returns:
            Forecast [batch, target_len, output_features]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :].to(device)

        # Add station embedding
        station_emb = self.meta_embed(meta)
        x = x + station_emb.unsqueeze(1)

        # Encode
        memory = self.encoder(x)

        # Create decoder queries
        tgt = torch.zeros(batch_size, target_len, self.d_model, device=device)
        tgt = tgt + self.pos_encoding[:, :target_len, :].to(device)
        tgt = tgt + station_emb.unsqueeze(1)

        # Create causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_len, device=device)

        # Decode
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        # Project to output
        return self.output_proj(output)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int = 0,
    total_epochs: int = 1
) -> float:
    """Train for one epoch with progress bar."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", unit="batch",
                dynamic_ncols=True, leave=True)

    for X, Y, meta in pbar:
        X = X.to(device)
        Y = Y.to(device)
        meta = meta.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with torch.amp.autocast('cuda'):
            pred = model(X, meta, Y.size(1))
            loss = criterion(pred, Y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})

    return total_loss / num_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, Y, meta in dataloader:
            X = X.to(device)
            Y = Y.to(device)
            meta = meta.to(device)

            with torch.amp.autocast('cuda'):
                pred = model(X, meta, Y.size(1))
                loss = criterion(pred, Y)

            total_loss += loss.item()
            num_batches += 1
            all_preds.append(pred.cpu())
            all_targets.append(Y.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Calculate metrics
    # Temperature RMSE (first two features: TMAX, TMIN)
    temp_rmse = torch.sqrt(((preds[:, :, :2] - targets[:, :, :2]) ** 2).mean()).item()

    # MAE
    temp_mae = (preds[:, :, :2] - targets[:, :, :2]).abs().mean().item()

    return total_loss / num_batches, temp_rmse, temp_mae


def main():
    """Main training loop."""
    import argparse

    parser = argparse.ArgumentParser(description="Train LILITH model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--process-data", action="store_true", help="Process raw GHCN data first")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples for faster training (None = all)")
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw" / "ghcn_daily"
    processed_dir = base_dir / "data" / "processed"
    training_dir = processed_dir / "training"
    checkpoints_dir = base_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Process data if needed
    if args.process_data or not (training_dir / "X.npy").exists():
        logger.info("Processing GHCN data...")
        processor = GHCNProcessor(
            raw_dir,
            processed_dir,
            raw_dir / "ghcnd-stations.txt"
        )
        df = processor.process_all_stations(min_years=10)

        if df.empty:
            logger.error("No data to process!")
            return

        df.to_parquet(processed_dir / "ghcn_combined.parquet")

        X, Y, meta = processor.create_training_sequences(
            df,
            input_days=30,
            target_days=14,
            stride=7
        )

        if len(X) == 0:
            logger.error("No training sequences created!")
            return

        processor.save_training_data(X, Y, meta)

    # Load training data
    logger.info("Loading training data...")
    X = np.load(training_dir / "X.npy")
    Y = np.load(training_dir / "Y.npy")
    meta = np.load(training_dir / "meta.npy")

    logger.info(f"Loaded {len(X)} training samples")
    logger.info(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # Normalize features
    stats = np.load(training_dir / "stats.npz")
    X_mean, X_std = stats['X_mean'], stats['X_std']
    Y_mean, Y_std = stats['Y_mean'], stats['Y_std']

    X = (X - X_mean) / (X_std + 1e-6)
    Y = (Y - Y_mean) / (Y_std + 1e-6)

    # Normalize meta (lat, lon, elev, day_of_year)
    meta[:, 0] = meta[:, 0] / 90.0  # lat to [-1, 1]
    meta[:, 1] = meta[:, 1] / 180.0  # lon to [-1, 1]
    meta[:, 2] = meta[:, 2] / 5000.0  # elevation to ~[0, 1]
    # day_of_year already normalized to [0, 1]

    # Optionally subsample for faster iteration
    max_samples = args.max_samples
    if max_samples and len(X) > max_samples:
        logger.info(f"Subsampling from {len(X)} to {max_samples} samples for faster training")
        subsample_idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[subsample_idx]
        Y = Y[subsample_idx]
        meta = meta[subsample_idx]

    # Train/val split
    n_train = int(len(X) * 0.9)
    indices = np.random.permutation(len(X))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = WeatherDataset(X[train_idx], Y[train_idx], meta[train_idx])
    val_dataset = WeatherDataset(X[val_idx], Y[val_idx], meta[val_idx])

    # Use multiple workers on non-Windows or if explicitly enabled
    num_workers = 4 if os.name != 'nt' else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create model
    model = SimpleLILITH(
        input_features=X.shape[-1],
        output_features=Y.shape[-1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
        dropout=args.dropout
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    # Training loop
    best_val_loss = float('inf')
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"Training samples: {len(train_dataset):,} | Validation samples: {len(val_dataset):,}")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch, args.epochs)
        val_loss, val_rmse, val_mae = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Denormalize metrics for interpretable values
        temp_rmse_denorm = val_rmse * Y_std[:2].mean()
        temp_mae_denorm = val_mae * Y_std[:2].mean()

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Temp RMSE: {temp_rmse_denorm:.2f}°C | "
            f"Temp MAE: {temp_mae_denorm:.2f}°C"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_rmse': temp_rmse_denorm,
                'config': {
                    'input_features': X.shape[-1],
                    'output_features': Y.shape[-1],
                    'd_model': args.d_model,
                    'nhead': args.nhead,
                    'num_encoder_layers': args.layers,
                    'num_decoder_layers': args.layers,
                    'dropout': args.dropout
                },
                'normalization': {
                    'X_mean': X_mean.tolist(),
                    'X_std': X_std.tolist(),
                    'Y_mean': Y_mean.tolist(),
                    'Y_std': Y_std.tolist()
                }
            }
            torch.save(checkpoint, checkpoints_dir / "lilith_best.pt")
            logger.success(f"Saved best model with RMSE: {temp_rmse_denorm:.2f}°C")

    # Save final model
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': checkpoint['config'],
        'normalization': checkpoint['normalization'],
        'val_rmse': temp_rmse_denorm
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(final_checkpoint, checkpoints_dir / f"lilith_{timestamp}.pt")
    logger.success(f"Training complete! Best RMSE: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
