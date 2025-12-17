#!/usr/bin/env python3
"""
Train VAE model on Noordzij Cube dataset
"""
import argparse
import torch
from pathlib import Path

from model import VAE
from model.dataset import get_dataloaders
from train import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train VAE on Noordzij Cube dataset')
    parser.add_argument('--train-dir', type=str, default='dataset/noordzij_train',
                        help='Path to training data directory')
    parser.add_argument('--test-dir', type=str, default='dataset/noordzij_test',
                        help='Path to test data directory')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for checkpoints and visualizations')
    parser.add_argument('--latent-dim', type=int, default=16,
                        help='Dimensionality of latent space')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta parameter for VAE loss (KL weight)')
    parser.add_argument('--monitor-interval', type=int, default=5,
                        help='Interval for monitoring (in epochs)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/mps/cpu). Auto-detect if not specified')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print("="*80)
    print("VAE Training on Noordzij Cube Dataset")
    print("="*80)
    print(f"Device: {device}")
    print(f"Training data: {args.train_dir}")
    print(f"Test data: {args.test_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Beta (KL weight): {args.beta}")
    print("="*80 + "\n")
    
    # Load data
    print("Loading datasets...")
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}\n")
    
    # Create model
    print("Creating VAE model...")
    model = VAE(latent_dim=args.latent_dim)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        device=device,
        output_dir=args.output_dir,
        lr=args.lr,
        beta=args.beta,
        monitor_interval=args.monitor_interval
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(n_epochs=args.epochs)
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    print(f"Best test loss: {trainer.best_test_loss:.4f}")
    print(f"Outputs saved to: {args.output_dir}")
    print(f"Best model: {Path(args.output_dir) / 'checkpoints' / 'best_model.pth'}")
    print("="*80)


if __name__ == '__main__':
    main()
