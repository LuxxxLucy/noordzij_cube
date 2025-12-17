import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json
from .monitor import Monitor


class Trainer:
    """Trainer for VAE model with comprehensive monitoring"""
    
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        train_dataset,
        test_dataset,
        device,
        output_dir='outputs',
        lr=1e-3,
        beta=1.0,
        monitor_interval=5
    ):
        """
        Args:
            model: VAE model
            train_loader: Training data loader
            test_loader: Test data loader
            train_dataset: Training dataset (for monitoring)
            test_dataset: Test dataset (for monitoring)
            device: Device to train on
            output_dir: Directory to save outputs
            lr: Learning rate
            beta: Beta parameter for VAE loss (KL weight)
            monitor_interval: Interval for monitoring (in epochs)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.beta = beta
        self.monitor_interval = monitor_interval
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Scheduler (optional)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Monitor
        self.monitor = Monitor(output_dir, device)
        
        # Checkpoint directory
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.best_test_loss = float('inf')
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            
            # Forward pass
            recon, mu, logvar = self.model(images)
            loss, recon_loss, kl_loss = self.model.loss_function(
                recon, images, mu, logvar, beta=self.beta
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track losses
            batch_size = images.size(0)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item() / batch_size:.4f}',
                'recon': f'{recon_loss.item() / batch_size:.4f}',
                'kl': f'{kl_loss.item() / batch_size:.4f}'
            })
        
        # Average losses
        n_samples = len(self.train_loader.dataset)
        avg_loss = total_loss / n_samples
        avg_recon_loss = total_recon_loss / n_samples
        avg_kl_loss = total_kl_loss / n_samples
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def evaluate(self):
        """Evaluate on test set"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                
                # Forward pass
                recon, mu, logvar = self.model(images)
                loss, recon_loss, kl_loss = self.model.loss_function(
                    recon, images, mu, logvar, beta=self.beta
                )
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        # Average losses
        n_samples = len(self.test_loader.dataset)
        avg_loss = total_loss / n_samples
        avg_recon_loss = total_recon_loss / n_samples
        avg_kl_loss = total_kl_loss / n_samples
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def save_checkpoint(self, filename='checkpoint.pth'):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_test_loss': self.best_test_loss,
            'train_losses': self.monitor.train_losses,
            'test_losses': self.monitor.test_losses,
        }
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"  Saved checkpoint: {save_path}")
    
    def load_checkpoint(self, filename='checkpoint.pth'):
        """Load model checkpoint"""
        load_path = self.checkpoint_dir / filename
        
        if not load_path.exists():
            print(f"  Checkpoint not found: {load_path}")
            return False
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_test_loss = checkpoint['best_test_loss']
        self.monitor.train_losses = checkpoint['train_losses']
        self.monitor.test_losses = checkpoint['test_losses']
        
        print(f"  Loaded checkpoint: {load_path}")
        return True
    
    def run_monitoring(self):
        """Run all monitoring visualizations"""
        print(f"\n  Running monitoring at epoch {self.epoch}...")
        
        # Reconstruction on train
        self.monitor.visualize_reconstruction(
            self.model, self.train_dataset, n_samples=5, epoch=self.epoch, split='train'
        )
        
        # Reconstruction on test
        self.monitor.visualize_reconstruction(
            self.model, self.test_dataset, n_samples=5, epoch=self.epoch, split='test'
        )
        
        # Interpolation on train
        self.monitor.visualize_interpolation(
            self.model, self.train_dataset, n_steps=7, epoch=self.epoch, split='train'
        )
        
        # Interpolation on test
        self.monitor.visualize_interpolation(
            self.model, self.test_dataset, n_steps=7, epoch=self.epoch, split='test'
        )
        
        # Plot losses
        self.monitor.plot_losses()
    
    def train(self, n_epochs):
        """
        Main training loop
        
        Args:
            n_epochs: Number of epochs to train
        """
        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Latent dim: {self.model.latent_dim}")
        print(f"Beta (KL weight): {self.beta}")
        print(f"Monitor interval: {self.monitor_interval} epochs\n")
        
        for epoch in range(n_epochs):
            self.epoch = epoch + 1
            
            # Train
            train_loss, train_recon_loss, train_kl_loss = self.train_epoch()
            
            # Evaluate
            test_loss, test_recon_loss, test_kl_loss = self.evaluate()
            
            # Update scheduler
            self.scheduler.step(test_loss)
            
            # Update monitor
            self.monitor.update_losses(train_loss, test_loss, train_recon_loss, train_kl_loss)
            
            # Print epoch summary
            print(f"\nEpoch {self.epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f})")
            print(f"  Test Loss:  {test_loss:.4f} (Recon: {test_recon_loss:.4f}, KL: {test_kl_loss:.4f})")
            
            # Save best model
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                self.save_checkpoint('best_model.pth')
                print(f"  New best model! Test loss: {test_loss:.4f}")
            
            # Periodic monitoring
            if self.epoch % self.monitor_interval == 0 or self.epoch == 1:
                self.run_monitoring()
            
            # Save regular checkpoint
            if self.epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.epoch}.pth')
        
        # Final monitoring
        print("\n" + "="*80)
        print("Training completed!")
        print("="*80)
        self.run_monitoring()
        self.save_checkpoint('final_model.pth')
        
        # Save training summary
        summary = {
            'n_epochs': n_epochs,
            'latent_dim': self.model.latent_dim,
            'beta': self.beta,
            'best_test_loss': self.best_test_loss,
            'final_train_loss': self.monitor.train_losses[-1],
            'final_test_loss': self.monitor.test_losses[-1],
        }
        
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining summary saved to: {self.output_dir / 'training_summary.json'}")
