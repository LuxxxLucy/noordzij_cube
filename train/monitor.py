import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch.nn.functional as F


class Monitor:
    """Monitoring and visualization for VAE training"""
    
    def __init__(self, output_dir, device):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Create subdirectories
        (self.output_dir / "reconstructions").mkdir(exist_ok=True)
        (self.output_dir / "interpolations").mkdir(exist_ok=True)
        (self.output_dir / "losses").mkdir(exist_ok=True)
        
        # Track losses
        self.train_losses = []
        self.test_losses = []
        self.train_recon_losses = []
        self.train_kl_losses = []
    
    def visualize_reconstruction(self, model, dataset, n_samples=5, epoch=0, split='train'):
        """
        Visualize reconstruction of random samples
        
        Args:
            model: VAE model
            dataset: Dataset to sample from
            n_samples: Number of samples to visualize
            epoch: Current epoch number
            split: 'train' or 'test'
        """
        model.eval()
        
        # Sample random indices
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        
        # Get images
        images = torch.stack([dataset[i]['image'] for i in indices]).to(self.device)
        
        with torch.no_grad():
            recon, _, _ = model(images)
        
        # Create figure
        fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
        
        for i in range(n_samples):
            # Original
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)
            
            # Reconstruction
            axes[1, i].imshow(recon[i].cpu().squeeze(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)
        
        plt.suptitle(f'Epoch {epoch} - {split.capitalize()} Reconstruction', fontsize=12)
        plt.tight_layout()
        
        save_path = self.output_dir / "reconstructions" / f"recon_{split}_epoch_{epoch:04d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved reconstruction: {save_path}")
    
    def find_distant_points(self, model, dataset, n_candidates=100):
        """
        Find two points that are far apart in latent space
        
        Args:
            model: VAE model
            dataset: Dataset to sample from
            n_candidates: Number of candidates to consider
        
        Returns:
            idx1, idx2: Indices of two distant points
        """
        model.eval()
        
        # Sample random candidates
        indices = np.random.choice(len(dataset), min(n_candidates, len(dataset)), replace=False)
        images = torch.stack([dataset[i]['image'] for i in indices]).to(self.device)
        
        with torch.no_grad():
            latents = model.encode(images)
        
        # Find two points with maximum distance
        latents_np = latents.cpu().numpy()
        max_dist = 0
        best_i, best_j = 0, 1
        
        for i in range(len(latents_np)):
            for j in range(i + 1, len(latents_np)):
                dist = np.linalg.norm(latents_np[i] - latents_np[j])
                if dist > max_dist:
                    max_dist = dist
                    best_i, best_j = i, j
        
        return indices[best_i], indices[best_j]
    
    def visualize_interpolation(self, model, dataset, n_steps=7, epoch=0, split='train'):
        """
        Visualize interpolation between two distant points
        
        Args:
            model: VAE model
            dataset: Dataset to sample from
            n_steps: Number of interpolation steps
            epoch: Current epoch number
            split: 'train' or 'test'
        """
        model.eval()
        
        # Find two distant points
        idx1, idx2 = self.find_distant_points(model, dataset)
        
        # Get the two images
        img1 = dataset[idx1]['image'].unsqueeze(0).to(self.device)
        img2 = dataset[idx2]['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Encode to latent space
            z1 = model.encode(img1)
            z2 = model.encode(img2)
            
            # Interpolate in latent space
            alphas = torch.linspace(0, 1, n_steps).to(self.device)
            interpolated_images = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                img_interp = model.decode(z_interp)
                interpolated_images.append(img_interp)
            
            interpolated_images = torch.cat(interpolated_images, dim=0)
        
        # Create figure
        fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 2, 2))
        
        for i in range(n_steps):
            axes[i].imshow(interpolated_images[i].cpu().squeeze(), cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'{i/(n_steps-1):.2f}', fontsize=8)
        
        plt.suptitle(f'Epoch {epoch} - {split.capitalize()} Interpolation', fontsize=12)
        plt.tight_layout()
        
        save_path = self.output_dir / "interpolations" / f"interp_{split}_epoch_{epoch:04d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved interpolation: {save_path}")
    
    def plot_losses(self):
        """Plot training losses"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Total loss
        if self.train_losses:
            axes[0].plot(self.train_losses, label='Train')
        if self.test_losses:
            axes[0].plot(self.test_losses, label='Test')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        if self.train_recon_losses:
            axes[1].plot(self.train_recon_losses, label='Recon Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Reconstruction Loss')
        axes[1].set_title('Reconstruction Loss (Train)')
        axes[1].grid(True, alpha=0.3)
        
        # KL loss
        if self.train_kl_losses:
            axes[2].plot(self.train_kl_losses, label='KL Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('KL Divergence')
        axes[2].set_title('KL Divergence (Train)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "losses" / "loss_curves.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved loss curves: {save_path}")
    
    def update_losses(self, train_loss, test_loss, train_recon_loss, train_kl_loss):
        """Update loss tracking"""
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_recon_losses.append(train_recon_loss)
        self.train_kl_losses.append(train_kl_loss)
