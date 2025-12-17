import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """CNN-based encoder that maps 128x128 image to latent space"""
    
    def __init__(self, latent_dim=16):
        super().__init__()
        
        # Input: 1 x 128 x 128
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # 32 x 64 x 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 64 x 32 x 32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 128 x 16 x 16
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 256 x 8 x 8
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # 512 x 4 x 4
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Flatten: 512 * 4 * 4 = 8192
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    """CNN-based decoder that maps latent vector to 128x128 image"""
    
    def __init__(self, latent_dim=16):
        super().__init__()
        
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # Input: 512 x 4 x 4
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 256 x 8 x 8
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 128 x 16 x 16
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 64 x 32 x 32
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 32 x 64 x 64
        self.deconv5 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # 1 x 128 x 128
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, 4, 4)
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = torch.sigmoid(self.deconv5(x))  # Output in [0, 1]
        
        return x


class VAE(nn.Module):
    """Variational Autoencoder for Noordzij Cube letter reconstruction"""
    
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def encode(self, x):
        """Encode input to latent space (returns mean)"""
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z):
        """Decode latent vector to image"""
        return self.decoder(z)
    
    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        """
        VAE loss = Reconstruction loss + KL divergence
        
        Args:
            recon_x: reconstructed image
            x: original image
            mu: mean of latent distribution
            logvar: log variance of latent distribution
            beta: weight for KL divergence (beta-VAE)
        """
        # Reconstruction loss (binary cross entropy)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
