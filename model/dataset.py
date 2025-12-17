import os
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class NoordZijDataset(Dataset):
    """Dataset for Noordzij Cube letter images"""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory containing PNG images and metadata.json
            transform: Optional transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Get all PNG files
        self.image_files = sorted(list(self.data_dir.glob("*.png")))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No PNG images found in {data_dir}")
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        # Get filename for reference
        filename = img_path.stem
        
        return {
            'image': image,
            'filename': filename,
            'idx': idx
        }
    
    def get_image_by_idx(self, idx):
        """Get a single image by index"""
        return self[idx]['image']
    
    def get_batch(self, indices):
        """Get a batch of images by indices"""
        return torch.stack([self[i]['image'] for i in indices])


def get_dataloaders(train_dir, test_dir, batch_size=32, num_workers=4):
    """
    Create train and test dataloaders
    
    Args:
        train_dir: Directory containing training images
        test_dir: Directory containing test images
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, test_loader, train_dataset, test_dataset
    """
    train_dataset = NoordZijDataset(train_dir)
    test_dataset = NoordZijDataset(test_dir)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset, test_dataset
