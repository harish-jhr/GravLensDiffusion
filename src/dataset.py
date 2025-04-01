import torch
from torch.utils.data import Dataset
import numpy as np

class LensingDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        
        self.data = np.load(npz_file)['Samples']  # Load images
        self.transform = transform

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        img = self.data[index]  # Shape: (1, 150, 150)
        img = torch.tensor(img, dtype=torch.float32)  # Convert to tensor

        # Normalize to [-1, 1] range (important for DDPM)
        img = (img - 0.5) * 2

        # Apply transformations (if any)
        if self.transform:
            img = self.transform(img)

        return img  


