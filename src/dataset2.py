import torch
from torch.utils.data import Dataset
import torch.functional as F
import numpy as np

class LensingDataset2(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.images = []
        #self.labels = []

        for  imgs in data.items():
            self.images.extend(imgs)  # Store all images
        
        self.images = np.array(self.images)  # Convert to NumPy array
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = self.images[index]  # Get image
        img_tensor = torch.tensor(img, dtype=torch.float32)  # Convert to tensor
        # Resize to (1, 64, 64) using bilinear interpolation
        img_tensor = F.interpolate(self.images, size=(1,64, 64), mode='bilinear', align_corners=False)
        img_tensor = (2 * img_tensor) - 1  
        return img_tensor



