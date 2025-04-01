from torch.utils.data import Dataset, DataLoader
import torch

class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list  # A list of images or tensors

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return torch.tensor(self.data_list[idx], dtype=torch.float32)  # Convert to tensor

