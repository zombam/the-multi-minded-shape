import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SDFDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.class_to_idx = {}
        class_folders = sorted(os.listdir(root_dir))
        for idx, class_name in enumerate(class_folders):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue  # Skip files like .DS_Store
            self.class_to_idx[class_name] = idx
            for file in os.listdir(class_path):
                if file.endswith('.npz'):
                    self.samples.append(os.path.join(class_path, file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sdf_file = self.samples[idx]
        label = self.labels[idx]
        sdf = np.load(sdf_file)['sdf']
        sdf = torch.tensor(sdf, dtype=torch.float32).unsqueeze(0)  # shape (1, D, H, W)
        return sdf, label