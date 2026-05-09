import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class IntrAClassificationDataset(Dataset):
    def __init__(self, csv_file, data_dir, augment=False):
        self.metadata = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.augment = augment

    def __len__(self):
        return len(self.metadata)

    def _augment(self, points):
        xyz, normals = points[:, :3], points[:, 3:]
        xyz += np.random.uniform(-0.2, 0.2, size=(1, 3))
        xyz *= np.random.uniform(0.67, 1.5)
        return np.concatenate([xyz, normals], axis=1)

    def __getitem__(self, idx):
        fname = self.metadata.iloc[idx, 0]
        label = self.metadata.iloc[idx, 1]

        points = np.load(os.path.join(self.data_dir, fname))  # (N, 6)
        if self.augment:
            points = self._augment(points)

        X = torch.from_numpy(points).float()
        Y_target = torch.tensor(label, dtype=torch.long)

        return X, Y_target