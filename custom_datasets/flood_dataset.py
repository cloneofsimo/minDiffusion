import os
from torch.utils.data import Dataset
from torchvision.io import read_image


class FloodDataset(Dataset):
    def __init__(self, path, transform=None):
        super(FloodDataset, self).__init__()
        assert os.path.exists(path), f"{path} doesn't exist"

        self.files = []
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                self.files.append(os.path.abspath(os.path.join(dirpath, f)))

        assert self.files, f"{path} folder is empty"
        self.transform = transform

    def __getitem__(self, index):
        # select random time series
        x = read_image(self.files[index])
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.files)
