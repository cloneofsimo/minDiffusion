import os
from torch.utils.data import Dataset
from torchvision.io import read_image


class MapsDataset(Dataset):
    def __init__(self, image_path, transform=None):
        super(MapsDataset, self).__init__()
        assert os.path.exists(image_path), f"{image_path} doesn't exist"

        self.files = []
        for dirpath, _, filenames in os.walk(image_path):
            for f in filenames:
                self.files.append(os.path.abspath(os.path.join(dirpath, f)))

        assert self.files, f"{image_path} folder is empty"
        self.transform = transform

    def __getitem__(self, index):
        x = read_image(self.files[index]).float()
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.files)
