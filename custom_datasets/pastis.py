from matplotlib import pyplot as plt
import numpy as np
import os

from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/PASTIS")
CROP_SHAPE = (64, 64)


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:,
                      top: top + new_h,
                      left: left + new_w]

        return image


class PastisDataset(Dataset):
    def __init__(self, path, transform=None):
        super(PastisDataset, self).__init__()
        assert os.path.exists(path), f"{path} doesn't exist"

        self.files = []
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                self.files.append(os.path.abspath(os.path.join(dirpath, f)))

        assert self.files, f"{path} folder is empty"
        self.transform = transform
        self.eps = 1e-6

    def __getitem__(self, index):
        x = np.load(self.files[index]).astype(np.float32)
        # select random time series
        random_ts = np.random.randint(0, x.shape[0])
        x = torch.from_numpy(x[random_ts, [2, 1, 0]])
        x_min = x.view(x.size(0), -1).min(dim=-1).values[:, None, None]
        x_max = x.view(x.size(0), -1).max(dim=-1).values[:, None, None]
        x = torch.floor((x - x_min) * (1 / (x_max - x_min + self.eps) * 255))
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.files)


def plot_imgs_pastis():
    """
    Util for plotting imgs from PASTIS
    """
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/PASTIS")
    dataset = PastisDataset(data_path)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    for batch_idx, batch in enumerate(dataloader):
        grid = make_grid(batch, nrow=4, normalize=True)
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()


def compute_stats_pastis():
    """
    Utility for computation of means and stds for Normalize
    """
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/PASTIS")
    dataset = PastisDataset(data_path)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    mean = torch.zeros(3)
    for batch_idx, batch in enumerate(dataloader):
        mean += batch.mean(axis=[2, 3]).sum(0)
    mean = mean / len(dataset)

    var = torch.zeros(3)
    for batch_idx, batch in enumerate(dataloader):
        var += (torch.sub(batch, mean[:, None, None]) ** 2).mean(axis=[2, 3]).sum(0)
    std = torch.sqrt(var / (len(dataset)))

    return mean.tolist(), std.tolist()

def compute_max():
    """
    Utility for computation of means and stds for Normalize
    """
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/PASTIS")
    dataset = PastisDataset(data_path)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    max_val = 0.0
    for batch_idx, batch in enumerate(dataloader):
        max_val = max(max_val, torch.max(batch))
    return max_val

if __name__ == "__main__":
    # print(compute_stats_pastis())
    plot_imgs_pastis()