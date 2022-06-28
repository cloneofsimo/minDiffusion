import os
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from custom_datasets import MapsDataset


def plot_imgs(dataset):
    """
    Util for plotting imgs
    """
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    for batch_idx, batch in enumerate(dataloader):
        grid = make_grid(batch, nrow=4, normalize=False)
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()


def compute_stats(dataset):
    """
    Utility for computation of means and stds for Normalize
    """

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


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/maps_train/images")
    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Normalize((0, 0, 0),
                             (255, 255, 255)),
    ])
    dataset = MapsDataset(image_path=data_path,
                          transform=tf)
    plot_imgs(dataset)
