from matplotlib import pyplot as plt
import numpy as np
import os

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/PASTIS")
# means and stds of 3 channels after compute_stats() function
MEANS = [1378.9217751242898, 1312.8329745205965, 1132.4872602982955]
STDS = [733.1572029807351, 587.0386005748402, 550.9132863825017]
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

    def __getitem__(self, index):
        x = np.load(self.files[index]).astype(np.float32)
        # select random time series
        random_ts = np.random.randint(0, x.shape[0])
        x = torch.from_numpy(x[random_ts, [2, 1, 0]])
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.files)


def get_rgb_pastis(x):
    """Utility function to get a displayable rgb image
    from a Sentinel-2 time series.
    """
    im = x.cpu().numpy()
    mx = im.max(axis=(1, 2))
    mi = im.min(axis=(1, 2))
    im = (im - mi[:, None, None])/(mx - mi)[:, None, None]
    im = im.swapaxes(0, 2).swapaxes(0, 1)
    im = np.clip(im, a_max=1, a_min=0)
    return im


def plot_imgs_pastis():
    """
    Util for plotting imgs from PASTIS
    """

    tf = transforms.Compose([
        RandomCrop(CROP_SHAPE),
        transforms.Normalize(MEANS, STDS)
    ])

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/PASTIS")
    dataset = PastisDataset(data_path, transform=tf)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    for batch_idx, batch in enumerate(dataloader):
        for img_idx in range(batch.shape[0]):
            img = get_rgb_pastis(batch[img_idx])
            plt.imshow(img)
            plt.show()


def compute_stats():
    """
    Utility for computation of means and stds for Normalize
    """
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/PASTIS")
    dataset = PastisDataset(data_path)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    means_stacked, stds_stacked = np.empty((0, 3)), np.empty((0, 3))
    for batch_idx, batch in enumerate(dataloader):
        means = batch.mean(axis=[2, 3]).numpy()
        stds = batch.std(axis=[2, 3]).numpy()
        means_stacked = np.vstack((means_stacked, means))
        stds_stacked = np.vstack((stds_stacked, stds))

    return means_stacked.mean(axis=0).tolist(), stds_stacked.mean(axis=0).tolist()


if __name__ == "__main__":
    print(compute_stats())
