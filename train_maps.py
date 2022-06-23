"""
Extremely Minimalistic Implementation of DDPM

https://arxiv.org/abs/2006.11239

Everything is self contained. (Except for pytorch and torchvision... of course)

run it with `python superminddpm.py`
"""
import time
import os
import logging
import sys
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
from optparse import OptionParser
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid

logging.basicConfig(stream=sys.stdout,
                    format='[%(levelname)s] - [%(asctime)s] - [%(filename)s:%(lineno)d] - %(message)s',
                    level=logging.INFO)


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
        self.files = []
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                self.files.append(os.path.abspath(os.path.join(dirpath, f)))
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


def get_rgb(x):
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


def plot_imgs():

    tf = transforms.Compose([
        RandomCrop((64, 64)),
        transforms.Normalize([560, 676, 546],
                             [558, 415, 284])
    ])

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/PASTIS")
    dataset = PastisDataset(data_path, transform=tf)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    for batch_idx, batch in enumerate(dataloader):
        for img_idx in range(batch.shape[0]):
            img = get_rgb(batch[img_idx])
            plt.imshow(img)
            plt.show()


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 7, padding=3),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)


class DummyEpsModel(nn.Module):
    """
    This should be unet-like, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """

    def __init__(self, n_channel: int) -> None:
        super(DummyEpsModel, self).__init__()
        self.conv = nn.Sequential(  # with batchnorm
            blk(n_channel, 64),
            blk(64, 128),
            blk(128, 256),
            blk(256, 512),
            blk(512, 256),
            blk(256, 128),
            blk(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1),
        )

    def forward(self, x, t) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        return self.conv(x)


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(x_i, i / self.n_T)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i


def train_maps(n_epoch: int = 100, device="cuda:0", data_loaders=os.cpu_count()//2, train_ckpt=1,
                load_weights=False) -> None:

    model = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)

    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/ddpm_mnist.pth")
    if load_weights and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        logging.info("Loaded saved weights")
    model.to(device)

    tf = transforms.Compose([
        RandomCrop((64, 64)),
        transforms.Normalize([560, 676, 546],
                             [558, 415, 284])
    ])

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/PASTIS")
    dataset = PastisDataset(data_path, transform=tf)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=data_loaders)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)

    # plot_imgs()
    try:
        for i in range(n_epoch):
            model.train()

            pbar = tqdm(dataloader)
            loss_ema = None
            for x, _ in pbar:
                optim.zero_grad()
                x = x.to(device)
                loss = model(x)
                loss.backward()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
                pbar.set_description(f"loss: {loss_ema:.4f}")
                optim.step()

            if (i + 1) % train_ckpt == 0:
                start_time = time.time()
                torch.save(model.state_dict(), weights_path)
                logging.info(f"Saving weights took {time.time()-start_time}; it_num: {i+1} out of {n_epoch}")

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt, please wait for saving weights ...")
        torch.save(model.state_dict(), weights_path)


def eval_maps(device="cuda:0"):
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/ddpm_mnist.pth")
    if not os.path.exists(weights_path):
        raise ValueError(f"path {weights_path} doesn't exist")

    model = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    logging.info("Loaded model")

    model.eval()
    with torch.no_grad():
        xh = model.sample(16, (1, 28, 28), device)
        grid = make_grid(xh, nrow=4)
        save_image(grid, "./contents/ddpm_sample.png")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-m", "--mode", dest="mode", help="mode = [train/eval]", default='train')
    parser.add_option("--gpu_num", dest="gpu_num", help="gpu_num, -1 means cpu usage", type="int", default=-1)
    parser.add_option("--n_epochs", dest="n_epochs", help="number of epochs", type="int", default=100)
    parser.add_option("--data_loaders", dest="data_loaders",
                      help="number of cpus for data loading. Advised using less than cpu count",
                      type="int", default=os.cpu_count()//2)
    parser.add_option("--train_ckpt", dest="train_ckpt", help="saves model every train_ckpt iterations",
                      type="int", default=1)
    parser.add_option("--load_weights", dest="load_weights", action="store_true",
                      help="load weights and continue training", default=False)

    (options, args) = parser.parse_args()

    device = "cpu"
    if options.gpu_num != -1:
        assert torch.cuda.is_available(), "cuda isn't available, check cuda installation"
        gpu_count = torch.cuda.device_count()
        assert options.gpu_num < gpu_count, f"gpu {options.gpu_num} doesn't exist; total gpus: {gpu_count}"
        device = f"cuda:{str(options.gpu_num)}"

    if options.mode == 'train':
        logging.info("Training starts")
        train_maps(options.n_epochs, device, options.data_loaders, options.train_ckpt, options.load_weights)
    elif options.mode == 'eval':
        logging.info("Eval starts")
        eval_maps(device)
    else:
        raise ValueError(f"incorrect mode {options.mode}, must be in [train, eval]")
