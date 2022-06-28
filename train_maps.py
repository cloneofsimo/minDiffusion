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
from tqdm import tqdm
from optparse import OptionParser
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from custom_datasets import MapsDataset
from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/maps_train/images")
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/ddpm_maps.pth")
RESIZE_SHAPE = (64, 64)


logging.basicConfig(stream=sys.stdout,
                    format='[%(levelname)s] - [%(asctime)s] - [%(filename)s:%(lineno)d] - %(message)s',
                    level=logging.INFO)


def train_maps(n_epoch: int = 100, device="cuda:0", data_loaders=os.cpu_count()//2, train_ckpt=1,
                load_weights=False) -> None:

    model = DDPM(eps_model=NaiveUnet(3, 3, n_feat=256), betas=(1e-4, 0.02), n_T=1000)

    if load_weights and os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH))
        logging.info("Loaded saved weights")
    model.to(device)

    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Normalize((0, 0, 0),
                             (255, 255, 255)),
    ])

    dataset = MapsDataset(
        image_path=DATA_PATH,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=data_loaders)
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)

    try:
        for i in range(n_epoch):
            model.train()

            pbar = tqdm(dataloader)
            loss_ema = None
            for x in pbar:
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
                torch.save(model.state_dict(), WEIGHTS_PATH)
                save_time = time.time() - start_time

                start_time = time.time()
                eval_maps(device, model, str(i+1))
                eval_time = time.time() - start_time
                logging.info(f"Saving weights took {save_time:.3f}s; Eval took {eval_time:.3f}s \
                it_num: {i+1} out of {n_epoch}")

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt, please wait for saving weights ...")
        torch.save(model.state_dict(), WEIGHTS_PATH)


def eval_maps(device="cuda:0", model=None, it_num='last'):
    if model is None:
        assert os.path.exists(WEIGHTS_PATH), f"path {WEIGHTS_PATH} doesn't exist"
        model = DDPM(eps_model=NaiveUnet(3, 3, n_feat=256), betas=(1e-4, 0.02), n_T=1000)
        model.load_state_dict(torch.load(WEIGHTS_PATH))
        model.to(device)
        logging.info("Loaded model")

    model.eval()
    with torch.no_grad():
        xh = model.sample(16, (3, 64, 64), device)
        grid = make_grid(xh, nrow=4, normalize=True, scale_each=True)
        save_image(grid, f"./contents/ddpm_sample_{it_num}.png")


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

    if not os.path.exists(os.path.dirname(WEIGHTS_PATH)):
        os.mkdir(os.path.dirname(WEIGHTS_PATH))

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
        raise Exception(f"incorrect mode {options.mode}, must be in [train, eval]")
