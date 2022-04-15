# minDiffusion

<!-- #region -->
<p align="center">
<img  src="contents/_ddpm_sample_19.png">
</p>

Goal of this educational repository is to provide a self-contained, minimalistic implementation of diffusion models using Pytorch.

Many implementations of diffusion models can be a bit overwhelming. Here, `superminddpm` : under 200 lines of code, fully self contained implementation of DDPM with Pytorch is a good starting point for anyone who wants to get started with Denoising Diffusion Models, without having to spend time on the details.

Simply:

```
$ python superminddpm.py
```

Above script is self-contained. (Of course, you need to have pytorch and torchvision installed. Latest version should suffice. We do not use any cutting edge features.)

If you want to use the bit more refactored code, that runs CIFAR10 dataset:

```
$ python train_cifar10.py
```

Currently has:

- [x] Tiny implementation of DDPM
- [x] MNIST, CIFAR dataset.
- [x] Simple unet structure.

TODOS

- [ ] DDIM
- [ ] Classifier Guidance
- [ ] Multimodality

# Updates!

- Using more parameter yields better result for MNIST.
- More comments in superminddpm.py
