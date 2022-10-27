## Audio Encoders - PyTorch

A collection of audio autoencoders, in PyTorch.

## Install

```bash
pip install audio-encoders-pytorch
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/audio-encoders-pytorch?style=flat&colorA=black&colorB=black)](https://pypi.org/project/audio-encoders-pytorch/)


## Usage

### AutoEncoder1d
```py
from audio_encoders_pytorch import AutoEncoder1d

autoencoder = AutoEncoder1d(
    in_channels=2,              # Number of input channels
    channels=32,                # Number of base channels
    multipliers=[1, 1, 2, 2],   # Channel multiplier between layers (i.e. channels * multiplier[i] -> channels * multiplier[i+1])
    factors=[4, 4, 4],          # Downsampling/upsampling factor per layer
    num_blocks=[2, 2, 2]        # Number of resnet blocks per layer
)

x = torch.randn(1, 2, 2**18)    # [1, 2, 262144]
x_recon = autoencoder(x)        # [1, 2, 262144]
```

### Discriminator1d
```py
from audio_encoders_pytorch import Discriminator1d

discriminator = Discriminator1d(
    in_channels=2,                  # Number of input channels
    channels=32,                    # Number of base channels
    multipliers=[1, 1, 2, 2],       # Channel multiplier between layers (i.e. channels * multiplier[i] -> channels * multiplier[i+1])
    factors=[8, 8, 8],              # Downsampling factor per layer
    num_blocks=[2, 2, 2],           # Number of resnet blocks per layer
    use_loss=[True, True, True]     # Whether to use this layer as GAN loss
)

wave_true = torch.randn(1, 2, 2**18)
wave_fake = torch.randn(1, 2, 2**18)

loss_generator, loss_discriminator = discriminator(wave_true, wave_fake)
# tensor(0.613949, grad_fn=<MeanBackward0>) tensor(0.097330, grad_fn=<MeanBackward0>)
```



## Citations

A-Pipeline inspired by
```bibtex
@misc{2208.08706,
Author = {Marco Pasini and Jan Schlüter},
Title = {Musika! Fast Infinite Waveform Music Generation},
Year = {2022},
Eprint = {arXiv:2208.08706},
}
```
