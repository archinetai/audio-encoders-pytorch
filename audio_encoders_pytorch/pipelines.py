from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .modules import Discriminator1d, STFTAutoEncoder1d
from .utils import prefix_dict, to_list


def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


class APipeline(nn.Module):
    def __init__(
        self,
        autoencoder: STFTAutoEncoder1d,
        discriminator: Discriminator1d,
        loss_stft_weight: float = 1.0,
        num_stage_steps: Union[int, Sequence[int]] = 50000,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.discriminator = discriminator
        self.loss_stft_weight = loss_stft_weight

        self.num_stage_steps = to_list(num_stage_steps)
        assert len(self.num_stage_steps) == 1, "only 1 num_stage_steps required"
        self.register_buffer("step_id", torch.tensor(0))
        self.register_buffer("stage_id", torch.tensor(0))

        # Init multi resolution stft loss
        import auraloss

        scales = [2048, 1024, 512, 256, 128]
        hop_sizes, win_lengths, overlap = [], [], 0.75
        for scale in scales:
            hop_sizes += [int(scale * (1.0 - overlap))]
            win_lengths += [scale]
        self.stft_loss_fn = auraloss.freq.SumAndDifferenceSTFTLoss(
            fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths
        )

    def step(self):
        # Check if next pipeline stage has to be activated
        for i, step in enumerate(self.num_stage_steps):
            if self.step_id == step:
                self.stage_id += 1
                self.setup_stage(self.stage_id)
                print(f"Stage {self.stage_id-1} completed.")
        self.step_id += 1

    def setup_stage(self, stage_id: int) -> None:
        if stage_id == 1:
            # Freeze encoder
            freeze_model(self.autoencoder.encoder)

    def forward(
        self, wave: Tensor, with_info: bool = False
    ) -> Union[Tuple[Tensor, Optional[Tensor]], Tuple[Tensor, Optional[Tensor], Dict]]:
        wave_pred, info = self.autoencoder(wave, with_info=True)

        if self.stage_id == 0:
            # Train full autoencoder, only log-magnitude spectrogram l1 loss
            log_magnitude_pred = info["decoder_log_magnitude"]
            magnitude, _ = self.autoencoder.stft.encode(wave)
            log_magnitude = torch.log(magnitude)
            loss_g = F.l1_loss(log_magnitude, log_magnitude_pred)
            loss_d = None

        elif self.stage_id == 1:
            # Freeze encoder, train decoder with phase and discriminator loss
            loss_stft = self.stft_loss_fn(wave, wave_pred)
            loss_g, loss_d, info_d = self.discriminator(wave, wave_pred, with_info=True)
            loss_g += self.loss_stft_weight * loss_stft

            info = {
                **dict(loss_stft=loss_stft),
                **info,
                **prefix_dict("discriminator_", info_d),
            }

        if self.training:
            self.step()

        return (loss_g, loss_d, info) if with_info else (loss_g, loss_d)


class BPipeline(nn.Module):
    def __init__(
        self,
        autoencoder: STFTAutoEncoder1d,
        num_stage_steps: Union[int, Sequence[int]] = 50000,
    ):
        super().__init__()
        self.autoencoder = autoencoder

        self.num_stage_steps = to_list(num_stage_steps)
        assert len(self.num_stage_steps) == 1, "only 1 num_stage_steps required"
        self.register_buffer("step_id", torch.tensor(0))
        self.register_buffer("stage_id", torch.tensor(0))

        # Init multi resolution stft loss
        import auraloss

        scales = [2048, 1024, 512, 256, 128]
        hop_sizes, win_lengths, overlap = [], [], 0.75
        for scale in scales:
            hop_sizes += [int(scale * (1.0 - overlap))]
            win_lengths += [scale]
        self.stft_loss_fn = auraloss.freq.SumAndDifferenceSTFTLoss(
            fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths
        )

    def step(self):
        # Check if next pipeline stage has to be activated
        for i, step in enumerate(self.num_stage_steps):
            if self.step_id == step:
                self.stage_id += 1
                self.setup_stage(self.stage_id)
                print(f"Stage {self.stage_id-1} completed.")
        self.step_id += 1

    def setup_stage(self, stage_id: int) -> None:
        if stage_id == 1:
            # Freeze encoder
            freeze_model(self.autoencoder.encoder)

    def forward(
        self, wave: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict]]:
        wave_pred, info = self.autoencoder(wave, with_info=True)

        if self.stage_id == 0:
            # Train full autoencoder, only log-magnitude spectrogram l1 loss
            log_magnitude_pred = info["decoder_log_magnitude"]
            magnitude, _ = self.autoencoder.stft.encode(wave)
            log_magnitude = torch.log(magnitude)
            loss = F.l1_loss(log_magnitude, log_magnitude_pred)

        elif self.stage_id == 1:
            # Freeze encoder, train decoder with stft loss
            loss = self.stft_loss_fn(wave, wave_pred)

        if self.training:
            self.step()

        return (loss, info) if with_info else loss
