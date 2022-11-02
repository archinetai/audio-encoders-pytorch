from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .modules import AutoEncoder1d


class StackedPipeline(nn.Module):
    def __init__(
        self, autoencoders: Sequence[AutoEncoder1d], num_stage_steps: Sequence[int]
    ):
        super().__init__()
        assert_message = "len(num_stage_steps)+1 must equal len(autoencoders)"
        assert len(autoencoders) == len(num_stage_steps) + 1, assert_message

        self.autoencoders = autoencoders
        self.num_stage_steps = num_stage_steps
        self.register_buffer("step_id", torch.tensor(0))
        self.register_buffer("stage_id", torch.tensor(0))

        # Init multi resolution stft loss
        import auraloss

        scales = [2048, 1024, 512, 256, 128]
        hop_sizes, win_lengths, overlap = [], [], 0.75
        for scale in scales:
            hop_sizes += [int(scale * (1.0 - overlap))]
            win_lengths += [scale]
        self.loss_fn = auraloss.freq.SumAndDifferenceSTFTLoss(
            fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths
        )

    def step(self):
        # Check if next pipeline stage has to be activated
        for i, step in enumerate(self.num_stage_steps):
            if self.step_id == step:
                self.stage_id += 1
                self.stage_changed()
                print(f"Stage {self.stage_id-1} completed.")
        self.step_id += 1

    def stage_changed(self) -> None:
        num_stages = len(self.num_stage_steps)
        for i in range(self.stage_id):  # type: ignore
            self.autoencoders[i].requires_grad_(False)
            self.autoencoders[i].eval()
        for i in range(self.stage_id, num_stages):  # type: ignore
            self.autoencoders[i].requires_grad_(True)
            self.autoencoders[i].train()

    def encode(
        self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict]]:
        info: Dict = dict(encoders=[])
        for i in range(self.stage_id + 1):  # type: ignore
            x, info_encoder = self.autoencoders[i].encode(x, with_info=True)
            info["encoders"] += [info_encoder]
        return (x, info) if with_info else x

    def decode(
        self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict]]:
        info: Dict = dict(decoders=[])
        for i in reversed(range(self.stage_id + 1)):  # type: ignore
            x, info_decoder = self.autoencoders[i].decode(x, with_info=True)
            info["decoders"] += [info_decoder]
        return (x, info) if with_info else x

    def forward(
        self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict]]:

        z, info_encoders = self.encode(x, with_info=True)
        y, info_decoders = self.decode(z, with_info=True)
        info = dict(**info_encoders, **info_decoders, latent=z)
        loss = self.loss_fn(x, y)

        if self.training:
            self.step()

        return (loss, info) if with_info else loss
