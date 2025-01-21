from typing import Any

import seisbench.models as sbm
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .unet import UNet


class PhaseNet(sbm.WaveformModel):
    def __init__(
        self,
        log_scale=True,
        add_polarity=True,
        add_event=True,
        sampling_rate=100,
    ) -> None:
        super().__init__(
            in_samples=4096,
            output_type="array",
            pred_sample=(0, 4096),
            labels=["N", "P", "S", "None", "Up", "Down"],
            sampling_rate=sampling_rate,
        )
        self.add_event = add_event
        self.add_polarity = add_polarity

        self.backbone = UNet(log_scale=log_scale, add_polarity=add_polarity, add_event=add_event)

        self.phase_picker = UNetHead(16, 3, feature_names="phase")
        if self.add_event:
            self.event_detector = UNetHead(32, 1, feature_names="event")
            self.event_timer = EventHead(32, 1, feature_names="event")
        if self.add_polarity:
            self.polarity_picker = UNetHead(16, 3, feature_names="polarity")

    def forward(self, data: Tensor, logits: bool = False) -> dict[str, Tensor]:
        # data: (batch, channel, time, station)
        features = self.backbone(data)
        # features: (batch, station, channel, time)

        output_phase = self.phase_picker(features, logits)
        output = {"phase": output_phase}
        if self.add_event:
            output_event_center = self.event_detector(features, logits)
            output["event_center"] = output_event_center
            output_event_time = self.event_timer(features)
            output["event_time"] = output_event_time
        if self.add_polarity:
            output_polarity = self.polarity_picker(features, logits)
            output["polarity"] = output_polarity

        return output

    def annotate_batch_pre(
            self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        return batch.unsqueeze(-1)  # Add fake station dimension

    def annotate_batch_post(
            self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        y_phase = batch["phase"][..., 0]
        y_polarity = batch["polarity"][..., 0]

        y_full = torch.concat([y_phase, y_polarity], dim=1)

        return torch.transpose(y_full, -1, -2)


class UNetHead(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=(7, 1), padding=(3, 0), feature_names: str = "phase"
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.feature_names = feature_names
        self.layers = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding
        )

    def forward(self, features, logits: bool = False) -> Tensor:
        x = features[self.feature_names]
        x = self.layers(x)
        if logits:
            return x
        else:
            return F.softmax(x, dim=1)


class EventHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(7, 1),
        padding=(3, 0),
        scaling=1000.0,
        feature_names: str = "event",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.feature_names = feature_names
        self.scaling = scaling
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
        )

    def forward(self, features):
        x = features[self.feature_names]
        return self.layers(x) * self.scaling
