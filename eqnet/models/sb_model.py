from typing import Any

import seisbench.models as sbm
import seisbench.util as sbu
import torch
import numpy as np
from torch import Tensor, nn
import torch.nn.functional as F

from .unet import UNet


class PhaseNet(sbm.WaveformModel):
    # TODO: Update args
    _annotate_args = sbm.WaveformModel._annotate_args.copy()
    _annotate_args["*_threshold"] = ("Detection threshold for the provided phase", 0.3)
    _annotate_args["blinding"] = (
        "Number of prediction samples to discard on each side of each window prediction",
        (0, 0),
    )
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 0.5)

    def __init__(
        self,
        log_scale=True,
        add_polarity=True,
        add_event=True,
        sampling_rate=100,
        **kwargs,
    ) -> None:
        super().__init__(
            in_samples=1024,
            output_type="array",
            pred_sample=(0, 1024),
            labels=["N", "P", "S", "Polarity", "Event center", "Event time"],
            sampling_rate=sampling_rate,
            **kwargs,
        )
        self.add_event = add_event
        self.add_polarity = add_polarity

        self.backbone = UNet(
                channels=3,
                dim=16,
                out_dim=32,
                log_scale=log_scale,
                add_polarity=add_polarity,
                add_event=add_event,
            )

        self.phase_picker = UNetHead(32, 3, feature_name="phase")
        if self.add_event:
            self.event_detector = UNetHead(32, 1, feature_name="event")
            self.event_timer = EventHead(32, 1, feature_name="event")
        if self.add_polarity:
            self.polarity_picker = UNetHead(32, 1, feature_name="polarity")

    def forward(self, data: Tensor, logits: bool = False) -> dict[str, Tensor]:
        # data: (batch, channel, station, time)
        features = self.backbone(data)
        # features: (batch, channel, station, time)

        output_phase = self.phase_picker(features, logits)
        output = {"phase": output_phase}
        if self.add_event:
            output_event_center = self.event_detector(features, logits)
            output["event_center"] = output_event_center
            output_event_time = self.event_timer(features)
            output["event_time"] = output_event_time
        if self.add_polarity:
            output_polarity = self.polarity_picker(features, logits)
            output["polarity"] = (output_polarity - 0.5) * 2.0  # Convert to -1, 1

        return output

    def _get_in_pred_samples(self, block: np.ndarray) -> tuple[int, tuple[int, int]]:
        in_samples = 2 ** int(np.log2(block.shape[-1]))  # The largest power of 2 below the block shape
        in_samples = min(max(in_samples, 2 ** 10), 2 ** 20)  # Enforce upper and lower bounds
        pred_sample = (0, in_samples)
        return in_samples, pred_sample

    def annotate_batch_pre(
            self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        return batch.unsqueeze(-2)  # Add fake station dimension

    def annotate_batch_post(
            self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        y_phase = batch["phase"][..., 0, :]
        y_polarity = batch["polarity"][..., 0, :]
        y_center = torch.repeat_interleave(batch["event_center"][..., 0, :], 16, dim=-1)
        y_time = torch.repeat_interleave(batch["event_time"][..., 0, :], 16, dim=-1)

        y_full = torch.concat([y_phase, y_polarity, y_center, y_time], dim=1)

        prenan, postnan = argdict.get(
            "blinding", self._annotate_args.get("blinding")[1]
        )
        if prenan > 0:
            y_full[..., :prenan] = np.nan
        if postnan > 0:
            y_full[..., -postnan:] = np.nan

        return torch.transpose(y_full, -1, -2)

    def classify_aggregate(self, annotations, argdict) -> sbu.ClassifyOutput:
        """
        Converts the annotations to discrete picks (mode "pick") or events (mode "event").

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks
        """
        mode = argdict.get("mode", "pick")
        if mode == "pick":
            return self.classify_aggregate_pick(annotations, argdict)
        elif mode == "event":
            return self.classify_aggregate_event(annotations, argdict)
        else:
            raise NotImplementedError(f"Mode '{mode}' unknown")

    def classify_aggregate_pick(self, annotations, argdict) -> sbu.ClassifyOutput:
        """
        Converts the annotations to discrete thresholds using
        :py:func:`~seisbench.models.base.WaveformModel.picks_from_annotations`.
        Trigger onset thresholds for picks are derived from the argdict at keys "[phase]_threshold".

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks
        """
        picks = sbu.PickList()
        for phase in "PS":
            picks += self.picks_from_annotations(
                annotations.select(channel=f"{self.__class__.__name__}_{phase}"),
                argdict.get(
                    f"{phase}_threshold", self._annotate_args.get("*_threshold")[1]
                ),
                phase,
            )

        picks = sbu.PickList(sorted(picks))

        return sbu.ClassifyOutput(self.name, picks=picks)

    def classify_aggregate_event(self, annotations, argdict) -> sbu.ClassifyOutput:
        """

        """
        # TODO: Document
        # TODO: Implement

        return sbu.ClassifyOutput(self.name)

class UNetHead(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=(1, 1), padding=(0, 0), feature_name: str = "phase"
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.feature_name = feature_name
        self.layers = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding
        )

    def forward(self, features, logits: bool = False) -> Tensor:
        x = features[self.feature_name]
        x = self.layers(x)
        if logits:
            return x
        else:
            return F.softmax(x, dim=1) if self.out_channels > 1 else torch.sigmoid(x)


class EventHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(1, 1),
        padding=(0, 0),
        scaling=1000.0,
        feature_name: str = "event",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.feature_name = feature_name
        self.scaling = scaling
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
        )

    def forward(self, features):
        x = features[self.feature_name]
        return self.layers(x) * self.scaling
