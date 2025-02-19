import numpy as np
import torch
from scipy.signal import argrelextrema, savgol_filter
import ruptures as rpt


class InvalidSampleException(Exception):
    pass


class InvalidLengthException(InvalidSampleException):
    pass


class MADNormalize(object):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - torch.median(x)  # , dim=1, keepdim=True)[0]
        mad = torch.median(torch.abs(centered))  # , dim=1, keepdim=True)[0]
        return centered / (1.4826 * mad)


class RandomWindow(object):
    def __init__(self, window_size: int, leftpad: int = 0) -> None:
        self.window_size = window_size
        self.leftpad = leftpad

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # if too short, pad with zeros
        if len(x) < self.window_size + self.leftpad:
            x = torch.nn.functional.pad(x, (0, self.leftpad + self.window_size - len(x)), mode="constant", value=0)

        if len(x) == self.window_size + self.leftpad:
            return x[self.leftpad :]
        else:
            start = np.random.randint(self.leftpad, len(x) - self.window_size)

            return x[start : start + self.window_size]


class ScaleToFixedLength(object):
    def __init__(self, length: int) -> None:
        self.length = length

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # interpolate to fixed length
        oldtype = x.dtype
        x = x.float()
        x = torch.nn.functional.interpolate(x.unsqueeze(0).unsqueeze(0), size=self.length, mode="linear").squeeze()
        x = x.type(oldtype)
        return x


class FixedWindow(object):
    def __init__(self, window_size: int, leftpad: int = 0) -> None:
        self.window_size = window_size
        self.leftpad = leftpad

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # if too short, pad with zeros
        if len(x) < self.window_size + self.leftpad:
            x = torch.nn.functional.pad(x, (0, self.leftpad + self.window_size - len(x)), mode="constant", value=0)

        if len(x) == self.window_size + self.leftpad:
            return x[self.leftpad :]
        else:
            start = self.leftpad
            return x[start : start + self.window_size]


class RemoveOutliers(object):
    def __init__(self, std: float = 3, window_size: int = 800) -> None:
        self.std = std
        self.window_size = window_size

    def __call__(self, x: torch.Tensor, plot: bool = False) -> torch.Tensor:
        if self.std == 0:
            return x

        squueze = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squueze = True

        windows = x.abs().unfold(1, self.window_size, 1)
        threshold = windows.mean(2) + self.std * windows.std(dim=2)

        # pad to original size
        threshold = torch.nn.functional.pad(threshold, (self.window_size // 2, self.window_size // 2), mode="replicate")
        threshold = threshold[:, : x.shape[-1]]

        outliers = torch.abs(x) > threshold

        if plot:
            self._plot(x, threshold, outliers)

        x[outliers] = x[outliers].sign() * threshold[outliers]
        x[torch.abs(x) > threshold] = 0

        if squueze:
            x = x.squeeze()
        return x

    def _plot(self, x: torch.Tensor, threshold: torch.Tensor, outliers: torch.Tensor) -> None:
        import plotly.express as px
        import plotly.graph_objects as go

        i = 0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=torch.arange(len(x[0])), y=x[0], mode="lines", line_color="blue", name="signal"))
        fig.add_trace(
            go.Scatter(
                x=torch.arange(len(threshold[i])),
                y=threshold[i],
                name=f"threshold at {self.std} std",
                mode="lines",
                line_color="red",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=torch.arange(len(x[i]))[outliers[i]],
                y=x[i][outliers[i]],
                mode="markers",
                name="outliers",
            )
        )
        fig.show()


class FilterByLength(object):
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if len(x) > self.max_length:
            raise InvalidLengthException(f"Length of {len(x)} is larger than max length of {self.max_length}")
        return x


class MaximaAligner(object):
    def __init__(
        self,
        target_pos_first_maxima: int = 100,
        target_pos_second_maxima: int = 5000,
        savgol_width: int = 500,
        savgol_order: int = 6,
        max_threshold: float = 1,
    ) -> None:
        self.target_pos_first_maxima = target_pos_first_maxima
        self.target_pos_second_maxima = target_pos_second_maxima
        self.savgol_width = savgol_width
        self.savgol_order = savgol_order
        self.max_threshold = max_threshold

    def __call__(self, x: torch.Tensor, plot: bool = False) -> torch.Tensor:
        assert x.ndim == 1
        dtype = x.dtype
        x = x.numpy()

        x_smooth = savgol_filter(x, self.savgol_width, self.savgol_order)
        idx_above_thres = np.where(x_smooth > self.max_threshold)[0]

        local_maximas = argrelextrema(x_smooth[idx_above_thres], np.greater)[0]
        if len(local_maximas) < 2:
            return torch.from_numpy(x).type(dtype)
            # raise InvalidSampleException("Could not find two local maximas")
        first_maxima = idx_above_thres[local_maximas[0]]
        second_maxima = idx_above_thres[local_maximas[1]]

        stretch = (self.target_pos_second_maxima - self.target_pos_first_maxima) / (second_maxima - first_maxima)
        delta = self.target_pos_first_maxima - first_maxima * stretch

        idx = np.arange(len(x))
        y = np.interp(idx / stretch - delta, idx, x, left=0, right=0)

        if plot:
            import matplotlib.pyplot as plt

            y_smooth = savgol_filter(y, 500, 6)

            plt.figure(figsize=(20, 10))
            plt.plot(x, alpha=0.5, c="k")
            plt.plot(y_smooth, c="r")
            plt.plot(y, c="g")
            plt.scatter(first_maxima, x_smooth[first_maxima], c="r")
            plt.scatter(self.target_pos_first_maxima, x_smooth[first_maxima], c="g")
            plt.scatter(second_maxima, x_smooth[second_maxima], c="r")
            plt.scatter(self.target_pos_second_maxima, x_smooth[second_maxima], c="g")

            plt.show()

        return torch.from_numpy(y).type(dtype)


class Rupture(object):
    def __init__(self, part, model="l2") -> None:
        part = part.upper().strip()
        assert part in ["DNA", "RNA"]
        self.part = part
        self.model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        algo = rpt.Dynp(model=self.model).fit(x.numpy())
        bkp = algo.predict(n_bkps=1)[0]

        if self.part == "RNA":
            seg_data = x[bkp:]
        elif self.part == "DNA":
            seg_data = x[:bkp]

        return seg_data
