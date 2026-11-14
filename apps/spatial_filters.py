import cv2
import torch
import numpy as np
import torch.nn as nn
import apps.main as mn
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanSquaredError


def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate model on a DataLoader.
    """
    model.eval()
    device = mn.get_device()
    model.to(device)

    mse_metric = MeanSquaredError().to(device)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x).contiguous()
            mse_metric.update(preds, y)

    return mse_metric.compute().item()


class GaussianFilter(nn.Module):
    """
    PyTorch wrapper for Gaussian filtering using cv2.GaussianBlur.
    Operates without trainable weights, accepting and returning PyTorch tensors.
    """

    def __init__(self, ksize: int = 3, sigma_x: float = 1.0):
        """
        Initialize Gaussian filter with aperture size and standard deviation.
        :param ksize: kernel aperture size (positive and odd integer)
        :param sigma_x: Gaussian kernel standard deviation along X-axis
        """
        super().__init__()
        ksize = max(1, int(ksize))
        if ksize % 2 == 0:
            ksize += 1
        self.ksize = ksize
        self.sigma_x = float(sigma_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial Gaussian filtering to input tensor.
        :param x: input PyTorch tensor [H, W], [C, H, W], or [B, C, H, W]
        :return: filtered PyTorch tensor with preserved device, dtype, and shape
        """
        device = x.device
        dtype = x.dtype
        ndim = x.ndim

        # convert into tensor
        if ndim == 2:
            # [H, W] -> [1, 1, H, W]
            x = x.unsqueeze(0).unsqueeze(0)
        elif ndim == 3:
            # [C, H, W] -> [1, C, H, W]
            x = x.unsqueeze(0)

        # if the path has dimension (B, H, W, C), then convert into (B, C, H, W) for PyTorch
        is_channels_first = x.shape[1] in (1, 3)

        np_imgs = x.detach().cpu().numpy()
        if is_channels_first:
            np_imgs = np.ascontiguousarray(np_imgs.transpose(0, 2, 3, 1))

        # apply Gaussian filter
        filtered = [cv2.GaussianBlur(img, (self.ksize, self.ksize), self.sigma_x) for img in np_imgs]
        out_np = np.stack(filtered, axis=0)

        # cv2.GaussianBlur strips trailing singleton channel dimension for grayscale (H, W, 1) -> (H, W)
        if out_np.ndim == 3:
            out_np = np.expand_dims(out_np, axis=-1)

        # convert back to PyTorch tensor and restore device and dtype
        out_tensor = torch.from_numpy(out_np).to(device=device, dtype=dtype)
        if is_channels_first:
            out_tensor = out_tensor.permute(0, 3, 1, 2).contiguous()

        # restore original tensor
        if ndim == 2:
            # [1, 1, H, W] -> [H, W]
            out_tensor = out_tensor.squeeze(0).squeeze(0)
        elif ndim == 3:
            # [1, C, H, W] -> [C, H, W]
            out_tensor = out_tensor.squeeze(0)

        return out_tensor


class MedianFilter(nn.Module):
    """
    PyTorch wrapper for spatial median filtering using cv2.medianBlur.
    Operates without trainable weights, accepting and returning PyTorch tensors.
    """

    def __init__(self, ksize: int = 3) -> None:
        """
        Initialize median filter with kernel aperture size.
        :param ksize: aperture linear size (positive and odd integer, >= 3)
        """
        super().__init__()
        ksize = max(1, int(ksize))
        if ksize % 2 == 0:
            ksize += 1
        self.ksize = ksize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply non-linear median filtering to input tensor.
        :param x: input PyTorch tensor [H, W], [C, H, W], or [B, C, H, W]
        :return: filtered PyTorch tensor with preserved device, dtype, and shape
        """
        device = x.device
        dtype = x.dtype
        ndim = x.ndim

        # convert into tensor
        if ndim == 2:
            # [H, W] -> [1, 1, H, W]
            x = x.unsqueeze(0).unsqueeze(0)
        elif ndim == 3:
            # [C, H, W] -> [1, C, H, W]
            x = x.unsqueeze(0)

        # if the path has dimension (B, H, W, C), then convert into (B, C, H, W) for PyTorch
        is_channels_first = x.shape[1] in (1, 3)

        np_imgs = x.detach().cpu().numpy()
        if is_channels_first:
            np_imgs = np.ascontiguousarray(np_imgs.transpose(0, 2, 3, 1))

        # apply median filter
        filtered = [cv2.medianBlur(img, self.ksize) for img in np_imgs]
        out_np = np.stack(filtered, axis=0)

        # cv2.medianBlur strips trailing singleton channel dimension for grayscale (H, W, 1) -> (H, W)
        if out_np.ndim == 3:
            out_np = np.expand_dims(out_np, axis=-1)

        # convert back to PyTorch tensor and restore device and dtype
        out_tensor = torch.from_numpy(out_np).to(device=device, dtype=dtype)
        if is_channels_first:
            out_tensor = out_tensor.permute(0, 3, 1, 2).contiguous()

        # restore original tensor
        if ndim == 2:
            # [1, 1, H, W] -> [H, W]
            out_tensor = out_tensor.squeeze(0).squeeze(0)
        elif ndim == 3:
            # [1, C, H, W] -> [C, H, W]
            out_tensor = out_tensor.squeeze(0)

        return out_tensor


class BilateralFilter(nn.Module):
    """
    PyTorch wrapper for edge-preserving bilateral filtering using cv2.bilateralFilter.
    Operates without trainable weights, accepting and returning PyTorch tensors.
    """

    def __init__(self, d: int = 5, sigma_color: float = 0.1, sigma_space: float = 1.5) -> None:
        """
        Initialize bilateral filter with neighborhood diameter and photometric/geometric sigmas.
        :param d: diameter of each pixel neighborhood (typically 5-9)
        :param sigma_color: filter sigma in color/intensity space (for normalized [0.0, 1.0] tensors)
        :param sigma_space: filter sigma in coordinate space
        """
        super().__init__()
        self.d = max(1, int(d))
        self.sigma_color = float(sigma_color)
        self.sigma_space = float(sigma_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply edge-preserving bilateral filtering to input tensor.
        :param x: input PyTorch tensor [H, W], [C, H, W], or [B, C, H, W]
        :return: filtered PyTorch tensor with preserved device, dtype, and shape
        """
        device = x.device
        dtype = x.dtype
        ndim = x.ndim

        # convert into tensor
        if ndim == 2:
            # [H, W] -> [1, 1, H, W]
            x = x.unsqueeze(0).unsqueeze(0)
        elif ndim == 3:
            # [C, H, W] -> [1, C, H, W]
            x = x.unsqueeze(0)

        # if the path has dimension (B, H, W, C), then convert into (B, C, H, W) for PyTorch
        is_channels_first = x.shape[1] in (1, 3)

        np_imgs = x.detach().cpu().numpy()
        if is_channels_first:
            np_imgs = np.ascontiguousarray(np_imgs.transpose(0, 2, 3, 1))

        # apply bilateral filter
        filtered = [cv2.bilateralFilter(img, self.d, self.sigma_color, self.sigma_space) for img in np_imgs]
        out_np = np.stack(filtered, axis=0)

        # cv2.bilateralFilter strips trailing singleton channel dimension for grayscale (H, W, 1) -> (H, W)
        if out_np.ndim == 3:
            out_np = np.expand_dims(out_np, axis=-1)

        # convert back to PyTorch tensor and restore device and dtype
        out_tensor = torch.from_numpy(out_np).to(device=device, dtype=dtype)
        if is_channels_first:
            out_tensor = out_tensor.permute(0, 3, 1, 2).contiguous()

        # restore original tensor
        if ndim == 2:
            # [1, 1, H, W] -> [H, W]
            out_tensor = out_tensor.squeeze(0).squeeze(0)
        elif ndim == 3:
            # [1, C, H, W] -> [C, H, W]
            out_tensor = out_tensor.squeeze(0)

        return out_tensor
