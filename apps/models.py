import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure

# activation functions
ACTIVATIONS = {
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "prelu": nn.PReLU,
}


def get_activation(name: str) -> nn.Module:
    """
    Instantiate selected activation layer by name.

    :param name: activation identifier
    :return: instantiated nn.Module activation layer
    """
    act_name = name.lower().strip()
    if act_name not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation '{name}'")
    return ACTIVATIONS[act_name]()


class ConvBlock(nn.Module):
    """
    Dual convolutional block: Conv2d -> Activation -> Conv2d -> Activation.
    """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 activation_name: str = "silu"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            get_activation(activation_name),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            get_activation(activation_name),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvAutoencoder(nn.Module):
    """
    Symmetric Convolutional Autoencoder (Encoder-Bottleneck-Decoder) with reflection padding.
    """

    def __init__(self, in_channels: int = 3,
                 base_channels: int = 16,
                 activation: str = "silu"):
        super().__init__()
        c = base_channels
        self.in_channels = in_channels
        self.activation_name = activation

        # encoder
        self.enc1 = ConvBlock(in_channels, c, activation)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ConvBlock(c, c * 2, activation)
        self.pool2 = nn.MaxPool2d(2, 2)

        # bottleneck
        self.bottleneck = ConvBlock(c * 2, c * 4, activation)

        # decoder
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2, activation)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c, activation)

        # maps back to RGB channels
        self.out_conv = nn.Sequential(
            nn.Conv2d(c, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply denoising autoencoder reconstruction.
        :param x: input PyTorch tensor [H, W], [C, H, W], or [B, H, W, C]
        :return: reconstructed PyTorch tensor with preserved device, dtype, and shape
        """
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
        if not is_channels_first:
            h, w = x.shape[1], x.shape[2]
            x = x.permute(0, 3, 1, 2)
        else:
            h, w = x.shape[2], x.shape[3]

        if x.shape[1] == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)

        # scale uint8 to float [0.0, 1.0]
        x_norm = x.float() / 255.0 if dtype == torch.uint8 else x

        # pad to nearest multiple of 16 for arbitrary BSD68 resolutions
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x_norm = nn.functional.pad(x_norm, (0, int(pad_w), 0, int(pad_h)), mode="reflect")

        # forward pass through encoder, bottleneck, and decoder
        e1 = self.enc1(x_norm)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.out_conv(d1)[:, :, :h, :w]

        # restore channels-last layout (B, H, W, C)
        if not is_channels_first:
            out = out.permute(0, 2, 3, 1)

        # return uint8 [0..255] in eval mode, float in train mode
        if not self.training:
            out = (out * 255.0).round().clamp(0.0, 255.0).to(dtype=dtype)
        else:
            out = out * 255.0

        # restore original tensor
        if ndim == 2:
            # [1, 1, H, W] -> [H, W]
            out = out.squeeze(0).squeeze(0) if (out.ndim == 4 and out.shape[1] == 1) else (out.squeeze(0)[..., 0] if not is_channels_first else out.squeeze(0)[0])
        elif ndim == 3:
            # [1, C, H, W] -> [C, H, W]
            out = out.squeeze(0)

        return out


class CompositeLoss(nn.Module):
    """
    Scale-Balanced Composite Loss for Image Restoration:
    L = alpha * (s * MSE_norm) + beta * (1 - SSIM)
    where alpha + beta = 1.0, and scale factor s = 50.0 balances MSE and SSIM magnitudes.
    """

    def __init__(self, alpha: float = 0.85,
                    beta: float = 0.15,
                    ssim_scale: float = 50.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.scale = ssim_scale
        self.mse = nn.MSELoss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, preds: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        if preds.ndim == 4 and preds.shape[-1] in (1, 3):
            preds = preds.permute(0, 3, 1, 2)
        if targets.ndim == 4 and targets.shape[-1] in (1, 3):
            targets = targets.permute(0, 3, 1, 2)

        p = preds.float() / 255.0 if preds.dtype == torch.uint8 else (preds / 255.0 if preds.max() > 1.0 else preds)
        t = targets.float() / 255.0 if targets.dtype == torch.uint8 else (targets / 255.0 if targets.max() > 1.0 else targets)

        loss_mse = self.mse(p, t) * self.scale
        loss_ssim = 1.0 - self.ssim(p, t)
        return self.alpha * loss_mse + self.beta * loss_ssim
