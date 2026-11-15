import math
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

        if self.ssim.device != p.device:
            self.to(p.device)
        if t.device != p.device:
            t = t.to(p.device)

        loss_mse = self.mse(p, t) * self.scale
        loss_ssim = 1.0 - self.ssim(p, t)
        return self.alpha * loss_mse + self.beta * loss_ssim


class ComplexExpFourierConvKAN2D(nn.Module):
    """
    2D Convolutional Kolmogorov-Arnold Network Layer with Complex Exponential Harmonic Kernel:
    phi(x) = w_base * Act(x) + Re{ sum_{k=1}^{G} C_k * exp(i * (k * omega_k * x_norm + phi_k)) }

    By Euler's identity:
        exp(i * theta_k) = cos(theta_k) + i * sin(theta_k),
        where theta_k = k * omega_k * pi * tanh(x) + phi_k.
    The spatial filter C_k = W_{k, re} - i * W_{k, im} acts over the K x K neighborhood:
        Re{ C_k * exp(i * theta_k) } = W_{k, re} * cos(theta_k) + W_{k, im} * sin(theta_k).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int] = 3,
                 grid_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 activation_name: str = "silu",
                 smooth_initialization: bool = True,
                 learnable_phase: bool = True,
                 learnable_omega: bool = True,
                 base_branch: bool = True,
                 add_bias: bool = True):
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be a positive integer!")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups!")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups!")
        if grid_size < 1:
            raise ValueError("grid_size must be at least 1!")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.grid_size = grid_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.smooth_initialization = smooth_initialization
        self.learnable_phase = learnable_phase
        self.learnable_omega = learnable_omega
        self.base_branch = base_branch

        # Harmonic frequencies k = 1, 2, ..., G shaped as (1, 1, G, 1, 1)
        harmonics = torch.arange(1, grid_size + 1, dtype=torch.float32).view(1, 1, grid_size, 1, 1)
        self.register_buffer("harmonics", harmonics)

        # Frequency scale omega_k and phase offset phi_k
        param_shape = (1, in_channels, grid_size, 1, 1)
        if learnable_omega:
            self.omega = nn.Parameter(torch.ones(*param_shape))
        else:
            self.register_buffer("omega", torch.ones(*param_shape))

        if learnable_phase:
            self.phase = nn.Parameter(torch.zeros(*param_shape))
        else:
            self.register_buffer("phase", torch.zeros(*param_shape))

        # Base residual branch
        if base_branch:
            self.base_activation = get_activation(activation_name)
            self.base_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False
            )
        else:
            self.base_activation = None
            self.base_conv = None

        # Fourier harmonic spatial convolution: 2 * G * in_channels -> out_channels
        self.fourier_conv = nn.Conv2d(
            in_channels=2 * grid_size * in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=add_bias
        )

        self._init_spectral_weights()

    def _init_spectral_weights(self) -> None:
        """
        Initialize Fourier spatial weights with 1/k^2 spectral decay
        for smooth Sobolev initialization and numerical stability.
        """
        if self.base_conv is not None:
            nn.init.kaiming_uniform_(self.base_conv.weight, a=math.sqrt(5))
        if self.fourier_conv.bias is not None:
            nn.init.zeros_(self.fourier_conv.bias)

        with torch.no_grad():
            kh, kw = self.kernel_size
            in_per_group = self.in_channels // self.groups
            fan_in = in_per_group * kh * kw

            w = self.fourier_conv.weight.view(
                self.out_channels, in_per_group, 2, self.grid_size, kh, kw
            )
            for k_idx in range(self.grid_size):
                k_val = float(k_idx + 1)
                if self.smooth_initialization:
                    attenuation = (k_val ** 2) * math.sqrt(fan_in)
                else:
                    attenuation = math.sqrt(self.grid_size * fan_in)
                std = 1.0 / max(attenuation, 1e-6)
                w[:, :, :, k_idx, :, :].normal_(mean=0.0, std=std)

    def compute_complex_exponential_basis(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate real trigonometric projections of the complex exponential basis:
        Z_k(x) = exp(i * theta_k) = cos(theta_k) + i * sin(theta_k)
        """
        x_norm = math.pi * torch.tanh(x).unsqueeze(2)  # (B, C_in, 1, H, W)
        theta = self.harmonics * self.omega * x_norm + self.phase  # (B, C_in, G, H, W)
        return torch.cos(theta), torch.sin(theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = x.shape
        re_part, im_part = self.compute_complex_exponential_basis(x)
        basis_features = torch.stack([re_part, im_part], dim=2).reshape(
            bsz, self.in_channels * 2 * self.grid_size, height, width
        )
        fourier_out = self.fourier_conv(basis_features)
        if self.base_branch and self.base_conv is not None:
            base_out = self.base_conv(self.base_activation(x))
            return base_out + fourier_out
        return fourier_out


class FourierKANBlock(nn.Module):
    """
    Fourier KAN Convolutional Block:
    ComplexExpFourierConvKAN2D -> Activation -> Conv2d -> Activation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 grid_size: int = 3,
                 activation_name: str = "silu",
                 smooth_initialization: bool = True,
                 learnable_phase: bool = True,
                 learnable_omega: bool = True,
                 base_branch: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            ComplexExpFourierConvKAN2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                grid_size=grid_size,
                padding=1,
                activation_name=activation_name,
                smooth_initialization=smooth_initialization,
                learnable_phase=learnable_phase,
                learnable_omega=learnable_omega,
                base_branch=base_branch
            ),
            get_activation(activation_name),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            get_activation(activation_name),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FourierKANAutoencoder(nn.Module):
    """
    Symmetric Fourier Kolmogorov-Arnold Autoencoder (Encoder-Bottleneck-Decoder)
    with reflection padding, multiscale pooling, skip connections, and harmonic kernels.
    """

    def __init__(self,
                 in_channels: int = 3,
                 base_channels: int = 24,
                 grid_size: int = 3,
                 activation: str = "silu",
                 smooth_initialization: bool = True,
                 learnable_phase: bool = True,
                 learnable_omega: bool = True,
                 base_branch: bool = True):
        super().__init__()
        c = base_channels
        self.in_channels = in_channels
        self.base_channels = c
        self.grid_size = grid_size
        self.activation_name = activation
        self.smooth_initialization = smooth_initialization
        self.learnable_phase = learnable_phase
        self.learnable_omega = learnable_omega
        self.base_branch = base_branch

        kwargs = dict(
            grid_size=grid_size,
            activation_name=activation,
            smooth_initialization=smooth_initialization,
            learnable_phase=learnable_phase,
            learnable_omega=learnable_omega,
            base_branch=base_branch,
        )

        # encoder
        self.enc1 = FourierKANBlock(in_channels, c, **kwargs)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = FourierKANBlock(c, c * 2, **kwargs)
        self.pool2 = nn.MaxPool2d(2, 2)

        # bottleneck
        self.bottleneck = FourierKANBlock(c * 2, c * 4, **kwargs)

        # decoder
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = FourierKANBlock(c * 4, c * 2, **kwargs)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = FourierKANBlock(c * 2, c, **kwargs)

        # maps back to RGB channels
        self.out_conv = nn.Sequential(
            nn.Conv2d(c, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply denoising Fourier KAN autoencoder reconstruction.
        :param x: input PyTorch tensor [H, W], [C, H, W], or [B, H, W, C]
        :return: reconstructed PyTorch tensor with preserved device, dtype, and shape
        """
        dtype = x.dtype
        ndim = x.ndim

        # convert into tensor
        if ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif ndim == 3:
            x = x.unsqueeze(0)

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
            out = out.squeeze(0).squeeze(0) if (out.ndim == 4 and out.shape[1] == 1) else (out.squeeze(0)[..., 0] if not is_channels_first else out.squeeze(0)[0])
        elif ndim == 3:
            out = out.squeeze(0)

        return out

