import cv2
import keras
import torch
import random
import numpy as np
import scipy.fft as sp_fft
import apps.reporter as rpt
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from IPython.display import display
from pandas import Series, DataFrame
from skimage.util import random_noise
from IPython.core.display import Markdown
from pandas.io.formats.style import Styler
from torch.utils.data import IterableDataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


def get_device() -> torch.device:
    """
    Detect and return the optimal available PyTorch hardware acceleration device.
    Priority hierarchy: NVIDIA CUDA -> Apple Silicon MPS -> CPU.
    :return: torch.device instance configured for the fastest available compute backend
    """
    # resolve hardware acceleration backend: cuda -> mps -> cpu
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train_test_split_by_order(array,
                              test_size: float = 0.2) -> tuple:
    """
    Split arrays or matrices into order to train and test subset
    :param array: input data
    :param test_size: the proportion of the dataset for the test split
    :return: tuple of train and test arrays
    """
    n_samples = len(array)
    if n_samples <= 1:
        raise ValueError("The array must contain more than one element!")

    if test_size <= 0.0 or 1.0 <= test_size:
        raise ValueError("Invalid test size, it should be between 0.0 and 1.0!")

    n_test = int(n_samples * test_size)
    if n_test == 0:
        n_test = 1

    if isinstance(array, (Series, DataFrame)):
        train_array = array.iloc[:-n_test]
        test_array = array.iloc[-n_test:]
    else:
        train_array = array[:-n_test]
        test_array = array[-n_test:]

    return train_array, test_array


def calculate_integer_layout(total_size: int,
                             tile_size: int,
                             min_overlap: float = 0.0,
                             include: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate 1D start and end coordinates for slicing with uniform step and minimal overlap
    :param total_size: full size of the dimension (image height or width)
    :param tile_size: target patch size (e.g. 128)
    :param min_overlap: minimal overlap (ratio < 1.0 or absolute pixels >= 1.0)
    :param include: if True, returns [start, end) for Python slicing; if False, returns [start, end] inclusive pixel index
    :return: tuple of start and end positions as numpy int32 arrays
    """
    if total_size <= tile_size:
        end_pos = total_size if include else total_size - 1
        return np.array([0], dtype=np.int32), np.array([end_pos], dtype=np.int32)

    # normalization min overlap
    if min_overlap <= 0.0:
        overlap_px = 0
    elif min_overlap < 1.0:
        overlap_px = int(np.floor(tile_size * min_overlap))
    else:
        overlap_px = int(min_overlap)

    # protection overlap
    if overlap_px >= tile_size:
        overlap_px = tile_size - 1

    useful_size = tile_size - overlap_px
    total_shift = total_size - tile_size

    # calculate intervals for full overlap
    intervals = int(np.ceil(total_shift / useful_size))
    n = intervals + 1

    # step and reminder
    base_step, remainder = divmod(total_shift, intervals)

    # generate coordinates
    start_positions = []
    end_positions = []
    current_pos = 0

    for i in range(n):
        start_positions.append(current_pos)
        end_pos = current_pos + tile_size - (0 if include else 1)
        end_positions.append(end_pos)

        step = base_step + (1 if i < remainder else 0)
        current_pos += step
    return np.asarray(start_positions, dtype=np.int32), np.asarray(end_positions, dtype=np.int32)


def image_slicer(image: np.ndarray,
                 patch_size: tuple[int, int],
                 min_overlap: float = 0.0,
                 is_flat: bool = True) -> list[list[np.ndarray]] | list[np.ndarray]:
    """
    Crop input image into 2D grid [row][col] or flat list of fixed-size patches.
    :param image: input image as numpy ndarray (H, W) or (H, W, C)
    :param patch_size: target patch size as (height, width)
    :param min_overlap: minimal overlap (ratio < 1.0 or absolute pixels >= 1.0)
    :param is_flat: if True, returns flat 1D list of patches; if False, returns 2D grid [rows][cols]
    :return: 2D grid or 1D list of cropped patches
    """
    # get size of image
    img_height, img_width = image.shape[:2]
    patch_height, patch_width = patch_size

    # calculate coordinates for slicing
    y_starts, y_ends = calculate_integer_layout(img_height, patch_height, min_overlap=min_overlap)
    x_starts, x_ends = calculate_integer_layout(img_width, patch_width, min_overlap=min_overlap)

    # 2D list of cropped patches: [rows][cols]
    patches = []

    # slicing
    for y1, y2 in zip(y_starts, y_ends):
        row_patches = []
        for x1, x2 in zip(x_starts, x_ends):
            cropped = image[y1:y2, x1:x2]
            row_patches.append(cropped)
        patches.append(row_patches)

    # flatten to 1D if requested
    if is_flat:
        return [patch for row in patches for patch in row]

    return patches


def calculate_fft_spectrum(image: np.ndarray,
                           workers: int = -1) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Compute 2D Fast Fourier Transform and extract amplitudes, phases, and spatial frequencies.
    :param image: input image as 2D (H, W) or 3D (H, W, C) numpy ndarray
    :param workers: number of parallel CPU worker threads (-1 for all cores)
    :return: (log_amplitudes, raw_amplitudes, phases, (freq_y, freq_x))
    """
    image_2d = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    h, w = image_2d.shape[:2]

    spectrum_complex = sp_fft.fftshift(sp_fft.fft2(image_2d.astype(np.float32), workers=workers))

    raw_amplitudes = np.asanyarray(np.abs(spectrum_complex).astype(np.float32))
    log_amplitudes = np.asanyarray(np.log1p(raw_amplitudes).astype(np.float32))

    phases = np.asanyarray(np.angle(spectrum_complex).astype(np.float32))

    freq_y = np.asanyarray(sp_fft.fftshift(sp_fft.fftfreq(h, d=1.0)).astype(np.float32))
    freq_x = np.asanyarray(sp_fft.fftshift(sp_fft.fftfreq(w, d=1.0)).astype(np.float32))

    return log_amplitudes, raw_amplitudes, phases, (freq_y, freq_x)


def calculate_metrics(original_image: np.ndarray,
                      changed_image: np.ndarray,
                      data_range: float = 255.0,
                      caption: str = "Metrics",
                      is_print: bool = True) -> Styler:
    """
    Calculate and report comprehensive image denoising and restoration quality metrics.
    :param original_image: ground truth reference image (uint8 or float)
    :param changed_image: predicted / restored / filtered image
    :param data_range: peak signal value for PSNR and SSIM dynamic range (default: 255.0)
    :param caption: title caption for the printed report
    :param is_print: if True, prints the formatted report via Reporter
    :return: DataFrame Styler with computed metrics
    """
    rp = rpt.Reporter()
    rp.tolerance = 4

    # set similar dimension
    if original_image.ndim < changed_image.ndim:
        original_image = np.repeat(original_image[..., np.newaxis], 3, axis=-1)
    elif original_image.ndim > changed_image.ndim:
        changed_image = np.repeat(changed_image[..., np.newaxis], 3, axis=-1)

    original_flat = original_image.flatten()
    changed_flat = changed_image.flatten()

    # Peak Signal-To-Noise Ratio
    psnr = peak_signal_noise_ratio(original_image, changed_image, data_range=data_range)
    rp.add_item("PSNR, dB", rp.format_value(psnr))

    # Structural Similarity Index
    channel_axis = -1 if original_image.ndim == 3 else None
    ssim = structural_similarity(original_image, changed_image, data_range=data_range, channel_axis=channel_axis)
    rp.add_item("SSIM", rp.format_value(ssim))

    # Mean Absolute Error
    mae = mean_absolute_error(original_flat, changed_flat)
    rp.add_item("MAE", rp.format_value(mae))  # type: ignore

    # Mean Squared Error
    mse = mean_squared_error(original_flat, changed_flat)
    rp.add_item("MSE", rp.format_value(mse))  # type: ignore

    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    rp.add_item("RMSE", rp.format_value(rmse))  # type: ignore

    # R2 - coefficient of determination
    r2 = r2_score(original_flat, changed_flat)
    rp.add_item(f"R²\n(coefficient of determination)", rp.format_value(r2))

    # Mean Absolute Percentage Error
    non_zero_mask = original_flat > 0
    if np.any(non_zero_mask):
        mape = mean_absolute_percentage_error(original_flat[non_zero_mask], changed_flat[non_zero_mask])
        rp.add_item("MAPE\n(excluded zero values)", rp.format_value(mape))
    else:
        rp.add_item("MAPE\n(excluded zero values)", "N/A")

    df = rp.get_pd_report(caption)

    # Print results
    if is_print:
        rp.print_pd_report(caption)

    return df


class TqdmProgressCallback(keras.callbacks.Callback):
    def __init__(self, total_epochs=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.pbar = None

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(total=self.total_epochs, desc="Total training", leave=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = float(logs.get('loss', 0.0))
        acc = float(logs.get('accuracy', 0.0))
        status = float((epoch + 1.0) / self.total_epochs)
        self.pbar.write(f"Epoch: {epoch + 1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}, Status: {status:.2%}")
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        if self.pbar:
            self.pbar.close()

    def on_predict_begin(self, logs=None):
        total_steps = self.params.get('steps', None)
        self.pbar = tqdm(total=total_steps, desc="Predicting", leave=True)

    def on_predict_batch_end(self, batch, logs=None):
        if self.pbar:
            self.pbar.update(1)

    def on_predict_end(self, logs=None):
        if self.pbar:
            self.pbar.close()


class GeometricAugmentor:
    """
    Random D4 dihedral symmetry geometric augmentor for 2D patches and images.
    Applies random horizontal flips, vertical flips, and orthogonal rotations (0, 90, 180, 270 deg).
    """

    def __init__(self, p_flip_h: float = 0.5, p_flip_v: float = 0.5, p_rot: float = 0.75, seed: int = -1):
        """
        Initialize geometric augmentor with probability clipping.
        :param p_flip_h: probability of horizontal flip (clipped to [0.0, 1.0])
        :param p_flip_v: probability of vertical flip (clipped to [0.0, 1.0])
        :param p_rot: probability of orthogonal rotation (clipped to [0.0, 1.0])
        :param seed: pseudo-random seed for standalone usage (-1 disables deterministic seed)
        """
        self.p_flip_h = float(np.clip(p_flip_h, 0.0, 1.0))
        self.p_flip_v = float(np.clip(p_flip_v, 0.0, 1.0))
        self.p_rot = float(np.clip(p_rot, 0.0, 1.0))
        self.seed = seed
        # set default random generator
        self.rng = np.random.default_rng(seed if seed > -1 else None)

    def reset_seed(self, seed: int | None = None) -> None:
        """
        Reset pseudo-random generator state for reproducible multi-run experiments.
        :param seed: optional new seed; if None, resets using initial seed
        """
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed if self.seed > -1 else None)

    def __call__(self, image: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Apply random geometric transformations to input image or patch.
        :param image: input image as numpy ndarray (H, W) or (H, W, C)
        :param rng: optional injected numpy random Generator for synchronized pipeline
        :return: augmented image as contiguous numpy ndarray
        """
        # select active generator
        active_rng = rng if rng is not None else self.rng

        # horizontal flip
        if self.p_flip_h > 0.0 and active_rng.random() < self.p_flip_h:
            image = np.fliplr(image)

        # vertical flip
        if self.p_flip_v > 0.0 and active_rng.random() < self.p_flip_v:
            image = np.flipud(image)

        # orthogonal rotation (90, 180, 270 degrees)
        if self.p_rot > 0.0 and active_rng.random() < self.p_rot:
            k = int(active_rng.integers(1, 4))
            image = np.rot90(image, k=k)

        # ensure contiguous array in memory
        return np.ascontiguousarray(image)


class NoiseInjector:
    """
    Image noise injector using scikit-image standard primitives and custom 2D harmonic stripes.
    Supports 6 noise models: AWGN, Salt & Pepper, Speckle, Poisson, Periodic Striped, Hybrid Mix,
    and an identity pass-through mode ('clean').
    """

    STATISTICAL_MODES = ('gaussian', 'salt_and_pepper', 'speckle', 'poisson')
    AVAILABLE_MODES = ('gaussian', 'salt_and_pepper', 'speckle', 'poisson', 'periodic', 'hybrid', 'clean')

    def __init__(self, mode: str = 'random',
                 sigma: float | tuple[float, float] = (10.0, 50.0),
                 p_sp: float | tuple[float, float] = (0.01, 0.05),
                 speckle_var: float | tuple[float, float] = (0.01, 0.05),
                 stripe_amplitude: float | tuple[float, float] = (15.0, 40.0),
                 stripe_frequency: float | tuple[float, float] = (4.0, 12.0),
                 seed: int = -1):
        """
        Image noise injector using scikit-image standard primitives and custom 2D harmonic stripes.
        Supports 6 noise models: AWGN, Salt & Pepper, Speckle, Poisson, Periodic Striped, Hybrid Mix,
         and an identity pass-through mode ('clean').
        """
        self.mode = mode
        self.sigma = sigma
        self.p_sp = p_sp
        self.speckle_var = speckle_var
        self.stripe_amplitude = stripe_amplitude
        self.stripe_frequency = stripe_frequency
        self.seed = seed
        self.rng = np.random.default_rng(seed if seed > -1 else None)

    def reset_seed(self, seed: int | None = None) -> None:
        """
        Reset internal pseudo-random number generator for scientific reproducibility.
        """
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed if self.seed > -1 else None)

    @staticmethod
    def _sample(val: float | tuple[float, float], rng: np.random.Generator) -> float:
        """
        Sample scalar value from interval or return fixed constant.
        """
        return float(rng.uniform(val[0], val[1])) if isinstance(val, (tuple, list)) else float(val)

    def _add_periodic(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Harmonic 2D periodic striped noise: A * sin(2pi * (fx*x + fy*y) + phi).
        """
        amp = self._sample(self.stripe_amplitude, rng)
        freq = self._sample(self.stripe_frequency, rng)
        theta = float(rng.uniform(0.0, np.pi))
        phi = float(rng.uniform(0.0, 2.0 * np.pi))

        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]
        fx = (freq * np.cos(theta)) / float(w)
        fy = (freq * np.sin(theta)) / float(h)

        stripe = amp * np.sin(2.0 * np.pi * (fx * x + fy * y) + phi)
        if image.ndim == 3:
            stripe = stripe[..., np.newaxis]

        return np.clip(image.astype(np.float32) + stripe, 0, 255).astype(image.dtype)

    def _add_statistical_noise(self, image: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
        """
        Add noise using scikit-image primitives and map back to uint8 [0, 255].
        """
        match mode:
            case 'gaussian':
                sigma = self._sample(self.sigma, rng)
                var = (sigma / 255.0) ** 2
                noisy = random_noise(image, mode='gaussian', var=var, rng=rng)
            case 'salt_and_pepper':
                amount = self._sample(self.p_sp, rng)
                noisy = random_noise(image, mode='s&p', amount=amount, rng=rng)
            case 'speckle':
                var = self._sample(self.speckle_var, rng)
                noisy = random_noise(image, mode='speckle', var=var, rng=rng)
            case 'poisson':
                noisy = random_noise(image, mode='poisson', rng=rng)
            case _:
                return image

        return (noisy * 255.0).astype(image.dtype)

    def __call__(self, image: np.ndarray, mode: str | None = None,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Inject configured or stochastic noise into the input image patch.
        """
        # select active generator
        active_rng = rng if rng is not None else self.rng
        # select active mode
        if mode is not None:
            active_mode = mode
        else:
            # random mode
            active_mode = str(active_rng.choice(self.AVAILABLE_MODES)) if self.mode == 'random' else self.mode

        if active_mode not in self.AVAILABLE_MODES:
            raise ValueError(f"Unknown noise mode: {active_mode}")

        match active_mode:
            case 'hybrid':
                image = self._add_periodic(image, active_rng)
                # update mode to add skimage noise
                active_mode = str(active_rng.choice(self.STATISTICAL_MODES))
                image = self._add_statistical_noise(image, active_mode, active_rng)
            case 'periodic':
                image = self._add_periodic(image, active_rng)
            case 'clean':
                image = image.copy()
            case _:
                image = self._add_statistical_noise(image, active_mode, active_rng)

        # ensure contiguous array in memory
        return np.ascontiguousarray(image)


class StreamPatchDataset(IterableDataset):
    """
    Streaming PyTorch IterableDataset for memory-efficient patch-based image processing.
    Loads images on the fly, slices them into fixed-size patches,
     and streams them sequentially without loading the full dataset into RAM.
    """

    def __init__(self, file_paths: list, patch_size: tuple[int, int] = (96, 96), min_overlap: float = 0.0,
                 shuffle: bool = True, seed: int = -1, is_mixed_color_type: bool = False,
                 g_augmentor: GeometricAugmentor | None = None, n_augmentor: NoiseInjector | None = None):
        """
        Initialize the streaming patch dataset.
        :param file_paths: list of image file paths (Path objects or strings)
        :param patch_size: target patch size as (height, width) tuple
        :param min_overlap: minimal overlap (ratio < 1.0 or absolute pixels >= 1.0)
        :param shuffle: if True, shuffles file order per epoch and patches within each image
        :param seed: pseudo-random generator seed for reproducible shuffling (-1 disables seed)
        :param is_mixed_color_type: if True, converts 2D grayscale images into 3-channel RGB
        :param g_augmentor: optional geometric augmentor callable applied to patches
        :param n_augmentor: optional noise injector callable applied to patches
        """
        self.initial_file_paths = file_paths.copy()
        self.file_paths = file_paths.copy()
        self.patch_size = patch_size
        self.min_overlap = min_overlap
        self.shuffle = shuffle
        self.is_mixed_color_type = is_mixed_color_type
        self.geom_augmentor = g_augmentor
        self.noise_augmentor = n_augmentor
        # set seed of the random
        self.seed = seed
        self.rng = np.random.default_rng(seed if seed > -1 else None)

    def reset_seed(self, seed: int | None = None) -> None:
        """
        Reset random generator and restore original file order for reproducible multi-model training.
        :param seed: optional new pseudo-random seed; if None, resets using initial seed
        """
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed if self.seed > -1 else None)
        # reset list of paths
        self.file_paths = self.initial_file_paths.copy()

    def __iter__(self):
        """
        Iterate over image files, slice them into patches, and yield individual patches.
        :return: generator yielding individual cropped patches as numpy ndarrays
        """
        # shuffle paths
        if self.shuffle:
            self.rng.shuffle(self.file_paths)

        # processing each path
        for file_path in self.file_paths:
            # load image
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"File read error: {file_path}")
                continue
            else:
                # get image type color or grayscale
                is_color = True if img.ndim == 3 else False
                # apply slicing
                patches = image_slicer(np.asarray(img), self.patch_size, min_overlap=self.min_overlap)
                # shuffle patches
                if self.shuffle:
                    self.rng.shuffle(patches)
                # provide patches
                for patch in patches:
                    # apply geometric augmentation
                    if self.geom_augmentor is not None:
                        clean_patch = self.geom_augmentor(patch, rng=self.rng)
                    else:
                        clean_patch = patch

                    # for other augmentations
                    augmented_patch = clean_patch.copy()

                    # apply noice augmentation
                    if self.noise_augmentor is not None:
                        augmented_patch = self.noise_augmentor(augmented_patch, rng=self.rng)

                    # check if we have a mix of images
                    if self.is_mixed_color_type and not is_color:
                        # convert into RGB
                        augmented_patch = cv2.cvtColor(augmented_patch, cv2.COLOR_GRAY2RGB)
                        clean_patch = cv2.cvtColor(clean_patch, cv2.COLOR_GRAY2RGB)
                    elif is_color:
                        # convert into RGB
                        augmented_patch = cv2.cvtColor(augmented_patch, cv2.COLOR_BGR2RGB)
                        clean_patch = cv2.cvtColor(clean_patch, cv2.COLOR_BGR2RGB)

                    yield augmented_patch, clean_patch
