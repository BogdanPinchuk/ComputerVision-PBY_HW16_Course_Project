import cv2
import keras
import random
import numpy as np
from tqdm.auto import tqdm
import apps.reporter as rpt
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.core.display import Markdown
from pandas import Series, DataFrame
from pandas.io.formats.style import Styler
from torch.utils.data import IterableDataset


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


def train_test_split_by_order(array, test_size: float = 0.2) -> tuple:
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


def calculate_integer_layout(total_size: int, tile_size: int, min_overlap: float = 0.0, include: bool = True) \
        -> tuple[np.ndarray, np.ndarray]:
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


def image_slicer(image: np.ndarray, patch_size: tuple[int, int], min_overlap: float = 0.0, is_flat: bool = True) -> \
        list[list[np.ndarray]] | list[np.ndarray]:
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


class StreamPatchDataset(IterableDataset):
    """
    Streaming PyTorch IterableDataset for memory-efficient patch-based image processing.
    Loads images on the fly, slices them into fixed-size patches,
     and streams them sequentially without loading the full dataset into RAM.
    """

    def __init__(self, file_paths: list, patch_size: tuple[int, int] = (96, 96), min_overlap: float = 0.0,
                 shuffle: bool = True, seed: int = -1, is_mixed_color_type: bool = False,
                 augmentor=None):
        """
        Initialize the streaming patch dataset.
        :param file_paths: list of image file paths (Path objects or strings)
        :param patch_size: target patch size as (height, width) tuple
        :param min_overlap: minimal overlap (ratio < 1.0 or absolute pixels >= 1.0)
        :param shuffle: if True, shuffles file order per epoch and patches within each image
        :param seed: pseudo-random generator seed for reproducible shuffling (-1 disables seed)
        :param is_mixed_color_type: if True, converts 2D grayscale images into 3-channel RGB
        :param augmentor: optional geometric augmentor callable applied to patches
        """
        self.initial_file_paths = file_paths.copy()
        self.file_paths = file_paths.copy()
        self.patch_size = patch_size
        self.min_overlap = min_overlap
        self.shuffle = shuffle
        self.is_mixed_color_type = is_mixed_color_type
        self.geo_augmentor = augmentor
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
                    # check if we have a mix of images
                    if self.is_mixed_color_type and not is_color:
                        # convert into RGB
                        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
                    elif is_color:
                        # convert into RGB
                        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

                    # apply geometric augmentation
                    if self.geo_augmentor is not None:
                        clean_patch = self.geo_augmentor(patch, rng=self.rng)
                    else:
                        clean_patch = patch

                    # for other augmentations
                    augmented_patch = clean_patch.copy()

                    yield augmented_patch, clean_patch
