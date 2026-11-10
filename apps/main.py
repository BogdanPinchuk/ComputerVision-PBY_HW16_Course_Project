import cv2
import keras
import numpy as np
from tqdm.auto import tqdm
import apps.reporter as rpt
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.core.display import Markdown
from pandas import Series, DataFrame
from pandas.io.formats.style import Styler


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


def train_test_split_by_order(array, test_size: float) -> tuple:
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


def image_slicer(image, patch_size: tuple[int, int], min_overlap: float = 0.0) -> list[list[np.ndarray]]:
    """
    Crop input image into 2D grid of fixed-size patches [row][col].
    :param image: input image as numpy ndarray (H, W) or (H, W, C)
    :param patch_size: target patch size as (height, width)
    :param min_overlap: minimal overlap (ratio < 1.0 or absolute pixels >= 1.0)
    :return: 2D list of cropped patches
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

    return patches
