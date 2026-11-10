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
