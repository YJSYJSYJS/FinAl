"""Image loading and augmentation for frontal chest radiographs."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


MIMIC_MEAN = [0.4978]
MIMIC_STD = [0.2449]


def resize_and_pad(image, size):
    """Reproduce the OpenCV resize and padding used to train FinAl."""
    if image.ndim != 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image has an invalid size")
    if size <= 0:
        raise ValueError("size must be positive")

    height, width = image.shape
    if height >= width:
        resized_height = size
        resized_width = int(width * size / float(height))
    else:
        resized_height = int(height * size / float(width))
        resized_width = size
    if resized_height < 1 or resized_width < 1:
        raise ValueError("image aspect ratio is too extreme for the requested size")

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    vertical_padding = size - resized_height
    horizontal_padding = size - resized_width
    top = int(np.floor(vertical_padding / 2))
    bottom = int(np.ceil(vertical_padding / 2))
    left = int(np.floor(horizontal_padding / 2))
    right = int(np.ceil(horizontal_padding / 2))
    padded = np.pad(
        resized,
        ((top, bottom), (left, right)),
        mode="constant",
        constant_values=0,
    )
    return Image.fromarray(padded).convert("RGB")


def load_image(path, image_size=256, transform=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("image not found: {}".format(path))
    image_array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image_array is None:
        raise ValueError("failed to decode image: {}".format(path))
    image = resize_and_pad(image_array, image_size)
    return transform(image) if transform is not None else image


class DataTransforms:
    def __init__(self, is_train=True, crop_size=224):
        crop = transforms.RandomCrop(crop_size) if is_train else transforms.CenterCrop(crop_size)
        self.transform = transforms.Compose(
            [crop, transforms.ToTensor(), transforms.Normalize(mean=MIMIC_MEAN, std=MIMIC_STD)]
        )

    def __call__(self, image):
        return self.transform(image)
