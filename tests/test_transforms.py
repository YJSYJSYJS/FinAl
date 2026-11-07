import unittest

import cv2
import numpy as np

from final.transforms import resize_and_pad


def original_final_resize(image, size):
    """Reference implementation from the FinAl training data pipeline."""
    height, width = image.shape
    if height >= width:
        desired_size = (size, int(width * size / float(height)))
    else:
        desired_size = (int(height * size / float(width)), size)
    resized = cv2.resize(
        image,
        desired_size[::-1],
        interpolation=cv2.INTER_AREA,
    )

    if height >= width:
        left = int(np.floor((size - resized.shape[1]) / 2))
        right = int(np.ceil((size - resized.shape[1]) / 2))
        padding = ((0, 0), (left, right))
    else:
        top = int(np.floor((size - resized.shape[0]) / 2))
        bottom = int(np.ceil((size - resized.shape[0]) / 2))
        padding = ((top, bottom), (0, 0))
    return np.pad(resized, padding, mode="constant", constant_values=0)


class TransformParityTest(unittest.TestCase):
    def test_resize_and_pad_matches_original_final_pipeline(self):
        for height, width in ((311, 517), (513, 333), (1024, 1024)):
            image = np.fromfunction(
                lambda y, x: (7 * y + 13 * x + (x * y) % 251) % 256,
                (height, width),
                dtype=int,
            ).astype(np.uint8)
            expected = original_final_resize(image, 256)
            actual = np.asarray(resize_and_pad(image, 256).convert("L"))
            np.testing.assert_array_equal(actual, expected)

    def test_resize_and_pad_rejects_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "invalid size"):
            resize_and_pad(np.empty((0, 10), dtype=np.uint8), 256)
        with self.assertRaisesRegex(ValueError, "positive"):
            resize_and_pad(np.zeros((10, 10), dtype=np.uint8), 0)


if __name__ == "__main__":
    unittest.main()
