import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.layers import Rescaling

# MobileNetV2 expects inputs scaled to [-1, 1]
_rescale_layer = Rescaling(scale=1.0 / 127.5, offset=-1.0)


def load_and_preprocess(image_path, target_size=(224, 224)) -> np.ndarray:
    """Load an image file and return a batch-ready, rescaled array (shape (1, H, W, 3))."""
    img = load_img(image_path, target_size=target_size)
    array = img_to_array(img)
    array = _rescale_layer(array).numpy()
    return np.expand_dims(array, axis=0)