import argparse
import numpy as np
import tensorflow as tf
import matplotlib
from tensorflow.keras.utils import load_img, img_to_array, array_to_img

from preprocess import load_and_preprocess


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except AttributeError:
            continue
        if len(shape) == 4:
            return layer.name
    raise ValueError("Could not find a 4D conv layer in the model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        tape.watch(conv_output)
        if isinstance(predictions, list):
            predictions = predictions[0]
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(image_path, heatmap, output_path, alpha=0.4) -> None:
    img = img_to_array(load_img(image_path))
    heatmap_resized = np.uint8(255 * heatmap)
    jet_colors = matplotlib.colormaps["jet"](np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = array_to_img(jet_heatmap).resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    overlaid = jet_heatmap * alpha + img
    array_to_img(overlaid).save(output_path)
    print(f"Saved Grad-CAM overlay to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/evaluation/gradcam_overlay.jpg")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    last_conv_layer_name = find_last_conv_layer(model)

    img_array = load_and_preprocess(args.image, target_size=(224, 224))

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    overlay_heatmap(args.image, heatmap, args.output)


if __name__ == "__main__":
    main()