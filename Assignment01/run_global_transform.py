import gradio as gr
import cv2
import numpy as np


# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])


def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):
    if image is None:
        return None

    # Convert PIL image to RGB first, then to NumPy array
    image = image.convert("RGB")
    image = np.array(image)

    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.ones(
        (pad_size * 2 + image.shape[0], pad_size * 2 + image.shape[1], 3),
        dtype=np.uint8
    ) * 255

    image_new[
        pad_size:pad_size + image.shape[0],
        pad_size:pad_size + image.shape[1]
    ] = image
    image = image_new.copy()

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Move center to origin
    T_to_center = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # Horizontal flip around image center
    if flip_horizontal:
        F = np.array([
            [-1, 0, 0],
            [0,  1, 0],
            [0,  0, 1]
        ], dtype=np.float32)
    else:
        F = np.eye(3, dtype=np.float32)

    # Scale matrix
    S = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # Rotation matrix
    theta = np.deg2rad(rotation)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # Move origin back to image center
    T_back = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # Translation matrix
    T_translate = np.array([
        [1, 0, translation_x],
        [0, 1, translation_y],
        [0, 0, 1]
    ], dtype=np.float32)

    # Composite transformation:
    # center shift -> flip -> scale -> rotate -> shift back -> translate
    M = T_translate @ T_back @ R @ S @ F @ T_to_center

    # Convert 3x3 to 2x3 for warpAffine
    affine_M = M[:2, :]

    transformed_image = cv2.warpAffine(
        image,
        affine_M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    return transformed_image


# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(
                    minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale"
                )
                rotation = gr.Slider(
                    minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)"
                )
                translation_x = gr.Slider(
                    minimum=-300, maximum=300, step=10, value=0, label="Translation X"
                )
                translation_y = gr.Slider(
                    minimum=-300, maximum=300, step=10, value=0, label="Translation Y"
                )
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")

            image_output = gr.Image(label="Transformed Image")

        inputs = [
            image_input, scale, rotation,
            translation_x, translation_y,
            flip_horizontal
        ]

        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo


# Launch the Gradio interface
if __name__ == "__main__":
    interactive_transform().launch(inbrowser=True)
