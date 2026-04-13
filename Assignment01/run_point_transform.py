import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None


# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img


# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image

    if image is None:
        return None

    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 3, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 3, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(
            marked_image,
            tuple(points_src[i]),
            tuple(points_dst[i]),
            (0, 255, 0),
            1
        )

    return marked_image


# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    RBF-based image warping.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    source_pts : np.ndarray of shape (N, 2)
        Source control points.
    target_pts : np.ndarray of shape (N, 2)
        Target control points.
    alpha : float
        RBF strength parameter.
    eps : float
        Small constant for numerical stability.

    Return
    ------
    warped_image : np.ndarray
        Deformed image.
    """

    if image is None:
        return None

    warped_image = np.array(image).copy()

    # Need at least one valid pair
    if len(source_pts) == 0 or len(target_pts) == 0:
        return warped_image

    # Only use paired points
    n = min(len(source_pts), len(target_pts))
    source_pts = np.array(source_pts[:n], dtype=np.float32)
    target_pts = np.array(target_pts[:n], dtype=np.float32)

    if n == 1:
        # Single-point translation fallback
        displacement = target_pts[0] - source_pts[0]
        h, w = image.shape[:2]

        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x - displacement[0]).astype(np.float32)
        map_y = (grid_y - displacement[1]).astype(np.float32)

        warped_image = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101
        )
        return warped_image

    h, w = image.shape[:2]

    # Displacement at control points
    disp = target_pts - source_pts   # shape: (n, 2)

    # ------------------------------------------------------------------
    # Step 1: Solve RBF weights
    # Use phi(r) = exp(-alpha * r^2)
    # K w = d
    # ------------------------------------------------------------------
    diff = source_pts[:, None, :] - source_pts[None, :, :]   # (n, n, 2)
    dist2 = np.sum(diff ** 2, axis=2)                        # (n, n)
    K = np.exp(-alpha * dist2) + eps * np.eye(n, dtype=np.float32)

    wx = np.linalg.solve(K, disp[:, 0])   # weights for x displacement
    wy = np.linalg.solve(K, disp[:, 1])   # weights for y displacement

    # ------------------------------------------------------------------
    # Step 2: Compute dense displacement field on all pixels
    # ------------------------------------------------------------------
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    pixel_coords = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)   # (h, w, 2)

    # Distance from every pixel to every source control point
    # result shape: (h, w, n)
    diff_pixels = pixel_coords[:, :, None, :] - source_pts[None, None, :, :]
    dist2_pixels = np.sum(diff_pixels ** 2, axis=3)

    Phi = np.exp(-alpha * dist2_pixels)   # (h, w, n)

    field_x = np.sum(Phi * wx[None, None, :], axis=2)
    field_y = np.sum(Phi * wy[None, None, :], axis=2)

    # ------------------------------------------------------------------
    # Step 3: Backward warping
    # For each target pixel p, sample from source pixel p - field(p)
    # ------------------------------------------------------------------
    map_x = (grid_x - field_x).astype(np.float32)
    map_y = (grid_y - field_y).astype(np.float32)

    warped_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

    return warped_image


def run_warping():
    global points_src, points_dst, image

    if image is None:
        return None

    warped_image = point_guided_deformation(
        image,
        np.array(points_src),
        np.array(points_dst),
        alpha=1e-4
    )

    return warped_image


# Clear all selected points
def clear_points():
    global points_src, points_dst, image
    points_src.clear()
    points_dst.clear()
    return image


# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch(inbrowser=True)
