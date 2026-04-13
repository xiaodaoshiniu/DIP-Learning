import os
import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import torch

WORKING_IMAGE_HEIGHT = 300

# Avoid routing localhost Gradio checks through a system proxy.
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'
for _proxy_key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    proxy_value = os.environ.get(_proxy_key, '')
    if proxy_value.startswith('http://127.0.0.1') or proxy_value.startswith('http://localhost'):
        os.environ[_proxy_key] = ''


def initialize_polygon():
    return {'points': [], 'closed': False}


def prepare_working_image(image, target_height=WORKING_IMAGE_HEIGHT):
    if image is None:
        return None

    image = image.convert('RGB')
    target_width = max(1, int(round(image.width * target_height / image.height)))
    if image.size == (target_width, target_height):
        return image.copy()
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def load_foreground_image(image):
    working_image = prepare_working_image(image)
    return working_image, working_image, initialize_polygon(), None


def load_background_image(image):
    working_image = prepare_working_image(image)
    return working_image, working_image


def add_point(img_original, polygon_state, evt=None):
    if img_original is None:
        return None, polygon_state

    if evt is None:
        return img_original, polygon_state

    if polygon_state['closed']:
        return img_original, polygon_state

    x, y = evt.index
    polygon_state['points'].append((int(x), int(y)))

    img_with_poly = img_original.copy()
    draw = ImageDraw.Draw(img_with_poly)

    if len(polygon_state['points']) > 1:
        draw.line(polygon_state['points'], fill='red', width=2)

    for point in polygon_state['points']:
        draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill='blue')

    return img_with_poly, polygon_state


def close_polygon(img_original, polygon_state):
    if img_original is None:
        return None, polygon_state

    if not polygon_state['closed'] and len(polygon_state['points']) > 2:
        polygon_state['closed'] = True
        img_with_poly = img_original.copy()
        draw = ImageDraw.Draw(img_with_poly)
        draw.polygon(polygon_state['points'], outline='red')
        return img_with_poly, polygon_state
    return img_original, polygon_state


def update_background(background_image_original, polygon_state, dx, dy):
    if background_image_original is None:
        return None

    if polygon_state['closed']:
        img_with_poly = background_image_original.copy()
        draw = ImageDraw.Draw(img_with_poly)
        shifted_points = [(x + int(dx), y + int(dy)) for x, y in polygon_state['points']]
        draw.polygon(shifted_points, outline='red')
        return img_with_poly
    return background_image_original


def create_mask_from_points(points, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if len(points) < 3:
        return mask

    mask_image = Image.new('L', (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon([tuple(map(int, point)) for point in points.tolist()], outline=255, fill=255)
    return np.array(mask_image, dtype=np.uint8)


def shift_tensor_to_canvas(tensor, dx, dy, out_h, out_w):
    batch, channels, in_h, in_w = tensor.shape
    shifted = torch.zeros((batch, channels, out_h, out_w), device=tensor.device, dtype=tensor.dtype)

    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(in_w, out_w - dx)
    src_y1 = min(in_h, out_h - dy)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return shifted

    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    shifted[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = tensor[:, :, src_y0:src_y1, src_x0:src_x1]
    return shifted


def cal_poisson_loss(shifted_source_img, blended_img, background_mask):
    channels = shifted_source_img.shape[1]

    source_grad_x = shifted_source_img[:, :, :, 1:] - shifted_source_img[:, :, :, :-1]
    source_grad_y = shifted_source_img[:, :, 1:, :] - shifted_source_img[:, :, :-1, :]
    blended_grad_x = blended_img[:, :, :, 1:] - blended_img[:, :, :, :-1]
    blended_grad_y = blended_img[:, :, 1:, :] - blended_img[:, :, :-1, :]

    mask_x = torch.maximum(background_mask[:, :, :, 1:], background_mask[:, :, :, :-1])
    mask_y = torch.maximum(background_mask[:, :, 1:, :], background_mask[:, :, :-1, :])

    loss_x = (((blended_grad_x - source_grad_x) * mask_x) ** 2).sum() / (mask_x.sum() * channels).clamp_min(1.0)
    loss_y = (((blended_grad_y - source_grad_y) * mask_y) ** 2).sum() / (mask_y.sum() * channels).clamp_min(1.0)
    return loss_x + loss_y


def blending(foreground_image_original, background_image_original, dx, dy, polygon_state):
    if not polygon_state['closed'] or background_image_original is None or foreground_image_original is None:
        return background_image_original

    foreground_np = np.array(foreground_image_original.convert('RGB'))
    background_np = np.array(background_image_original.convert('RGB'))

    foreground_polygon_points = np.array(polygon_state['points']).astype(np.int64)
    foreground_mask = create_mask_from_points(foreground_polygon_points, foreground_np.shape[0], foreground_np.shape[1])
    if foreground_mask.sum() == 0:
        return background_image_original

    dx = int(dx)
    dy = int(dy)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    fg_img_tensor = torch.from_numpy(foreground_np).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    bg_img_tensor = torch.from_numpy(background_np).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    fg_mask_tensor = torch.from_numpy(foreground_mask).to(device).unsqueeze(0).unsqueeze(0).float() / 255.0

    shifted_fg_tensor = shift_tensor_to_canvas(fg_img_tensor, dx, dy, background_np.shape[0], background_np.shape[1])
    bg_mask_tensor = shift_tensor_to_canvas(fg_mask_tensor, dx, dy, background_np.shape[0], background_np.shape[1])
    if bg_mask_tensor.sum().item() == 0:
        return background_image_original

    blended_var = bg_img_tensor.clone().detach()
    blended_var.requires_grad_(True)

    optimizer = torch.optim.Adam([blended_var], lr=5e-2)
    iter_num = 1500
    for step in range(iter_num):
        blended_img = bg_img_tensor * (1.0 - bg_mask_tensor) + blended_var * bg_mask_tensor
        loss = cal_poisson_loss(shifted_fg_tensor, blended_img, bg_mask_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            blended_var.clamp_(0.0, 1.0)

        if step % 100 == 0:
            print(f'Optimize step: {step}, poisson loss: {loss.item():.6f}')

        if step == int(iter_num * 2 / 3):
            optimizer.param_groups[0]['lr'] *= 0.2

    result = bg_img_tensor * (1.0 - bg_mask_tensor) + blended_var.detach() * bg_mask_tensor
    result = result.cpu().permute(0, 2, 3, 1).squeeze().numpy() * 255
    return np.clip(result, 0, 255).astype(np.uint8)


def close_polygon_and_reset_dx(img_original, polygon_state, dx, dy, background_image_original):
    img_with_poly, updated_polygon_state = close_polygon(img_original, polygon_state)
    new_dx = gr.update(value=0)
    updated_background = update_background(background_image_original, updated_polygon_state, 0, dy)
    return img_with_poly, updated_polygon_state, updated_background, new_dx


with gr.Blocks(title='Poisson Image Blending', css="""
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .gr-button {
        font-size: 1em;
        padding: 0.75em 1.5em;
        border-radius: 8px;
        background-color: #6200ee;
        color: #ffffff;
        border: none;
    }
    .gr-button:hover {
        background-color: #3700b3;
    }
    .gr-slider input[type=range] {
        accent-color: #03dac6;
    }
    .gr-text, .gr-markdown {
        font-size: 1.1em;
    }
    .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
        color: #bb86fc;
    }
    .gr-input, .gr-output {
        background-color: #2c2c2c;
        border: 1px solid #3c3c3c;
    }
""") as demo:
    polygon_state = gr.State(initialize_polygon())

    gr.Markdown("<h1 style='text-align: center;'>Poisson Image Blending</h1>")
    gr.Markdown("<p style='text-align: center; font-size: 1.2em;'>Blend a selected area from a foreground image onto a background image with adjustable positions.</p>")
    gr.Markdown("<p style='text-align: center; font-size: 0.95em;'>Uploaded images are resized to a 300px working height so the points you click match the pixels used for blending.</p>")

    with gr.Row():
        with gr.Column():
            gr.Markdown('### Foreground Image')
            foreground_image_original = gr.Image(label='', type='pil', interactive=True, height=300)
            gr.Markdown("<p style='font-size: 0.9em;'>Upload the foreground image where the polygon will be selected.</p>")
            gr.Markdown('### Foreground Image with Polygon')
            foreground_image_with_polygon = gr.Image(label='', type='pil', interactive=True, height=300)
            gr.Markdown("<p style='font-size: 0.9em;'>Click on the image to define the polygon area. After selecting at least three points, click <strong>Close Polygon</strong>.</p>")
            close_polygon_button = gr.Button('Close Polygon')
        with gr.Column():
            gr.Markdown('### Background Image')
            background_image = gr.Image(label='', type='pil', interactive=True, height=300)
            gr.Markdown("<p style='font-size: 0.9em;'>Upload the background image where the polygon will be placed.</p>")

    with gr.Row():
        with gr.Column():
            gr.Markdown('### Background Image with Polygon Overlay')
            background_image_with_polygon = gr.Image(label='', type='pil', height=500)
            gr.Markdown("<p style='font-size: 0.9em;'>Adjust the position of the polygon using the sliders below.</p>")
        with gr.Column():
            gr.Markdown('### Blended Image')
            output_image = gr.Image(label='', type='pil', height=500)

    with gr.Row():
        with gr.Column():
            dx = gr.Slider(label='Horizontal Offset', minimum=-500, maximum=500, step=1, value=0)
        with gr.Column():
            dy = gr.Slider(label='Vertical Offset', minimum=-500, maximum=500, step=1, value=0)
        blend_button = gr.Button('Blend Images')

    foreground_image_original.change(
        fn=load_foreground_image,
        inputs=foreground_image_original,
        outputs=[foreground_image_original, foreground_image_with_polygon, polygon_state, background_image_with_polygon],
    )
    foreground_image_with_polygon.select(
        add_point,
        inputs=[foreground_image_original, polygon_state],
        outputs=[foreground_image_with_polygon, polygon_state],
    )
    close_polygon_button.click(
        fn=close_polygon_and_reset_dx,
        inputs=[foreground_image_original, polygon_state, dx, dy, background_image],
        outputs=[foreground_image_with_polygon, polygon_state, background_image_with_polygon, dx],
    )
    background_image.change(
        fn=load_background_image,
        inputs=background_image,
        outputs=[background_image, background_image_with_polygon],
    )
    dx.change(
        fn=update_background,
        inputs=[background_image, polygon_state, dx, dy],
        outputs=background_image_with_polygon,
    )
    dy.change(
        fn=update_background,
        inputs=[background_image, polygon_state, dx, dy],
        outputs=background_image_with_polygon,
    )
    blend_button.click(
        fn=blending,
        inputs=[foreground_image_original, background_image, dx, dy, polygon_state],
        outputs=output_image,
    )


if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=7860, inbrowser=False, share=False)
