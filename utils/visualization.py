"""
Simple visualisation helpers for weakly‑supervised segmentation.
The file avoids modern Python syntax that might not be available on
old interpreters (≤ 3.8):
  * No type‑hints that use the ``|`` union operator
  * No ``pathlib.Path`` in the public API – only strings are accepted
  * Comments and doc‑strings are written in English only

The public API is intentionally minimal:
    tensor_to_pil(img_tensor)          -> PIL.Image
    visualize_prediction(image, pred_mask, gt_mask=None, save_path=None)
    visualize_results_grid(images, preds, gt_masks=None, save_path=None)
    visualize_cam(image, cam, mask=None, save_path=None)

All helpers work with **foreground / background** setup
(``NUM_CLASSES = 2``).
"""

import os
import logging
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
NUM_CLASSES = 2      # background = 0, foreground = 1
ALPHA       = 128    # transparency for mask overlay (0‑255)

# Lazy font cache -------------------------------------------------------------
_FONT = None

def _get_font(size=16):
    """Return a TTF font; fallback to default if ``arial.ttf`` is missing."""
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.truetype("arial.ttf", size)
        except IOError:
            _FONT = ImageFont.load_default()
    return _FONT

# Colour palette --------------------------------------------------------------

def _build_palette(n):
    """Return *n* distinct RGB tuples. Index 0 is black (background)."""
    palette = [(0, 0, 0)]  # background colour
    for i in range(1, n):
        # simple HSV→RGB conversion for distinct colours
        hue = (i * 37) % 360
        h   = hue / 60.0
        s, v = 0.8, 0.9
        c   = v * s
        x   = c * (1 - abs(h % 2 - 1))
        m   = v - c
        if   0 <= h < 1: r, g, b = c, x, 0
        elif 1 <= h < 2: r, g, b = x, c, 0
        elif 2 <= h < 3: r, g, b = 0, c, x
        elif 3 <= h < 4: r, g, b = 0, x, c
        elif 4 <= h < 5: r, g, b = x, 0, c
        else:            r, g, b = c, 0, x
        palette.append(tuple(int(255 * (val + m)) for val in (r, g, b)))
    return palette

PALETTE = _build_palette(NUM_CLASSES)

# -----------------------------------------------------------------------------
# Basic tensor/array helpers
# -----------------------------------------------------------------------------

def tensor_to_pil(tensor):
    """Convert a torch Tensor to a ``PIL.Image``.

    Accepts
        * ``(C, H, W)`` floating tensor in **[0, 1]** – C == 1 or 3
        * ``(H, W)`` mask tensor 0/1
        * A batch tensor ``(N, C, H, W)`` – only the first sample is used
    """
    if not isinstance(tensor, torch.Tensor):
        # Already a PIL image – return as‑is
        return tensor

    if tensor.dim() == 4:
        tensor = tensor[0]  # first element of batch

    if tensor.dim() == 3 and tensor.size(0) in (1, 3):
        tensor = tensor.permute(1, 2, 0)  # to (H, W, C)

    array = tensor.cpu().numpy()
    if array.dtype != np.uint8:
        array = (np.clip(array, 0, 1) * 255).astype(np.uint8)

    if array.ndim == 2:
        return Image.fromarray(array, mode="L")
    return Image.fromarray(array)


def mask_to_numpy(mask):
    """Convert a mask to ``numpy.ndarray`` with shape ``(H, W)`` and dtype ``int``."""
    if isinstance(mask, torch.Tensor):
        if mask.dim() == 3 and mask.size(0) > 1:  # one‑hot → argmax
            mask = mask.argmax(0)
        mask = mask.squeeze().cpu()
        mask = mask.numpy()
    return mask.astype(np.int32)

# -----------------------------------------------------------------------------
# Colour helpers
# -----------------------------------------------------------------------------

def colourise_mask(mask_np):
    """Class‑index mask → colour RGB ``PIL.Image``."""
    h, w = mask_np.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(1, NUM_CLASSES):
        colour[mask_np == cls] = PALETTE[cls]
    return Image.fromarray(colour)


def overlay_mask(image_pil, mask_np, alpha=ALPHA):
    """Overlay *mask_np* (H,W) onto *image_pil* (RGBA blending)."""
    if image_pil.mode != "RGBA":
        base = image_pil.convert("RGBA")
    else:
        base = image_pil.copy()

    colour = colourise_mask(mask_np).convert("RGBA")
    colour.putalpha(alpha)
    return Image.alpha_composite(base, colour).convert("RGB")

# -----------------------------------------------------------------------------
# High‑level visualisation utilities
# -----------------------------------------------------------------------------

def visualize_prediction(image, pred_mask, gt_mask=None,
                         save_path=None, title=None):
    """Return a side‑by‑side visualisation as ``PIL.Image``.

    Columns:  *Original* | *GT* (optional) | *Mask* | *Overlap*
    """
    img_pil = tensor_to_pil(image)
    pred_np = mask_to_numpy(pred_mask)

    columns = [img_pil]
    labels  = ["Original"]

    if gt_mask is not None:
        gt_np = mask_to_numpy(gt_mask)
        columns.append(colourise_mask(gt_np))
        labels.append("GT")

    columns.append(colourise_mask(pred_np))
    labels.append("Mask")
    columns.append(overlay_mask(img_pil, pred_np))
    labels.append("Overlap")

    w, h = img_pil.size
    canvas = Image.new("RGB", (w * len(columns), h))
    for i, col in enumerate(columns):
        canvas.paste(col, (i * w, 0))

    draw = ImageDraw.Draw(canvas)
    font = _get_font()
    for i, lbl in enumerate(labels):
        draw.text((i * w + 10, 10), lbl, fill=(255, 255, 255), font=font)
    if title:
        draw.text((w, h - 20), title, fill=(255, 255, 255), font=font)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        canvas.save(save_path)
    return canvas


def visualize_results_grid(images, preds, gt_masks=None,
                           save_path=None, show_gt=True):
    """Plot a batch of samples in a grid.

    * ``images`` – list/tuple of images or tensors
    * ``preds``  – list/tuple of predicted masks
    * ``gt_masks`` optional – list/tuple of ground‑truth masks
    * ``show_gt`` – set ``False`` to hide GT column even if provided
    """
    num_samples = len(images)
    cols        = min(4, num_samples)
    rows        = (num_samples + cols - 1) // cols
    per_sample  = 4 if (show_gt and gt_masks is not None) else 3

    w, h = tensor_to_pil(images[0]).size
    grid = Image.new("RGB", (cols * per_sample * w, rows * h))

    for idx, (img, pred) in enumerate(zip(images, preds)):
        row, col = divmod(idx, cols)
        x_off    = col * per_sample * w
        y_off    = row * h

        gt = gt_masks[idx] if (show_gt and gt_masks is not None) else None
        vis = visualize_prediction(img, pred, gt_mask=gt)
        grid.paste(vis, (x_off, y_off))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        grid.save(save_path)
    return grid


def visualize_cam(image, cam, mask=None, save_path=None):
    """Visualise a class‑activation map (CAM)."""
    img_pil = tensor_to_pil(image)
    w, h    = img_pil.size

    cam_tensor = cam if isinstance(cam, torch.Tensor) else torch.tensor(cam)
    if cam_tensor.shape != torch.Size([h, w]):
        cam_tensor = F.interpolate(cam_tensor.unsqueeze(0).unsqueeze(0),
                                   size=(h, w), mode="bilinear",
                                   align_corners=False).squeeze()
    cam_tensor = (cam_tensor - cam_tensor.min()) / (cam_tensor.max() - cam_tensor.min() + 1e-8)
    heat       = (cam_tensor * 255).byte().cpu().numpy()
    heat_img   = Image.fromarray(heat, mode="L").convert("RGB")
    heat_img   = Image.blend(img_pil, heat_img, 0.5)

    if mask is not None:
        mask_np   = mask_to_numpy(mask)
        mask_over = overlay_mask(img_pil, mask_np, ALPHA // 2)
        canvas    = Image.new("RGB", (w * 3, h))
        canvas.paste(img_pil,  (0,     0))
        canvas.paste(heat_img, (w,     0))
        canvas.paste(mask_over,(w * 2, 0))
        labels = ["Original", "CAM", "Mask"]
    else:
        canvas = Image.new("RGB", (w * 2, h))
        canvas.paste(img_pil,  (0, 0))
        canvas.paste(heat_img, (w, 0))
        labels = ["Original", "CAM"]

    draw = ImageDraw.Draw(canvas)
    font = _get_font()
    for i, lbl in enumerate(labels):
        draw.text((i * w + 10, 10), lbl, fill=(255, 255, 255), font=font)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        canvas.save(save_path)
    return canvas