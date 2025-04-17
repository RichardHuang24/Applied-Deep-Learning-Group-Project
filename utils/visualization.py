"""
Visualization tools for weakly-supervised segmentation using PIL
"""
import os
import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
logger = logging.getLogger(__name__)

NUM_CLASSES = 2              # 0‑bg, 1‑fg 
ALPHA       = 128            # 0‑255 overlap
_FONT       = None           # Font for text overlay

def _get_font(size: int = 16):
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.truetype("arial.ttf", size)
        except IOError:
            _FONT = ImageFont.load_default()
    return _FONT


def _build_palette(n: int) -> List[tuple]:
    """Return n RGB tuples (idx‑0 background is black)."""
    palette = [(0, 0, 0)]
    for i in range(1, n):
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


def tensor_to_pil(t):
    """(C,H,W) / (H,W) Tensor → PIL. """
    if not isinstance(t, torch.Tensor):
        return t
    if t.dim() == 4:                       # batch[0]
        t = t[0]
    if t.dim() == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    arr = t.cpu().numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 1) * 255
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr)


def mask_to_np(mask: torch.Tensor | np.ndarray):
    """(H,W) int / (C,H,W) one‑hot, return (H,W) int numpy."""
    if isinstance(mask, torch.Tensor):
        if mask.dim() == 3 and mask.shape[0] > 1:
            mask = mask.argmax(0)
        mask = mask.squeeze().cpu().numpy()
    return mask.astype(np.int32)



def color_mask(mask_np: np.ndarray) -> Image.Image:
    """class‑index mask → RGB Image """
    h, w = mask_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(1, NUM_CLASSES):
        color[mask_np == cls] = PALETTE[cls]
    return Image.fromarray(color)


def overlay_mask(img: Image.Image, mask_np: np.ndarray, alpha=ALPHA) -> Image.Image:
    """mask overlap."""
    if img.mode != "RGBA":
        img_rgba = img.convert("RGBA")
    else:
        img_rgba = img.copy()
    color = color_mask(mask_np).convert("RGBA")
    color.putalpha(alpha)
    return Image.alpha_composite(img_rgba, color).convert("RGB")

def visualize_prediction(image, pred_mask, gt_mask=None,
                         save_path: str | Path | None = None,
                         title: str | None = None) -> Image.Image:
    """Return a 4‑column PIL (若无 GT → 3‑column)."""
    img_pil = tensor_to_pil(image)
    pred_np = mask_to_np(pred_mask)

    col_imgs = [img_pil]                   # Original
    labels   = ["Original"]

    if gt_mask is not None:
        gt_np   = mask_to_np(gt_mask)
        col_imgs.append(color_mask(gt_np))
        labels.append("GT")

    col_imgs.append(color_mask(pred_np))
    labels.append("Mask")
    col_imgs.append(overlay_mask(img_pil, pred_np))
    labels.append("Overlap")

    w, h = img_pil.size
    vis  = Image.new("RGB", (w * len(col_imgs), h))
    for i, im in enumerate(col_imgs):
        vis.paste(im, (i * w, 0))

    draw = ImageDraw.Draw(vis)
    font = _get_font(16)
    for i, lbl in enumerate(labels):
        draw.text((i * w + 10, 10), lbl, fill=(255, 255, 255), font=font)
    if title:
        draw.text((w, h - 20), title, fill=(255, 255, 255), font=font)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        vis.save(save_path)
    return vis


def visualize_results_grid(images: Sequence,
                           preds: Sequence,
                           gt_masks: Sequence | None = None,
                           save_path: str | Path | None = None,
                           show_gt: bool = True) -> Image.Image:
    """Batch。4‑column or 3‑column (no GT)。"""
    n      = len(images)
    cols   = min(4, n)
    rows   = (n + cols - 1) // cols
    subcol = 4 if (show_gt and gt_masks is not None) else 3

    w, h   = tensor_to_pil(images[0]).size
    grid   = Image.new("RGB", (cols * subcol * w, rows * h))

    for idx, (img, pmask) in enumerate(zip(images, preds)):
        r, c   = divmod(idx, cols)
        x_off  = c * subcol * w
        y_off  = r * h

        vis = visualize_prediction(
            img,
            pmask,
            gt_masks[idx] if show_gt and gt_masks is not None else None,
        )
        grid.paste(vis, (x_off, y_off))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        grid.save(save_path)
    return grid


def visualize_cam(image, cam, mask=None, save_path=None):
    img_pil = tensor_to_pil(image)
    w, h    = img_pil.size

    cam_t = cam if isinstance(cam, torch.Tensor) else torch.tensor(cam)
    if cam_t.shape != torch.Size([h, w]):
        cam_t = F.interpolate(cam_t.unsqueeze(0).unsqueeze(0),
                              size=(h, w), mode="bilinear",
                              align_corners=False).squeeze()
    cam_t  = (cam_t - cam_t.min()) / (cam_t.max() - cam_t.min() + 1e-8)
    heat   = (cam_t * 255).byte().cpu().numpy()
    heat   = Image.fromarray(heat, mode="L").convert("RGB")
    heat   = Image.blend(img_pil, heat, 0.5)

    if mask is not None:
        mask_np   = mask_to_np(mask)
        mask_over = overlay_mask(img_pil, mask_np, ALPHA // 2)
        canvas    = Image.new("RGB", (w * 3, h))
        canvas.paste(img_pil,        (0,     0))
        canvas.paste(heat,           (w,     0))
        canvas.paste(mask_over,      (w * 2, 0))
        labels = ["Original", "CAM", "Mask"]
    else:
        canvas = Image.new("RGB", (w * 2, h))
        canvas.paste(img_pil, (0, 0))
        canvas.paste(heat,    (w, 0))
        labels = ["Original", "CAM"]

    draw = ImageDraw.Draw(canvas)
    font = _get_font()
    for i, lbl in enumerate(labels):
        draw.text((i * w + 10, 10), lbl, fill=(255, 255, 255), font=font)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(save_path)
    return canvas