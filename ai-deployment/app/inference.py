import io
import time
import cv2
import numpy as np
import torch
from PIL import Image

IMG_SIZE = 512
THRESHOLD = 0.5
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocess(image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_rgb = np.array(pil_img)

    resized = cv2.resize(orig_rgb, (IMG_SIZE, IMG_SIZE))
    img_float = resized.astype(np.float32) / 255.0

    mean = np.array(MEAN, dtype=np.float32)
    std = np.array(STD, dtype=np.float32)
    normalized = (img_float - mean) / std

    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, orig_rgb


def postprocess(prob_mask: np.ndarray, orig_rgb: np.ndarray) -> dict:
    binary_mask = (prob_mask > THRESHOLD).astype(np.float32)

    h_orig, w_orig = orig_rgb.shape[:2]
    binary_resized = cv2.resize(
        binary_mask.astype(np.uint8),
        (w_orig, h_orig),
        interpolation=cv2.INTER_NEAREST
    )

    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= 100]

    bbox = None
    if valid_contours:
        largest = max(valid_contours, key=cv2.contourArea)
        b_x, b_y, b_w, b_h = cv2.boundingRect(largest.astype(np.int32))
        h_ratio = h_orig / IMG_SIZE
        w_ratio = w_orig / IMG_SIZE
        bbox = {
            "x": int(b_x * w_ratio),
            "y": int(b_y * h_ratio),
            "width": int(b_w * w_ratio),
            "height": int(b_h * h_ratio),
        }

    n_pixels = np.sum(binary_mask)
    total_pixels = IMG_SIZE * IMG_SIZE
    area_percent = float(n_pixels / total_pixels * 100)

    if n_pixels > 0:
        confidence = float(np.sum(prob_mask * binary_mask) / n_pixels)
    else:
        confidence = 0.0

    ss_score = area_percent * confidence

    if ss_score >= 50:
        level, level_name = 3, "DARURAT"
    elif ss_score >= 20:
        level, level_name = 2, "SIAGA"
    else:
        level, level_name = 1, "AMAN"

    overlay = orig_rgb.copy().astype(np.float32)
    flood_region = binary_resized == 1
    overlay[flood_region] = overlay[flood_region] * 0.4 + np.array([0, 120, 255]) * 0.6
    overlay_img = overlay.astype(np.uint8)

    _, buf = cv2.imencode(".png", cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
    overlay_bytes = buf.tobytes()

    return {
        "area_percent": round(area_percent, 2),
        "confidence": round(confidence, 4),
        "ss_score": round(ss_score, 2),
        "severity_level": level,
        "severity_name": level_name,
        "num_flood_regions": len(valid_contours),
        "bbox": bbox,
        "overlay_bytes": overlay_bytes,
    }


def run_inference(model: torch.nn.Module, image_bytes: bytes, device: torch.device) -> dict:
    tensor, orig_rgb = preprocess(image_bytes)
    tensor = tensor.to(device)

    start = time.time()
    model.eval()
    with torch.no_grad():
        out = model(tensor)
        prob_mask = torch.sigmoid(out).squeeze().cpu().numpy()
    elapsed_ms = int((time.time() - start) * 1000)

    result = postprocess(prob_mask, orig_rgb)
    result["inference_ms"] = elapsed_ms
    return result
