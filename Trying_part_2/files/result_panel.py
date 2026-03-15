import cv2
import numpy as np
from detect_tesseract import enhance_for_ocr
def build_result_panel(
    frame,
    bbox,
    crop_binary,
    crop_original,
    text_bin,  conf_bin,
    text_gray, conf_gray,
    text_col,  conf_col,
    panel_w=400
):
    """
    Build a side panel showing:
    - Each crop (binary, enhanced gray, colour)
    - OCR result + confidence per version
    """
    PANEL_H = frame.shape[0]
    panel   = np.ones((PANEL_H, panel_w, 3), dtype=np.uint8) * 30  # dark bg

    y_cursor = 20
    LINE_H   = 30
    CROP_H   = 60

    def put(img, text, pos, color=(255,255,255), scale=0.55, thick=1):
        cv2.putText(img, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    def draw_crop(crop_img, label, result_text, conf, color):
        nonlocal y_cursor
        put(panel, label, (10, y_cursor), (180, 180, 180), 0.5)
        y_cursor += 18

        if crop_img is not None and crop_img.size > 0:
            # Resize crop to fixed height for display
            ch, cw = crop_img.shape[:2]
            scale  = CROP_H / max(ch, 1)
            disp_w = min(int(cw * scale), panel_w - 20)
            disp   = cv2.resize(crop_img, (disp_w, CROP_H),
                                interpolation=cv2.INTER_NEAREST)
            # Convert to BGR if grayscale
            if len(disp.shape) == 2:
                disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            panel[y_cursor:y_cursor + CROP_H, 10:10 + disp_w] = disp
            y_cursor += CROP_H + 5

        result_str = f"{result_text}  ({conf}%)" if result_text else "---"
        put(panel, result_str, (10, y_cursor), color, 0.6, 2)
        y_cursor += LINE_H + 10

        # Divider
        cv2.line(panel, (10, y_cursor), (panel_w - 10, y_cursor),
                 (60, 60, 60), 1)
        y_cursor += 15

    # ── Title ─────────────────────────────────────────────────────────────────
    put(panel, "CENTER WORD OCR", (10, y_cursor), (0, 255, 255), 0.65, 2)
    y_cursor += LINE_H

    if bbox:
        put(panel, f"bbox: {bbox}", (10, y_cursor), (150, 150, 150), 0.45)
        y_cursor += LINE_H

    cv2.line(panel, (10, y_cursor), (panel_w - 10, y_cursor), (80, 80, 80), 1)
    y_cursor += 15

    # ── Version 1 — Binary ────────────────────────────────────────────────────
    draw_crop(crop_binary, "VERSION 1 — Binary",
              text_bin, conf_bin,
              (0, 255, 0) if conf_bin >= 60 else (0, 0, 255))

    # ── Version 2 — Enhanced Gray ─────────────────────────────────────────────
    gray_crop = enhance_for_ocr(crop_original) if crop_original is not None else None
    draw_crop(gray_crop, "VERSION 2 — Enhanced Gray",
              text_gray, conf_gray,
              (0, 255, 0) if conf_gray >= 60 else (0, 0, 255))

    # ── Version 3 — Colour ────────────────────────────────────────────────────
    draw_crop(crop_original, "VERSION 3 — Colour",
              text_col, conf_col,
              (0, 255, 0) if conf_col >= 60 else (0, 0, 255))

    # ── Winner ────────────────────────────────────────────────────────────────
    cv2.line(panel, (10, y_cursor), (panel_w - 10, y_cursor),
             (100, 100, 100), 1)
    y_cursor += 15

    results = {
        "Binary": conf_bin,
        "Gray"  : conf_gray,
        "Colour": conf_col,
    }
    winner      = max(results, key=results.get)
    winner_conf = results[winner]
    put(panel, f"WINNER: {winner} ({winner_conf}%)",
        (10, y_cursor), (0, 255, 255), 0.6, 2)

    return panel
