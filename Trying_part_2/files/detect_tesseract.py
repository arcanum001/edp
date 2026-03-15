import cv2
import numpy as np
import pytesseract
import os

def enhance_for_ocr(crop_original, debug=False):

    # Step 1 — Grayscale
    gray = cv2.cvtColor(crop_original, cv2.COLOR_BGR2GRAY)
    if debug:
        show("Step 1 - Grayscale", gray)
        cv2.waitKey(0)
        print(f"[ENHANCE] Step 1 — min={gray.min()} max={gray.max()} mean={gray.mean():.1f}")

    # Step 2 — CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)
    if debug:
        show("Step 2 - CLAHE", gray)
        cv2.waitKey(0)
        print(f"[ENHANCE] Step 2 — min={gray.min()} max={gray.max()} mean={gray.mean():.1f}")

    # Step 3 — Sharpen
    sharpen_kernel = np.array([[ 0, -1,  0],
                                [-1,  5, -1],
                                [ 0, -1,  0]])
    gray = cv2.filter2D(gray, -1, sharpen_kernel)
    if debug:
        show("Step 3 - Sharpened", gray)
        cv2.waitKey(0)
        print(f"[ENHANCE] Step 3 — min={gray.min()} max={gray.max()} mean={gray.mean():.1f}")

    # Step 4 — Otsu threshold  ← THIS IS WHAT WAS MISSING
    # Forces background → 255 (white)
    # Forces text       → 0   (black)
    # Now padding of 255 matches background perfectly
    _, gray = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if debug:
        show("Step 4 - Otsu Threshold (text=black  bg=white)", gray)
        cv2.waitKey(0)
        print(f"[ENHANCE] Step 4 — min={gray.min()} max={gray.max()} mean={gray.mean():.1f}")

    # Step 5 — Side by side
    if debug:
        original_gray = cv2.cvtColor(crop_original, cv2.COLOR_BGR2GRAY)
        comparison    = np.hstack([original_gray, gray])
        show("Step 5 - Before (left) vs After (right)", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return gray

def show(title, image, max_width=900, max_height=600):
    h, w  = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    cv2.imshow(f"{title}  [{w}x{h}]", image)


def ocr_center_word(word_img, min_conf=0, debug=False):

    # ── Step 1 — Original ─────────────────────────────────────────────────────
    if debug:
        show("OCR Step 1 - Original Input", word_img)
        cv2.waitKey(0)
        print(f"[OCR] Step 1 — input shape: {word_img.shape}")

    # ── Step 2 — Upscale if too small ─────────────────────────────────────────
    h, w      = word_img.shape[0:2]
    TARGET_H  = 64
    if h < TARGET_H:
        scale    = TARGET_H / h
        word_img = cv2.resize(word_img, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        if debug:
            show(f"OCR Step 2 - Upscaled  (was {h}px tall → now {word_img.shape[0]}px)",
                 word_img)
            cv2.waitKey(0)
            print(f"[OCR] Step 2 — upscaled by {scale:.2f}x  "
                  f"new shape: {word_img.shape}")
    else:
        if debug:
            show(f"OCR Step 2 - No upscale needed  (height={h}px >= {TARGET_H}px)",
                 word_img)
            cv2.waitKey(0)
            print(f"[OCR] Step 2 — no upscale needed  height={h}px")

    # ── Step 3 — Padding ──────────────────────────────────────────────────────
    PAD = 10

    if debug:
        # Show BEFORE padding with red border to show edges
        before_pad = word_img.copy()
        if len(before_pad.shape) == 2:
            before_pad = cv2.cvtColor(before_pad, cv2.COLOR_GRAY2BGR)
        # Draw red rectangle right at the edge to show text touching boundary
        cv2.rectangle(before_pad, (0, 0),
                      (before_pad.shape[1]-1, before_pad.shape[0]-1),
                      (0, 0, 255), 2)
        show("OCR Step 3a - BEFORE Padding  (red=image boundary)", before_pad)
        cv2.waitKey(0)

    word_img = cv2.copyMakeBorder(
        word_img,
        top=PAD, bottom=PAD, left=PAD, right=PAD,
        borderType=cv2.BORDER_CONSTANT,
        value=255     # white padding
    )

    if debug:
        # Show AFTER padding with green border to show new boundary
        after_pad = word_img.copy()
        if len(after_pad.shape) == 2:
            after_pad = cv2.cvtColor(after_pad, cv2.COLOR_GRAY2BGR)
        # Draw green rectangle at new edge
        cv2.rectangle(after_pad, (0, 0),
                      (after_pad.shape[1]-1, after_pad.shape[0]-1),
                      (0, 255, 0), 2)
        # Draw blue rectangle showing where old boundary was
        cv2.rectangle(after_pad, (PAD, PAD),
                      (after_pad.shape[1]-PAD-1, after_pad.shape[0]-PAD-1),
                      (255, 0, 0), 1)
        show("OCR Step 3b - AFTER Padding  (green=new boundary  blue=old boundary)",
             after_pad)
        cv2.waitKey(0)
        print(f"[OCR] Step 3 — added {PAD}px white padding all around  "
              f"new shape: {word_img.shape}")

    # ── Step 4 — OCR ──────────────────────────────────────────────────────────
    config = (
        "--oem 1 --psm 8 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
        "-c tessedit_char_blacklist=!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
        "-c preserve_interword_spaces=0 "
        "-c load_system_dawg=0 "
        "-c load_freq_dawg=0 "
    )

    data = pytesseract.image_to_data(
        word_img,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    if debug:
        print(f"[OCR] Step 4 — Tesseract raw output:")
        for i, text in enumerate(data["text"]):
            if str(data["conf"][i]) != "-1":
                print(f"         [{i}]  text='{text}'  conf={data['conf'][i]}%")

    # ── Step 5 — Find best result ─────────────────────────────────────────────


    best_text = None
    best_conf = 0

    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        conf = int(data["conf"][i])
        if conf > best_conf:
            best_conf = conf
            best_text = text

    if debug:
        # Draw result on the padded image
        result_vis = word_img.copy()
        if len(result_vis.shape) == 2:
            result_vis = cv2.cvtColor(result_vis, cv2.COLOR_GRAY2BGR)
        label = f"{best_text} ({best_conf}%)" if best_text else "Nothing found"
        color = (0, 255, 0) if best_conf >= min_conf else (0, 0, 255)
        cv2.putText(result_vis, label,
                    (5, result_vis.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        show(f"OCR Step 5 - Result  (green=accepted  red=rejected)", result_vis)
        cv2.waitKey(0)
        print(f"[OCR] Step 5 — best: '{best_text}'  conf: {best_conf}%")

    # ── Step 6 — Confidence gate ──────────────────────────────────────────────
    if best_conf < min_conf:
        if debug:
            print(f"[OCR] Step 6 — REJECTED  {best_conf}% < {min_conf}%")
            cv2.destroyAllWindows()
        return None, 0

    if debug:
        print(f"[OCR] Step 6 — ACCEPTED  '{best_text}'  {best_conf}%")
        cv2.destroyAllWindows()

    return best_text, best_conf


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    IMAGES_FOLDER = (r"D:\Projects\PythonProject\images"
                     )

    # ── Allowed image extensions ──────────────────────────────────────────────
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    # ── Loop over all images in folder ────────────────────────────────────────
    for filename in os.listdir(IMAGES_FOLDER):
        if not filename.lower().endswith(IMG_EXTENSIONS):
            continue  # skip non-image files

        image_path = os.path.join(IMAGES_FOLDER, filename)
        print(f"\n[INFO] Processing: {filename}")

        # ── Load image ─────────────────────────────────────────────
        original = cv2.imread(image_path)
        if original is None:
            print(f"[ERROR] Could not load: {image_path}")
            continue

        # ── Preprocess → binary ────────────────────────────────────
        from binarization import *
        binary = binarize(original, debug=False)

        # ── Find center word ───────────────────────────────────────
        from contour_extraction import *
        bbox, crop_binary, crop_original = find_center_word(
            binary_img=binary,
            original_img=original,
            min_area=100,
            debug=False
        )

        if bbox is None:
            print("[WARN] No word found in this image")
            continue

        # ── Prepare 3 versions ─────────────────────────────────────
        img_binary = crop_binary
        img_gray = enhance_for_ocr(crop_original, debug=False)
        img_colour = crop_original

        # ── Run OCR on all 3 ──────────────────────────────────────
        text_binary, conf_binary = ocr_center_word(img_binary, min_conf=0, debug=False)
        text_gray, conf_gray = ocr_center_word(img_gray, min_conf=0, debug=False)
        text_colour, conf_colour = ocr_center_word(img_colour, min_conf=0, debug=False


                                                   )

        # ── Compare results ───────────────────────────────────────
        results = {
            "Binary": (text_binary, conf_binary),
            "Grayscale": (text_gray, conf_gray),
            "Colour": (text_colour, conf_colour),
        }
        winner = max(results, key=lambda k: results[k][1])
        print(text_binary,conf_binary)
        print(text_gray,conf_gray)
        print(text_colour,conf_colour)
        print(f"  Winner → {winner}: '{results[winner][0]}' ({results[winner][1]}%)")
