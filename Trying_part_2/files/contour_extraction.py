import cv2
import numpy as np
def show(title, image, max_width=1200, max_height=900):
    """
    Show image scaled down to fit screen.
    Never zooms in — only scales down if too large.
    """
    h, w = image.shape[:2]

    # Calculate scale to fit within max dimensions
    scale = min(max_width / w, max_height / h, 1.0)  # 1.0 = never upscale

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = image

    cv2.imshow(f"{title}  [{w}x{h}]", resized)
def find_center_word(binary_img, original_img=None, min_area=100, debug=False):
    """
    Args:
        binary_img   : preprocessed binary image (required)
        original_img : original colour/grey image (optional)
        min_area     : minimum contour area to consider
        debug        : show step by step windows

    Returns:
        best_contour     : (x, y, w, h) in binary coords
        cropped_binary   : word crop from binary image
        cropped_original : word crop from original upscaled to match binary (None if not provided)
    """

    # ── Calculate scale factor between binary and original ────────────────────
    if original_img is not None:
        orig_h, orig_w = original_img.shape[:2]
        bin_h,  bin_w  = binary_img.shape[:2]
        scale_x = orig_w / bin_w   # e.g. 1000/2000 = 0.5
        scale_y = orig_h / bin_h   # e.g.  800/1600 = 0.5
        if debug:
            print(f"[DEBUG] Binary size  : {bin_w}x{bin_h}")
            print(f"[DEBUG] Original size: {orig_w}x{orig_h}")
            print(f"[DEBUG] Scale factor : x={scale_x:.3f}  y={scale_y:.3f}")
    else:
        scale_x = scale_y = 1.0

    # ── Step 1 — Show inputs ──────────────────────────────────────────────────
    if debug:
        show("Step 1 - Binary Image", binary_img)
        cv2.waitKey(0)
        print(f"[DEBUG] Step 1 — binary shape: {binary_img.shape}")
        if original_img is not None:
            show("Step 1 - Original Image", original_img)
            cv2.waitKey(0)
            print(f"[DEBUG] Step 1 — original shape: {original_img.shape}")

    # ── FIXED TARGET ROI (Replaces fragile contour logic) ─────────────────────
    bin_h, bin_w = binary_img.shape[:2]
    
    # Define a central box: 60% of image width, 20% of image height
    bw = int(bin_w * 0.60)
    bh = int(bin_h * 0.20)
    x  = int((bin_w - bw) / 2)
    y  = int((bin_h - bh) / 2)
    
    best_contour = (x, y, bw, bh)

    # ── Step 6 — Crop binary ──────────────────────────────────────────────────
    x, y, bw, bh = best_contour
    cropped_binary = binary_img[y:y + bh, x:x + bw]

    # ── Step 7 — Crop original using SCALED coords + upscale to match binary ──
    cropped_original = None
    ox = oy = obw = obh = 0

    if original_img is not None:

        # Convert binary coords → original coords
        ox  = int(x  * scale_x)
        oy  = int(y  * scale_y)
        obw = int(bw * scale_x)
        obh = int(bh * scale_y)

        # Clamp to original image bounds
        orig_h, orig_w = original_img.shape[:2]
        ox  = max(0, min(ox,  orig_w - 1))
        oy  = max(0, min(oy,  orig_h - 1))
        obw = max(1, min(obw, orig_w - ox))
        obh = max(1, min(obh, orig_h - oy))

        cropped_original = original_img[oy:oy + obh, ox:ox + obw]

        # Upscale original crop to match binary crop size exactly
        target_h, target_w = cropped_binary.shape[:2]
        cropped_original   = cv2.resize(
            cropped_original,
            (target_w, target_h),
            interpolation=cv2.INTER_CUBIC     # smooth for colour/grey image
        )

        if debug:
            print(f"[DEBUG] Binary coords  : x={x}  y={y}  w={bw}  h={bh}")
            print(f"[DEBUG] Original coords: x={ox} y={oy} w={obw} h={obh}")
            print(f"[DEBUG] Original crop upscaled to: {cropped_original.shape}")

    if debug:
        # ── Crop location on binary ───────────────────────────────────────────
        vis6b = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis6b, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.drawMarker(vis6b, tuple(img_center), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)
        cv2.putText(vis6b, f"Binary crop: ({x},{y})  {bw}x{bh}px",
                    (x, max(y - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        show("Step 6 - Crop Location on Binary", vis6b)
        cv2.waitKey(0)

        # ── Crop location on original ─────────────────────────────────────────
        if original_img is not None:
            vis6o = original_img.copy()
            cv2.rectangle(vis6o, (ox, oy), (ox + obw, oy + obh), (0, 255, 0), 2)
            cv2.putText(vis6o, f"Original crop: ({ox},{oy})  {obw}x{obh}px",
                        (ox, max(oy - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            show("Step 6 - Crop Location on Original", vis6o)
            cv2.waitKey(0)

        # ── Show both crops upscaled for visibility ───────────────────────────
        scale_up   = max(1, 100 // max(bh, 1))

        binary_big = cv2.resize(cropped_binary, None,
                                fx=scale_up, fy=scale_up,
                                interpolation=cv2.INTER_NEAREST)
        show(f"Step 6 - Cropped Binary  ({bw}x{bh}  shown {scale_up}x)", binary_big)
        cv2.waitKey(0)

        if cropped_original is not None:
            orig_big = cv2.resize(cropped_original, None,
                                  fx=scale_up, fy=scale_up,
                                  interpolation=cv2.INTER_NEAREST)
            show(f"Step 6 - Cropped Original  (upscaled to {cropped_original.shape[1]}x"
                 f"{cropped_original.shape[0]}  shown {scale_up}x)", orig_big)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        print(f"[DEBUG] Step 6 — binary crop   : {cropped_binary.shape}")
        print(f"[DEBUG] Step 6 — original crop : "
              f"{cropped_original.shape if cropped_original is not None else 'N/A'}")

    return best_contour, cropped_binary, cropped_original


# ── Run directly on one image ─────────────────────────────────────────────────
if __name__ == "__main__":
    original = cv2.imread("D:\\Projects\\PythonProject\\img1.jpeg")  # Replace with your image


    if original is None:
        print(f"[ERROR] Could not load: ")
        exit()

    from binarization import binarize
    binary = binarize(original, debug=False)

    bbox, cropped = find_center_word(binary,original, min_area=100, debug=True)

    if bbox:
        print(f"\n[RESULT] Center word found at {bbox}")
    else:
        print("\n[RESULT] No center word found")