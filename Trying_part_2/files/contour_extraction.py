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

    # ── Step 2 — Dilation ─────────────────────────────────────────────────────
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    word_mask = cv2.dilate(binary_img, kernel, iterations=1)

    if debug:
        show("Step 2 - Dilated Word Mask", word_mask)
        cv2.waitKey(0)
        print(f"[DEBUG] Step 2 — dilation done with kernel (15,3)")

    # ── Step 3 — Find contours ────────────────────────────────────────────────
    contours, _ = cv2.findContours(
        word_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if debug:
        vis3 = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis3, contours, -1, (0, 255, 255), 1)
        cv2.putText(vis3, f"Total contours: {len(contours)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        show("Step 3 - All Contours", vis3)
        cv2.waitKey(0)
        print(f"[DEBUG] Step 3 — found {len(contours)} contours")

    if not contours:
        print("[DEBUG] No contours found — returning None")
        return None, None, None

    # ── Step 4 — Area filter ──────────────────────────────────────────────────
    bin_h, bin_w = binary_img.shape
    img_center   = np.array([bin_w // 2, bin_h // 2])

    if debug:
        vis4           = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        kept, rejected = [], []
        for cnt in contours:
            (kept if cv2.contourArea(cnt) >= min_area else rejected).append(cnt)
        cv2.drawContours(vis4, rejected, -1, (0, 0, 255), 1)
        cv2.drawContours(vis4, kept,     -1, (0, 255, 0), 1)
        cv2.drawMarker(vis4, tuple(img_center), (255, 0, 0),
                       cv2.MARKER_CROSS, 20, 2)
        cv2.putText(vis4, f"Kept: {len(kept)}  Rejected: {len(rejected)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        show("Step 4 - Area Filter  (green=kept  red=rejected  blue=center)", vis4)
        cv2.waitKey(0)
        print(f"[DEBUG] Step 4 — kept {len(kept)}  rejected {len(rejected)}")

    # ── Step 5 — Find closest contour to image center ─────────────────────────
    best_contour  = None
    min_distance  = float("inf")
    all_distances = []

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        box_center   = np.array([x + bw // 2, y + bh // 2])
        dist         = np.linalg.norm(box_center - img_center)
        all_distances.append((dist, (x, y, bw, bh), box_center))
        if dist < min_distance:
            min_distance = dist
            best_contour = (x, y, bw, bh)

    if debug:
        vis5 = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        for dist, (x, y, bw, bh), box_center in all_distances:
            cv2.rectangle(vis5, (x, y), (x + bw, y + bh), (0, 200, 200), 1)
            cv2.circle(vis5, tuple(box_center.astype(int)), 4, (200, 200, 0), -1)
            cv2.line(vis5, tuple(box_center.astype(int)),
                     tuple(img_center), (100, 100, 100), 1)
            cv2.putText(vis5, f"{dist:.0f}px", (x, max(y - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
        if best_contour:
            bx, by, bw, bh = best_contour
            cv2.rectangle(vis5, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
            cv2.putText(vis5, f"BEST ({min_distance:.0f}px)",
                        (bx, max(by - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.drawMarker(vis5, tuple(img_center), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)
        show("Step 5 - Closest to Center  (green=best)", vis5)
        cv2.waitKey(0)
        print(f"[DEBUG] Step 5 — best: {best_contour}  dist: {min_distance:.1f}px")

    if best_contour is None:
        print("[DEBUG] No valid contour found")
        return None, None, None

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