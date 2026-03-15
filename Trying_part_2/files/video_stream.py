

import cv2
import numpy as np
from binarization import *
from contour_extraction import *
from detect_tesseract import *
from result_panel import *
def run_video(source=0, process_every=5):
    """
    source       : 0 = webcam, or path to video file
    process_every: run detection every N frames (reduces CPU load)
                   5 = process 1 in every 5 frames
    """

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    print(f"[INFO] Starting video stream  —  processing every {process_every} frames")
    print(f"[INFO] Press 'q' to quit")

    frame_count  = 0

    # Cached results — shown every frame but only updated every N frames
    cached_panel      = None
    cached_bbox       = None
    cached_crop_bin   = None
    cached_crop_orig  = None
    cached_text_bin   = None;  cached_conf_bin  = 0
    cached_text_gray  = None;  cached_conf_gray = 0
    cached_text_col   = None;  cached_conf_col  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended")
            break

        frame_count += 1

        # ── Draw crosshair on live feed ───────────────────────────────────────
        h_f, w_f = frame.shape[:2]
        cv2.line(frame, (w_f//2, 0),   (w_f//2, h_f),   (0, 0, 255), 1)
        cv2.line(frame, (0, h_f//2),   (w_f, h_f//2),   (0, 0, 255), 1)

        # ── Process every N frames ────────────────────────────────────────────
        if frame_count % process_every == 0:
            print(f"[FRAME {frame_count}] Processing...")

            # Step 1 — Binarize
            binary = binarize(frame, debug=False)

            # Step 2 — Find center word
            bbox, crop_bin, crop_orig = find_center_word(
                binary_img   = binary,
                original_img = frame,
                min_area     = 100
            )

            if bbox is not None and crop_orig is not None:

                # Step 3 — Prepare 3 versions
                img_binary  = crop_bin
               # img_gray    = enhance_for_ocr(crop_orig)
                img_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img_colour  = crop_orig

                # Step 4 — OCR all 3
                text_bin,  conf_bin  = ocr_center_word(img_binary,-1)
                text_gray, conf_gray = ocr_center_word(img_gray,-1)
                text_col,  conf_col  = ocr_center_word(img_colour,-1)

                print(f"  Binary    : {str(text_bin):>15}  {conf_bin}%")
                print(f"  Gray      : {str(text_gray):>15}  {conf_gray}%")
                print(f"  Colour    : {str(text_col):>15}  {conf_col}%")

                # Cache results
                cached_bbox      = bbox
                cached_crop_bin  = crop_bin
                cached_crop_orig = crop_orig
                cached_text_bin  = text_bin;  cached_conf_bin  = conf_bin
                cached_text_gray = text_gray; cached_conf_gray = conf_gray
                cached_text_col  = text_col;  cached_conf_col  = conf_col

                # Draw bbox on live frame
                x, y, bw, bh = bbox
                # Scale bbox back to original frame size
                bin_h, bin_w = binary.shape[:2]
                sx = w_f / bin_w
                sy = h_f / bin_h
                fx  = int(x  * sx);  fy  = int(y  * sy)
                fbw = int(bw * sx);  fbh = int(bh * sy)
                cv2.rectangle(frame, (fx, fy), (fx+fbw, fy+fbh),
                              (0, 255, 0), 2)

            # Build result panel from cached data
            if cached_crop_orig is not None:
                cached_panel = build_result_panel(
                    frame,
                    cached_bbox,
                    cached_crop_bin,
                    cached_crop_orig,
                    cached_text_bin,  cached_conf_bin,
                    cached_text_gray, cached_conf_gray,
                    cached_text_col,  cached_conf_col,
                )

        else:
            # Non-processing frame — still draw cached bbox on live feed
            if cached_bbox is not None:
                binary    = binarize(frame, debug=False)
                bin_h, bin_w = binary.shape[:2]
                sx = w_f / bin_w
                sy = h_f / bin_h
                x, y, bw, bh = cached_bbox
                fx  = int(x  * sx);  fy  = int(y  * sy)
                fbw = int(bw * sx);  fbh = int(bh * sy)
                cv2.rectangle(frame, (fx, fy), (fx+fbw, fy+fbh),
                              (0, 255, 0), 2)

        # ── Frame counter on live feed ────────────────────────────────────────
        cv2.putText(frame, f"Frame: {frame_count}  "
                           f"Processing every: {process_every}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)

        # ── Display live feed + panel side by side ────────────────────────────
        if cached_panel is not None:
            # Match heights
            fh, fw = frame.shape[:2]
            ph      = cached_panel.shape[0]
            if ph != fh:
                cached_panel = cv2.resize(cached_panel, (cached_panel.shape[1], fh))
            combined = np.hstack([frame, cached_panel])
        else:
            combined = frame

        show("Video Feed + OCR Results  [q = quit]", combined,
             max_width=1400, max_height=800)

        # ── Key controls ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+'):
            process_every = max(1, process_every - 1)
            print(f"[INFO] process_every → {process_every}")
        elif key == ord('-'):
            process_every += 1
            print(f"[INFO] process_every → {process_every}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video(
        source        = 0,    # ← 0 = webcam, or "video.mp4"
        process_every = 5     # ← process 1 in every 5 frames
    )