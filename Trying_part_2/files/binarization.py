# ============================================================
# binarization.py  —  Step 1: Image binarization
# ============================================================

import cv2
from skimage.filters import threshold_sauvola
from skimage import img_as_ubyte


def binarize(image, debug=True):

    # ── Step 1: Grayscale ────────────────────────────────────────────────────
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        cv2.imshow('Step 1 - Grayscale', gray)
        cv2.waitKey(0)

    # ── Step 2: Gaussian Blur ────────────────────────────────────────────────
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if debug:
        cv2.imshow('Step 2 - Gaussian Blur', gray)
        cv2.waitKey(0)

    # ── Step 3: CLAHE (Local Contrast Enhancement) ───────────────────────────
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe_obj.apply(gray)
    if debug:
        cv2.imshow('Step 3 - CLAHE Enhanced', enhanced)
        cv2.waitKey(0)

    # ── Step 4: Sauvola Thresholding ─────────────────────────────────────────
    window_size = max(25, gray.shape[0] // 20)
    if window_size % 2 == 0:
        window_size += 1
    sauvola_thresh = threshold_sauvola(enhanced, window_size=window_size)
    binary_sauvola = enhanced > sauvola_thresh
    thresh = 255 - img_as_ubyte(binary_sauvola)   # invert: text=255, bg=0
    if debug:
        cv2.imshow('Step 4 - Sauvola Threshold', thresh)
        cv2.waitKey(0)

    # ── Step 5: Morphology CLOSE (fill broken strokes) ───────────────────────
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    if debug:
        cv2.imshow('Step 5 - Morph Close (fill gaps)', morph)
        cv2.waitKey(0)

    # ── Step 6: Morphology OPEN (remove noise dots) ──────────────────────────
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_open)
    if debug:
        cv2.imshow('Step 6 - Morph Open (remove noise)', morph)
        cv2.waitKey(0)

    # ── Step 7: Upscale for better OCR accuracy ──────────────────────────────
    preprocessed = cv2.resize(morph, None, fx=2, fy=2,
                               interpolation=cv2.INTER_CUBIC)
    if debug:
        cv2.imshow('Step 7 - Upscaled (2x)', preprocessed)
        cv2.waitKey(0)

    # ── Final ─────────────────────────────────────────────────────────────────
    if debug:
        cv2.imshow('Final Preprocessed', preprocessed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return preprocessed


if __name__ == "__main__":
    # Load an example image (camera frame or saved image)
    Original_img = cv2.imread("D:\\Projects\\PythonProject\\img1.jpeg")  # Replace with your image

    # Preprocess
    preprocessed_img = binarize(Original_img, True)
