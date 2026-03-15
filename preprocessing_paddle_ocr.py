import cv2


def preprocess_for_paddleocr(image, debug=False):
    # Step 1: Upscale if image is small (most important for OCR)
    h, w = image.shape[:2]
    if h < 64 or w < 64:
        image = cv2.resize(image, (w*3, h*3), interpolation=cv2.INTER_CUBIC)

    # Step 2: CLAHE on the luminance channel (keeps color, boosts contrast)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe_obj.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Step 3: Light denoise (optional)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)

    if debug:
        cv2.imshow("Preprocessed for OCR", enhanced)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return enhanced  # BGR image — feed directly to PaddleOCR