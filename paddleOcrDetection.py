import cv2
import numpy as np
from paddleocr import PaddleOCR

# Initialize OCR (recognition only)
ocr = PaddleOCR(lang="en")

def paddle_ocr_center_word(word_img):
    h, w = word_img.shape[:2]

    target_height = 40
    if h < target_height:
        scale = target_height / h
        word_img = cv2.resize(word_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Convert to RGB (PaddleOCR expects RGB)
    if len(word_img.shape) == 2:
        word_img = cv2.cvtColor(word_img, cv2.COLOR_GRAY2RGB)
    else:
        word_img = cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB)

    # OCR
    result = ocr.ocr(word_img)

    text = ""
    if not result or result[0] is None:
        return text
    if result:
        for line in result[0]:
            recognized_text = line[1][0]  # the text
            confidence = line[1][1]  # the confidence score (0.0 to 1.0)

            if confidence > 0.0:  # optional: filter low-confidence results
                text += recognized_text + " "   # recognition result

    return text.strip()