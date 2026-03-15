import cv2
import numpy as np
import pytesseract
def ocr_center_word(word_img):
    h,w=word_img.shape[0:2]
    target_height=40
    if h<target_height:
        scale=target_height/h
        word_img = cv2.resize(word_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    #cv2.imshow("Step 2: Resized", word_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    config="--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    text=pytesseract.image_to_string(word_img, config=config)
    cv2.destroyAllWindows()
    return text.strip()