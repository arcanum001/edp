import cv2
import numpy as np

def find_center_word(binary_img,min_area=100):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    word_mask = cv2.dilate(binary_img, kernel, iterations=1)
    cv2.imshow("word",word_mask)

    contours, _ = cv2.findContours(
        word_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None
    h, w = binary_img.shape
    img_center = np.array([w // 2, h // 2])
    best_contour = None
    min_distance = float("inf")
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        box_center = np.array([x + bw // 2, y + bh // 2])

        dist = np.linalg.norm(box_center - img_center)

        if dist < min_distance:
            min_distance = dist
            best_contour = (x, y, bw, bh)

    if best_contour is None:
        return None, None

    x, y, bw, bh = best_contour
    cropped_word = binary_img[y:y + bh, x:x + bw]

    return best_contour, cropped_word



def find_center_word_paddle(binary_img,original_img,tesseract,min_area=100):
   # cv2.imshow("original image",original_img)
   # cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    word_mask = cv2.dilate(binary_img, kernel, iterations=1)
    cv2.imshow("word",word_mask)

    contours, _ = cv2.findContours(
        word_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None
    h, w = binary_img.shape
    img_center = np.array([w // 2, h // 2])
    best_contour = None
    min_distance = float("inf")
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        box_center = np.array([x + bw // 2, y + bh // 2])

        dist = np.linalg.norm(box_center - img_center)

        if dist < min_distance:
            min_distance = dist
            best_contour = (x, y, bw, bh)

    if best_contour is None:
        return None, None

    x, y, bw, bh = best_contour

    cropped_word = original_img[y:y + bh, x:x + bw]




    return best_contour, cropped_word
