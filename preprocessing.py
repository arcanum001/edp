import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage import img_as_ubyte

from contouring import find_center_word
from detection import ocr_center_word


def preprocess(image,debug=True):
    # Original image in BGR (from camera)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Object
    enhanced = clahe_obj.apply(gray)  # Apply to get image!

    if debug:
        cv2.imshow('clahe', enhanced)  # Show enhanced image, not object
        cv2.waitKey(0)
    window_size = 25  # size of local region; adjust based on text size
    sauvola_thresh = threshold_sauvola(enhanced, window_size=window_size)
    binary_sauvola = enhanced > sauvola_thresh
    preprocessed = img_as_ubyte(binary_sauvola)
    preprocessed = 255 - preprocessed
    thresh=preprocessed
    #gray = cv2.equalizeHist(gray)

    #if debug:
     #   cv2.imshow('gray',gray)
      #  cv2.waitKey(0)
    #gray = cv2.GaussianBlur(gray, (3, 3), 0)
    #if debug:
        #cv2.imshow("Denoised", gray)
       # cv2.waitKey(0)
    #thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,blockSize=11,C=2)
    if debug:
        cv2.imshow("Adaptive Threshold", thresh)
        cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Close small gaps in letters (connect broken strokes)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Remove tiny noise dots
    if debug:
        cv2.imshow("Morphology", morph)
        cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    if debug:
        cv2.imshow("Morphology", morph)
        cv2.waitKey(0)



    #preprocessed = cv2.resize(morph, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    preprocessed=morph
    if debug:
        cv2.imshow("Final Preprocessed", preprocessed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return preprocessed


if __name__ == "__main__":
    # Load an example image (camera frame or saved image)
    Original_img = cv2.imread("D:\\Projects\\PythonProject\\img1.jpeg")  # Replace with your image

    # Preprocess
    preprocessed_img = preprocess(Original_img, True)


    # Save for inspection
    cv2.imwrite("preprocessed_output.png", preprocessed_img)
    binary = cv2.imread("D:\\Projects\\PythonProject\\preprocessed_output.png", cv2.IMREAD_GRAYSCALE)

    bbox, center_word = find_center_word(binary)

    if bbox:
        x, y, w, h = bbox
        vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 4)

        cv2.imwrite("Center_Word.png", center_word)
        #cv2.imshow("detected_word",center_word)
        cv2.imwrite("detected_center_word.png", vis)
        cv2.waitKey(0)
    image=cv2.imread("D:\Projects\PythonProject\Center_Word.png")
    print("detected word" ,ocr_center_word(image))

