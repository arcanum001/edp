import cv2

from contouring import find_center_word,find_center_word_paddle
from detection import ocr_center_word
from paddleOcrDetection import paddle_ocr_center_word
from preprocessing import preprocess
from preprocessing_paddle_ocr import preprocess_for_paddleocr


def sign_detect(img):
    preprocessed_img = preprocess(img, False)
    bbox, center_word = find_center_word(preprocessed_img)
    print("detected word", ocr_center_word(center_word))
def sign_detect_paddleocr(img):
    enhanced_img = preprocess_for_paddleocr(img, False)
    binary_processed = preprocess(img, False)
    bbox, center_word = find_center_word_paddle(binary_processed,enhanced_img,False)
    if center_word is None:
        print("No word detected near center")
        return
   # cv2.imshow("centre word of paddleocr",center_word)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()
    print("paddle ocr detected word", paddle_ocr_center_word(center_word))

if __name__ == "__main__":
    Original_img = cv2.imread("D:\\Projects\\PythonProject\\img1.jpeg")
    #sign_detect(Original_img)
    sign_detect_paddleocr(Original_img)





