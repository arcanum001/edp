import os
import time

import cv2

from MainFnc import sign_detect, sign_detect_paddleocr

image_folder = "D:\Projects\PythonProject\images"  # 🔁 change this
image_paths = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

print(f"Total images found: {len(image_paths)}")

# -----------------------------
# ⏱️ Start timing
# -----------------------------
start_time = time.perf_counter()

for img_path in image_paths:
    print(f"Processing {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        continue

    # ---- detection only ----
    sign_detect(img)
    #sign_detect_paddleocr(img)



# -----------------------------
# ⏱️ End timing
# -----------------------------
end_time = time.perf_counter()

total_time = end_time - start_time
avg_time = total_time / len(image_paths)

print("Start time :", start_time)
print("End time   :", end_time)
print("Total time :", total_time, "seconds")
print("Avg / image:", avg_time, "seconds")
print("FPS approx :", 1 / avg_time)
