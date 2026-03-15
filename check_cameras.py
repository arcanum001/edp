import cv2

def find_cameras():
    print("Scanning for available cameras (indices 0 to 10)...")
    working_cameras = []
    
    for i in range(10):
        # CAP_DSHOW helps bypass some Windows-specific OpenCV camera bugs
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"  ✅ Camera found at index {i} (Resolution: {w}x{h})")
                working_cameras.append(i)
            cap.release()

    if not working_cameras:
        print("\n❌ No cameras found!")
    else:
        print(f"\n[INFO] You can use these indices in your script: {working_cameras}")

if __name__ == "__main__":
    find_cameras()