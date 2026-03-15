
import cv2
import numpy as np
import threading
import pytesseract

class ThreadedCamera:
    """
    Reads frames in a background thread so the buffer is constantly flushed.
    This guarantees we always get the most recent frame, eliminating lag.
    """
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.running = True
        self.ret = False
        self.frame = None
        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()
        else:
            self.running = False

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            self.ret = ret
            if ret:
                self.frame = frame

    def read(self):
        if self.frame is not None:
            return self.ret, self.frame.copy()
        return self.ret, None

    def release(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

# Tell pytesseract where to find the Tesseract-OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def run_video(source=0, process_every=5):
    """
    source       : 0 = webcam, or path to video file
    process_every: run detection every N frames (reduces CPU load)
                   5 = process 1 in every 5 frames
    """

    cap = ThreadedCamera(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    print(f"[INFO] Starting video stream  —  processing every {process_every} frames")
    print(f"[INFO] Press 'q' to quit")

    frame_count  = 0

    cached_results = []
    
    last_read_word = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended")
            break

        frame_count += 1
        

        # ── Process every N frames ────────────────────────────────────────────
        if frame_count % process_every == 0:
            print(f"[FRAME {frame_count}] Scanning text...")
            # Provide entire frame to deep-learning OCR model
            
            # Apply a mild sharpening filter to combat motion blur from gliding
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharp_frame = cv2.filter2D(frame, -1, kernel)
            
            # ── SPEED OPTIMIZATION: Crop a Region of Interest (ROI) ───────────
            # Only run heavy AI on the center of the image, ignoring the edges.
            h_f, w_f = sharp_frame.shape[:2]
            roi_w, roi_h = int(w_f * 0.60), int(h_f * 0.40)
            roi_x, roi_y = (w_f - roi_w) // 2, (h_f - roi_h) // 2
            
            roi_frame = sharp_frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            
            # ── LIGHTWEIGHT MODEL: Tesseract ──────────────────────────────
            gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            data = pytesseract.image_to_data(gray_roi, output_type=pytesseract.Output.DICT)
            
            cached_results = []
            for i in range(len(data['text'])):
                conf = int(data['conf'][i])
                text = data['text'][i].strip()
                if conf > 80 and text:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    # Convert to polygon format for existing drawing logic
                    adjusted_box = [
                        [x + roi_x, y + roi_y],
                        [x + w + roi_x, y + roi_y],
                        [x + w + roi_x, y + h + roi_y],
                        [x + roi_x, y + h + roi_y]
                    ]
                    # Tesseract gives conf out of 100, we scale to 1.0 to match earlier code
                    cached_results.append((adjusted_box, text, conf / 100.0))

        # ── Find the word closest to the center (The "Finger") ────────────────
        h_f, w_f = frame.shape[:2]
        center_x, center_y = w_f // 2, h_f // 2
        
        # Draw the active AI scanning zone to help the user aim
        roi_w, roi_h = int(w_f * 0.60), int(h_f * 0.40)
        roi_x, roi_y = (w_f - roi_w) // 2, (h_f - roi_h) // 2
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (80, 80, 80), 2)
        cv2.putText(frame, "AI SCAN ZONE", (roi_x + 5, roi_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

        # Draw a targeting crosshair and a solid red dot in the center ("virtual finger")
        cv2.drawMarker(frame, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)

        closest_dist = float('inf')
        target_word = None
        target_box = None

        for box, text, conf in cached_results:
            # Calculate the center of this word's bounding box
            poly_cx = sum(p[0] for p in box) / 4
            poly_cy = sum(p[1] for p in box) / 4
            dist = np.hypot(center_x - poly_cx, center_y - poly_cy)
            
            if dist < closest_dist:
                closest_dist = dist
                target_word = text
                target_box = box

        # ── Draw all boxes, highlighting the targeted one ─────────────────────
        for box, text, conf in cached_results:
            is_target = (box == target_box)
            color = (0, 0, 255) if is_target else (0, 255, 0) # Red if target, Green otherwise
            thickness = 4 if is_target else 2
            
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
            
            # Draw a dot at the center of the bounding box to show its anchor point
            b_cx = int(sum(p[0] for p in box) / 4)
            b_cy = int(sum(p[1] for p in box) / 4)
            cv2.circle(frame, (b_cx, b_cy), 5, color, -1)
            
            # Draw a tether line from the crosshair to the targeted word
            if is_target:
                cv2.line(frame, (center_x, center_y), (b_cx, b_cy), (0, 0, 255), 2)
            
            x, y = int(box[0][0]), int(box[0][1])
            display_text = f"{text} ({int(conf*100)}%)"
            cv2.putText(frame, display_text, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), thickness + 2)
            cv2.putText(frame, display_text, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)

        # ── Find the specific single word under the crosshair ─────────────────
        pointed_word = ""
        line_status = ""
        if target_word and target_box:
            box_left = min(p[0] for p in target_box)
            box_right = max(p[0] for p in target_box)
            box_width = box_right - box_left
            
            if box_width > 0:
                # Estimate where the crosshair falls along the phrase (0.0 to 1.0)
                ratio = (center_x - box_left) / box_width
                ratio = max(0.0, min(1.0, ratio))
                
                # Map percentage to string index
                char_index = int(ratio * len(target_word))
                
                # Split phrase into words and find the one at char_index
                words = target_word.split()
                current_idx = 0
                for w in words:
                    word_start = current_idx
                    word_end = current_idx + len(w)
                    if word_start <= char_index <= word_end + 1:
                        pointed_word = w
                        break
                    current_idx += len(w) + 1
                    
            # Clean punctuation for haptic engine (e.g. "paper," -> "paper")
            pointed_word = ''.join(e for e in pointed_word if e.isalnum()).lower()
            
            # ── Detect if this is the Start or End of a line ──────────────────
            b_cy = sum(p[1] for p in target_box) / 4
            b_h = max(p[1] for p in target_box) - min(p[1] for p in target_box)
            
            # Group words that share the same horizontal band (Y-axis)
            line_boxes = [b for b, t, c in cached_results if abs((sum(p[1] for p in b)/4) - b_cy) < (b_h / 2)]
            
            if line_boxes:
                # Find the furthest left and furthest right boxes in this line
                rightmost_box = max(line_boxes, key=lambda b: max(p[0] for p in b))
                leftmost_box = min(line_boxes, key=lambda b: min(p[0] for p in b))
                
                if target_box == rightmost_box:
                    line_status = " [END OF LINE]"
                elif target_box == leftmost_box:
                    line_status = " [START OF LINE]"
            
        # ── Debounce and Output the Single Word ───────────────────────────────
        if pointed_word and pointed_word != last_read_word:
            if closest_dist < 200: 
                print(f"\n[HAPTIC OUTPUT TRIGGER] --> {pointed_word}{line_status}")
                last_read_word = pointed_word
                
        # Draw the currently pointed word prominently at the top of the screen
        if pointed_word:
            display_text = f"TARGET: {pointed_word.upper()}{line_status}"
            cv2.putText(frame, display_text, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
            cv2.putText(frame, display_text, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        # ── Frame counter on live feed ────────────────────────────────────────
        cv2.putText(frame, f"Frame: {frame_count}  "
                           f"Processing every: {process_every}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)

        # ── Display live feed ─────────────────────────────────────────────────
        h, w = frame.shape[:2]
        scale = min(1200 / w, 900 / h, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            
        cv2.imshow("Deep Learning Scanner [q = quit]", frame)

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
        # To use your phone's camera, install an app like "IP Webcam" on Android
        # or "EpocCam" on iOS. Then, find the video stream URL in the app
        # and paste it here. It will look something like "http://192.168.1.5:8080/video"
        source        = "http://172.23.155.223:8080/video",
        process_every = 5     # ← process 1 in every 5 frames
    )