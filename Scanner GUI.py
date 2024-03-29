import cv2
import numpy as np
from pyzbar.pyzbar import decode
from datetime import datetime
import tkinter as tk
from PIL import Image, ImageTk

class QRBarcodeDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("QR & Barcode Detector")

        self.detecting = False

        self.start_button = tk.Button(master, text="Start Detecting", command=self.toggle_detection)
        self.start_button.pack()

        self.video_label = tk.Label(master)
        self.video_label.pack()

        self.cam = cv2.VideoCapture(0)
        self.last_save_time = datetime.now()

        self.update()

    def toggle_detection(self):
        self.detecting = not self.detecting
        if self.detecting:
            self.start_button.config(text="Stop Detecting")
        else:
            self.start_button.config(text="Start Detecting")

    def attempt_decode(self, frame):
        detected_codes = []
        decoded_objects = decode(frame)
        if decoded_objects:
            for obj in decoded_objects:
                data = obj.data.decode("utf-8")
                code_type = obj.type
                print(f"Decoded Data: {data}, Type: {code_type}")
                points = obj.polygon
                if len(points) >= 4:
                    points = np.array(points, dtype=np.int32)
                    cv2.polylines(frame, [points], True, (255, 0, 0), 3)
                    cv2.putText(frame, data, (obj.rect.left, obj.rect.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                detected_codes.append((data, code_type))
        return frame, detected_codes

    def save_codes_to_file(self, detected_codes):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"detected_codes_{timestamp}.txt"
        with open(file_name, "w") as file:
            for data, code_type in detected_codes:
                file.write(f"Type: {code_type}, Data: {data}\n")

    def update(self):
        ret, frame = self.cam.read()
        if ret:
            if self.detecting:
                processed_frame, detected_codes = self.attempt_decode(frame.copy())
                if detected_codes and (datetime.now() - self.last_save_time).total_seconds() > 5:
                    self.save_codes_to_file(detected_codes)
                    self.last_save_time = datetime.now()

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                frame = Image.fromarray(frame)
                processed_frame = Image.fromarray(processed_frame)

                frame = ImageTk.PhotoImage(image=frame)
                processed_frame = ImageTk.PhotoImage(image=processed_frame)

                self.video_label.config(image=frame)
                self.video_label.image = frame

            self.video_label.after(10, self.update)

    def __del__(self):
        if hasattr(self, 'cam'):
            self.cam.release()

def main():
    MachineLearning = tk.Tk()
    app = QRBarcodeDetectorApp(MachineLearning)
    MachineLearning.mainloop()

if __name__ == "__main__":
    main()
