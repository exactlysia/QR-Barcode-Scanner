import cv2
import numpy as np
from pyzbar.pyzbar import decode

def add_title(img, title, font_scale=0.7, thickness=2):
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Low-Pass Filter (Gaussian Blur)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # High-Pass Filter (Laplacian for edge detection)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian_abs = np.uint8(np.absolute(laplacian))

    # Threshold Image (Adaptive Thresholding)
    adaptive_thresh = cv2.adaptiveThreshold(laplacian_abs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Convert grayscale images back to BGR for consistency in concatenation
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    laplacian_bgr = cv2.cvtColor(laplacian_abs, cv2.COLOR_GRAY2BGR)
    adaptive_thresh_bgr = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)
    
    return gray_bgr, blurred_bgr, laplacian_bgr, adaptive_thresh_bgr

def attempt_decode(frame):
    decoded_objects = decode(frame)
    if decoded_objects:
        for obj in decoded_objects:
            # Decode barcode/QR code data and print to terminal
            barcode_data = obj.data.decode("utf-8")
            print(f"Decoded Data: {barcode_data}")  # Print decoded data to terminal

            # Annotate decoded data on the frame
            points = obj.polygon
            if len(points) > 4:
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                points = hull
            n = len(points)
            for j in range(0, n):
                cv2.line(frame, tuple(points[j]), tuple(points[(j + 1) % n]), (255, 0, 0), 3)
            cv2.putText(frame, barcode_data, (obj.rect.left, obj.rect.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return frame, bool(decoded_objects)

def main():
    cam = cv2.VideoCapture(0)

    while True:
        success, frame = cam.read()
        if not success:
            print("Failed to grab frame")
            break

        gray_bgr, blurred_bgr, laplacian_bgr, adaptive_thresh_bgr = preprocess_frame(frame)
        
        titles = ['Original', 'Grayscale', 'Low-Pass Filter', 'High-Pass Filter', 'Adaptive Threshold', 'Processed Decoding', 'Original Decoding']
        images = [frame, gray_bgr, blurred_bgr, laplacian_bgr, adaptive_thresh_bgr]
        
        processed_frame, decoded = attempt_decode(adaptive_thresh_bgr.copy())
        images.append(processed_frame)
        
        if not decoded:
            original_decoded_frame, _ = attempt_decode(frame.copy())
        else:
            original_decoded_frame = frame.copy()
        images.append(original_decoded_frame)
        
        # Add titles to images
        for img, title in zip(images, titles):
            add_title(img, title)
        
        # Ensure all images are resized to the same dimensions for concatenation
        images_resized = [cv2.resize(img, (320, 240)) for img in images]
        
        # Create a grid to show all transformations and decodings
        row1 = cv2.hconcat(images_resized[:4])
        row2 = cv2.hconcat(images_resized[4:7] + [np.zeros_like(images_resized[0])])  # Placeholder to fill the grid
        
        grid = cv2.vconcat([row1, row2])
        
        cv2.imshow('All Transformations and Decodings', grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
