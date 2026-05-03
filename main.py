import torch
import cv2
import easyocr
import numpy as np

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Load image
image_path = 'test3_car.jpg'  # change to your image file
img = cv2.imread(image_path)
assert img is not None, f"Image {image_path} not found"

# Inference
results = model(img)
results.print()

# OCR with EasyOCR
reader = easyocr.Reader(['en'])

for *box, conf, cls in results.xyxy[0]:
    x1, y1, x2, y2 = map(int, box)

    # Crop the plate region
    cropped = img[y1:y2, x1:x2]

    # Resize
    cropped_resized = cv2.resize(cropped, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Binarize image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ✅ Save the processed plate image for debugging
    cv2.imwrite("debug_plate.jpg", thresh)

    # OCR
    ocr_results = reader.readtext(thresh)

    print(f"\n📦 Detected Region: ({x1}, {y1}) → ({x2}, {y2}) | Confidence: {conf:.2f}")
    if ocr_results:
        for (_, text, prob) in ocr_results:
            print(f" 🔠 OCR Text: '{text}' (Confidence: {prob:.2f})")
    else:
        print(" 🕳️ No text detected in this region.")
