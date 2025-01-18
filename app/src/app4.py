import torch
import cv2
import pytesseract
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

# Load the YOLOv5 model
model_cars = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_cars.eval()

# Load the YOLOv5 model for license plate detection
model_plates = YOLO('license_plate_detector.pt')
model_plates.eval()

# Image transformation for YOLOv5
transform = transforms.Compose([transforms.ToTensor()])

# Function to detect objects (cars)
def detect_objects_cars(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model_cars(image_tensor)
    
    return prediction

# Function to detect objects (license plates)
def detect_objects_plates(image_path, box):
    #image = Image.open(image_path)
    image = cv2.imread(image_path)
    x1, y1, x2, y2 = box
    roi = image[y1:y2, x1:x2]
    results = model_plates(roi)
    return results

# Function for OCR (License Plate Text)
def extract_text_from_image(image, box):
    x1, y1, x2, y2 = box
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    return text, thresh


# Process an image for detection and OCR
def process_image(image_path, output_path):
    print(f"Processing image: {image_path}")
    results_cars = detect_objects_cars(image_path)

    image = cv2.imread(image_path)
    for box_car in results_cars[0]['boxes']:
        c_x1, c_y1, c_x2, c_y2 = box_car.int().numpy()
        cv2.rectangle(image, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2)
        print(f"Detected car box: {box_car}")

        results_plates = detect_objects_plates(image_path, (c_x1, c_y1, c_x2, c_y2))
        for license_plate in results_plates.boxes.data.tolist():
            p_x1, p_y1, p_x2, p_y2, score, class_id = license_plate

            cv2.rectangle(image, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 2)
            print(f"Detected plate box: {box_plate}")
            
            # Extract text from the ROI
            text, thresh = extract_text_from_image(image, (p_x1, p_y1, p_x2, p_y2))
            print(f"Detected license plate text: {text}")
            
            # Get bounding boxes for each character
            boxes = pytesseract.image_to_boxes(thresh, config='--psm 6')
            h, w, _ = image.shape
            for b in boxes.splitlines():
                b = b.split(' ')
                tx1, ty1, tx2, ty2 = int(b[1]) + x1, h - int(b[2]) + y1, int(b[3]) + x1, h - int(b[4]) + y1
                cv2.rectangle(image, (tx1, ty1, tx2, ty2), (255, 0, 0), 2)
    
    # Save the image with boxes
    cv2.imwrite(output_path, image)
    print(f"Saved processed image to: {output_path}")

if __name__ == "__main__":
    # Example of image processing
    process_image("data/car_image.jpg", "data/processed_car_image.jpg")