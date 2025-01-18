import torch
import cv2
import pytesseract
from torchvision import models, transforms
from PIL import Image

# Load the pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Image transformation for Faster R-CNN
transform = transforms.Compose([transforms.ToTensor()])

# Function to detect objects (cars)
def detect_objects(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(image_tensor)
    
    return prediction

# Function for OCR (License Plate Text)
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    return text

# Process an image for detection and OCR
def process_image(image_path):
    print(f"Processing image: {image_path}")
    prediction = detect_objects(image_path)
    for box in prediction[0]['boxes']:
        print(f"Detected box: {box}")
    
    text = extract_text_from_image(image_path)
    print(f"Detected license plate text: {text}")

if __name__ == "__main__":
    # Example of image processing
    process_image("data/car_image.jpg")
