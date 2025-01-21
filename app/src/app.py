import os
import glob
import torch
import cv2
import logging
from torchvision import models, transforms
from torchvision.ops import nms
#from torchvision.models import  resnet50, ResNet50_Weights
from PIL import Image
from ultralytics import YOLO
import numpy as np
from paddleocr import PaddleOCR #, draw_ocr

# Set logging level to suppress YOLOv5 messages
logging.getLogger('ultralytics').setLevel(logging.WARNING)
# Set logging level to suppress PaddleOCR debug messages
logging.getLogger('ppocr').setLevel(logging.WARNING)

# Initialize the PaddleOCR reader
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load the ResNet model
# https://pytorch.org/vision/stable/models.html
model_cars = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_cars.eval()

# Load the YOLOv5 model for license plate detection
# Reference: License plate detection using YOLOv8
# https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/blob/main/README.md
model_plates = YOLO('../models/YOLO_license_plate_detector.pt')
model_plates.eval()

# Image transformation for YOLOv5
transform = transforms.Compose([transforms.ToTensor()])

# Function to detect objects (cars)
#def detect_objects_cars(image_path, iou_threshold=0.5):
def detect_objects_cars(image_cv2, iou_threshold=0.5):
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)

    #image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        results = model_cars(image_tensor)

    # Extract bounding boxes, confidence scores, and class labels
    predictions = results[0]
    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']

    # Apply Non-Maximum Suppression (NMS)
    keep = nms(boxes, scores, iou_threshold)
    boxes = boxes[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()
    
    #return prediction
    return {"boxes": boxes, "scores": scores, "labels": labels}

# Function to detect objects (license plates)
def detect_objects_plates(roi, iou_threshold=0.5):
    results = model_plates(roi)
    predictions = results[0]  # Access the first element of the list

    # Extract bounding boxes, confidence scores, and class labels
    boxes = predictions.boxes.xyxy
    scores = predictions.boxes.conf
    labels = predictions.boxes.cls

    # Apply Non-Maximum Suppression (NMS)
    keep = nms(boxes, scores, iou_threshold)
    boxes = boxes[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()

    return {"boxes": boxes, "scores": scores, "labels": labels}

# Function for OCR (License Plate Text) using PaddleOCR
def extract_text_from_image(roi):
    # Convert the image to grayscale
    #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Perform OCR using PaddleOCR
    #result = ocr.ocr(gray, cls=True)
    result = ocr.ocr(roi, cls=True)
    if result == [None]:
        #return "", gray, 0
        return "", 0
    text = " ".join([line[1][0] for line in result[0]])
    confidence = min([line[1][1] for line in result[0]])
    #return text, gray, confidence
    return text, confidence

# Process an image for detection and OCR
def process_image(image, frame_num=0, rotate=False, output_path=""):
    print(f"Processing image started")
    if rotate:
        image = cv2.rotate(image, cv2.ROTATE_180)

    # Data structure to store car, plate, and text data
    car_data = []

    #results_cars = detect_objects_cars(image_path)
    results_cars = detect_objects_cars(image)
    boxes = results_cars['boxes']

    #image = cv2.imread(image_path)
    for car_index, box_car in enumerate(boxes):
        c_x1, c_y1, c_x2, c_y2 = box_car.astype(int)

        # Extract the region of interest (ROI) for the car
        roi_car = image[c_y1:c_y2, c_x1:c_x2]            

        # Check if the ROI has non-zero dimensions
        if roi_car.shape[0] == 0 or roi_car.shape[1] == 0:
            #print(f"Skipping empty ROI for car: {box_car}")
            continue

        #print(f"Detected car box: {box_car}")
        #c_text = f"Detected car box: {box_car}"

        results_plates = detect_objects_plates(roi_car)
        boxes_plates = results_plates['boxes']
        for plate_index, box_plate in enumerate(boxes_plates):
            #print(f"Detected plate box: {box_plate}")
            #p_text = f"Detected license plate text: {box_plate}"
            bp_x1, bp_y1, bp_x2, bp_y2 = box_plate.astype(int)

            # Adjust coordinates relative to the original image
            p_x1, p_y1, p_x2, p_y2 = bp_x1 + c_x1, bp_y1 + c_y1, bp_x2 + c_x1, bp_y2 + c_y1

            # Extract the region of interest (ROI) for the number plate
            roi_plate = image[p_y1:p_y2, p_x1:p_x2]            

            # Extract text from the ROI
            #text, plate_image, confidence = extract_text_from_image(roi_plate)
            text, confidence = extract_text_from_image(roi_plate)

            if text != "":
                #print(f"Detected license plate text: {text}")

                # Initialize car entry
                if plate_index == 0:
                    car_entry = {
                        "car_id": car_index,
                        "car_box": box_car, #.tolist(),
                        "plates": []
                    }

                # Add plate data to car entry
                car_entry["plates"].append({
                    "plate_box": box_plate, #.tolist(),
                    "text": text,
                    "text_confidence": round(confidence, 5)
                })

                # Add car entry to car data
                car_data.append(car_entry)

    # Sometimes a car box overlaps more than one car and this results in there being more that one plate in the car box roi
    #
    # Need to flag if more than one plate is detected and then determine which one is the correct one.
    # this will need to be done by creating a list of the cars and the plates detected in each car box.
    # we can then compare the plates detected in each car box to the plates detected in other car boxes.
    # a car with only one plate detected will likey have the correct plate assigned to it.
    # if a car has more than one plate detected then we can compare the plates detected in that car box to the plates detected in other car boxes.
    # this can then be used to remove the false positives.
    # we will need to set up a suitable data structure to store the car, plate, and text data.
    # we will only add cars that have plates with text to the data structure.

    # Process car data to remove false positives
    print(f"Starting review of car data/plate. Length of car data: ", len(car_data))
    for car in car_data:
        print(f"Reviewing car: {car}")
        #car_box = car["car_box"]
        car_plates = car["plates"]
        if len(car_plates) > 1:
            # Compare plates with other cars
            for other_car in car_data:
                print(f"Reviewing other_car: {other_car}")
                if np.array_equal(other_car["car_box"], car["car_box"]): # Skip the same car
                    print(f"Skipping other_car as it's the same as car")
                    continue
                #other_car_box = other_car["car_box"]
                other_car_plates = other_car["plates"]
                l=len(car_plates)
                i = 0
                while i < l:
                    car_plate = car_plates[i]
                    car_plate_text = car_plate["text"]
                    for j in range(len(other_car_plates)):
                        other_car_plate = other_car_plates[j]
                        other_car_plate_text = other_car_plate["text"]
                        print(f"Reviewing car_plate: {car_plate}, and other_car_plate: {other_car_plate}")
                        # Check if the plate is a false positive
                        #if np.array_equal(car_plate_text, other_car_plate_text):
                        if car_plate_text == other_car_plate_text:
                            print(f"Removing false positive plate: {car_plate}")
                            car_plates.pop(i)
                            l-=1 # Decrement the length of car_plates list as we have removed an element
                    i+=1 # Increment the index for the car_plates list

                #for car_plate in car_plates: # Compare each car_plate with each other_car_plate
                #    car_plate_text = car_plate["text"]
                #    for other_car_plate in other_car_plates:
                #        other_car_plate_text = other_car_plate["text"]
                #        print(f"Reviewing car_plate: {car_plate}, and other_car_plate: {other_car_plate}")
                #        # Check if the plate is a false positive
                #        if np.array_equal(car_plate_text, other_car_plate_text):
                #            print(f"Removing false positive plate: {car_plate}")
                #            #if np.any(plate):  # or np.all(plate) depending on your condition ???????
                #            #if np.all(plate):  # or np.all(plate) depending on your condition ???????
                #            car_plates.remove(car_plate)
                #            break

    # Process car data to draw bounding boxes and text
    print(f"Starting drawing boxes and text")
    # Draw the frame number
    text_0 = "frame num: " + str(frame_num)
    cv2.putText(image, text_0, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) # Draws PLATE text above the plate bounding box
    for car in car_data:
        box_car = car["car_box"]
        car_id = car["car_id"]
        plates = car["plates"]
        c_x1, c_y1, c_x2, c_y2 = box_car.astype(int)
        for plate in plates:
            box_plate = plate["plate_box"]
            text_1 = "car: " + str(car_id) +", plate: " +plate["text"]
            text_2 = "confidence: " + str(plate["text_confidence"])
            bp_x1, bp_y1, bp_x2, bp_y2 = box_plate.astype(int)
            p_x1, p_y1, p_x2, p_y2 = bp_x1 + c_x1, bp_y1 + c_y1, bp_x2 + c_x1, bp_y2 + c_y1

            # Extract the region of interest (ROI) for the number plate
            cv2.rectangle(image, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2) # Draws CAR bounding box
            cv2.rectangle(image, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 2) # Draws PLATE bounding box inside the car bounding box
            cv2.putText(image, text_1, (p_x1, p_y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) # Draws PLATE text above the plate bounding box
            cv2.putText(image, text_2, (p_x1, p_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) # Draws PLATE text above the plate bounding box

    # Save the image with boxes
    if output_path!="":
        cv2.imwrite(output_path, image)
        print(f"Saved processed image to: {output_path}")

    print(f"Image processing finished\n")
    return image
  


# Process video frames
def process_videos(input_path, output_path, frame_gap=20):
    print(f"Processing videos from: {input_path}, to {output_path}\n")
    os.makedirs(output_path, exist_ok=True)
    input_files = glob.glob(os.path.join(input_path, "*.mp4"))

    print(f"video file path list: {input_files}")

    # Process each image file
    for input_file_path in input_files:
        input_file_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_path, f"processed_{input_file_name}")

        print(f"video file paths: {input_file_path}, {output_file_path}")

        cap = cv2.VideoCapture(input_file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_file_path}")
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        frame_num = 0
        next_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num != next_frame:
                frame_num += 1
                continue

            next_frame = frame_num + frame_gap
            print(f"processing frame: {frame_num}")
            # Process the frame
            #output_image_file_path = os.path.join(output_path, f"processed_{input_file_name}_{str(frame_num)}.jpg")
            processed_frame = process_image(frame, frame_num, rotate=True) #, output_image_file_path)

            # Check if processed_frame is None
            if processed_frame is None:
                print(f"Error processing frame {frame_num}")
                frame_num += 1
                continue

            #height, width, channels = processed_frame.shape
            #size = processed_frame.size
            #print(f"Processed frame shape: Height: {height}, Width: {width}, Channels: {channels}")
            #print(f"Processed frame size: {size} pixels")

            # Write the processed frame to the output video
            out.write(processed_frame)
            frame_num += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    print(f"Processing videos finished\n")
    

# Process video frames
def process_images(input_path, output_path):
    print(f"Processing images from: {input_path}, to {output_path}\n")
    os.makedirs(output_path, exist_ok=True)

    #input_image_files = glob.glob(os.path.join(input_path, "*.jpg"))
    # Define the list of file extensions
    extensions = ["*.jpg", "*.jpeg", "*.png"]

    # Collect all files with the specified extensions
    input_image_files = []
    for ext in extensions:
        input_image_files.extend(glob.glob(os.path.join(input_path, ext)))

    # Process each image file
    image_num = 0
    for input_image_file in input_image_files:
        input_file_name = os.path.basename(input_image_file)
        output_file_name = os.path.join(output_path, f"processed_{input_file_name}")
        print(f"Processing image: {input_file_name}, to {output_file_name}")
        image = cv2.imread(input_image_file)
        process_image(image, frame_num=image_num, output_path=output_file_name)
        image_num += 1


if __name__ == "__main__":
    print(f"Processing has started\n")

    input_folder = "../data"
    output_folder = "../data/processed"
    process_videos(input_folder, output_folder)

    process_images(input_folder, output_folder)

