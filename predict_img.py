import os
import cv2
from ultralytics import YOLO

IMAGES_DIR = os.path.join('.', 'test_img')
OUTPUT_DIR = os.path.join('.', 'output')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the path to your model
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

# Load the YOLO model
model = YOLO(model_path)

# Define the threshold for detections
threshold = 0.5

# List all image files in the directory
image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    # Load the image
    image_path = os.path.join(IMAGES_DIR, image_file)
    image = cv2.imread(image_path)

    # Perform object detection
    results = model(image)[0]

    # Process detection results
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Draw bounding box and label on the image
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Save the annotated image
    output_image_path = os.path.join(OUTPUT_DIR, f'detected_{image_file}')
    cv2.imwrite(output_image_path, image)

print("Object detection completed.")
