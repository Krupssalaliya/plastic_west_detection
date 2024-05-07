from ultralytics import YOLO
import cv2
import keyboard

# Initialize YOLO model
model = YOLO("D:\\plastic_west_detection\\runs\\detect\\train\\weights\\best.pt")

# Make predictions and display them
results = model.predict(source="0", show=True)
print(results)

# Wait for 'q' key press to exit
print("Press 'q' to exit.")
keyboard.wait('q')

# Close OpenCV windows
cv2.destroyAllWindows()
