import cv2
from roboflow import Roboflow
import time

# Replace with your actual API key
rf = Roboflow(api_key="HuoHipicKjXja6vmRU5W")

# Replace with the name of your project
project = rf.workspace().project("interlinked-swx7a")

# Replace with the appropriate model version
model = project.version(2).model

# Set desired frame dimensions and processing interval
desired_width = 640
desired_height = 480
desired_frame_interval = 1.0 / 60.0  # 30 FPS

# Initialize the webcam with the desired resolution
cap = cv2.VideoCapture(0)  # 0 for the default camera, you can change it if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Thresholding parameters
threshold_value = 128  # Adjust this value as needed
max_value = 255
threshold_type = cv2.THRESH_BINARY

# Initialize a variable to track the time of the last frame processing
last_frame_time = time.time()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if there's an issue reading the frame

    # Calculate the time elapsed since the last frame processing
    elapsed_time = time.time() - last_frame_time

    # Process the frame if enough time has passed
    if elapsed_time >= desired_frame_interval:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to the grayscale frame
        _, thresholded_frame = cv2.threshold(gray_frame, threshold_value, max_value, threshold_type)

        # Perform prediction on the thresholded frame
        prediction = model.predict(thresholded_frame, confidence=40, overlap=30).json()

        # Process the prediction results and draw bounding boxes on the original frame
        if 'objects' in prediction:
            for obj in prediction['objects']:
                label = obj['label']
                x, y, w, h = obj['relative_coordinates']
                x, y, w, h = int(x * frame.shape[1]), int(y * frame.shape[0]), int(w * frame.shape[1]), int(h * frame.shape[0])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow('Object Detection', frame)

        # Update the last frame time
        last_frame_time = time.time()

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
