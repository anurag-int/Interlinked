import cv2
from roboflow import Roboflow

# Replace with your actual API key
rf = Roboflow(api_key="HuoHipicKjXja6vmRU5W")

# Replace with the name of your project
project = rf.workspace().project("interlinked-swx7a")

# Replace with the appropriate model version
model = project.version(2).model

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 for the default camera, you can change it if needed

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if there's an issue reading the frame

    # Perform prediction on the frame
    prediction = model.predict(frame, confidence=40, overlap=30).json()

    # Check if 'objects' key exists in the prediction dictionary
    if 'objects' in prediction:
        # Process the prediction results and draw bounding boxes on the frame
        for obj in prediction['objects']:
            label = obj['label']
            x, y, w, h = obj['relative_coordinates']
            x, y, w, h = int(x * frame.shape[1]), int(y * frame.shape[0]), int(w * frame.shape[1]), int(h * frame.shape[0])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Object Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
