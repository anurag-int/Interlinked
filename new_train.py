from ultralytics import YOLO 
import cv2
# Load a model 
model = YOLO("yolov8n.pt") 
# load a pretrained model 
# Use the model 
results = model.train(data="E:\Python codes\Learn\OpenCV\Interlinked\Interlinked.v1i.yolov8\data.yaml", epochs=300) # train the model
#results = model.train(data="coco128.yaml", epochs=1)
#$results = model.predict(data="config.yaml")
# results = model.val() # evaluate model performance on the validation data set 0
results = model.val()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open the webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read frame from the webcam.")
        break

    # Perform object detection on the frame
    results = model(frame)
    # Display the frame with object detection results
       
    cv2.imshow('Object Detection', frame)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()# predict on an image
# success = YOLO("yolov8n.pt").export(format="onnx") # export a model to ONNX