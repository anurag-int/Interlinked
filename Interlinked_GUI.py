import tkinter as tk
import cv2
import webbrowser
from PIL import Image, ImageTk
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import threading

# Initialize YOLO model
model = YOLO(r"E:\Python codes\Learn\OpenCV\Interlinked\best.pt")

# Global variable to track detected objects and their URLs
object_detected = False
object_urls = {
    "minion": "https://www.instagram.com/?utm_source=pwa_homescreen",
    "umbrella": "https://github.com/",
    # Add more object names and URLs as needed
}

# Function to perform object detection using YOLOv8
def detect_objects(frame):
    results = model.predict(frame)  # Detect objects directly on the BGR frame

    detected_objects = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            object_name = model.names[int(c)]
            detected_objects.append(object_name)

    # Check if an object is detected
    object_detected = bool(detected_objects)
    return object_detected, detected_objects

# Function to open a website based on the detected objects
def open_website():
    global object_detected
    cap = cv2.VideoCapture(0)  # Open the webcam

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame from the webcam.")
            break

        object_detected, detected_objects = detect_objects(frame)
        
        if object_detected:
            for obj in detected_objects:
                if obj in object_urls:
                    webbrowser.open(object_urls[obj])  # Open the URL corresponding to the detected object
            break  # Exit the loop after opening the website

        # Display the frame with object detection results
        cv2.imshow('Object Detection', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Create a Tkinter application window
root = tk.Tk()
root.title("YOLOv8 Object Detection")

# Create a label to display the webcam feed
video_label = tk.Label(root)
video_label.pack()

# Button to open a website
website_button = tk.Button(root, text="Open Website", command=open_website)
website_button.pack()

# Function to continuously update the webcam feed
def update_video():
    cap = cv2.VideoCapture(0)  # Open the webcam

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame from the webcam.")
            break

        # Perform object detection on the frame
        results = model.predict(frame)  # Detect objects directly on the BGR frame

        for r in results:
            annotator = Annotator(frame)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                annotator.box_label(b, model.names[int(c)])

            frame = annotator.result()

        # Display the frame with object detection results in the Tkinter window
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        video_label.config(image=img)
        video_label.image = img

        # Check if an object is detected
        object_detected, _ = detect_objects(frame)

        # Update the button state
        if object_detected:
            website_button.config(state=tk.NORMAL)
        else:
            website_button.config(state=tk.DISABLED)

        # Update the Tkinter window
        root.update()

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Start the function to continuously update the webcam feed in a separate thread
video_thread = threading.Thread(target=update_video)
video_thread.daemon = True  # Close the thread when the main program exits
video_thread.start()

# Start the Tkinter main loop
root.mainloop()
