import cv2
import tkinter as tk
import webbrowser
from PIL import Image, ImageTk

# Function to detect faces and ask the user to open the link
def detect_faces_and_ask():
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Update the GUI with the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame in the GUI
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, detect_faces_and_ask)  # Update every 10 milliseconds

    # If a face is detected, ask the user to open the link
    if len(faces) > 0:
        ask_label.config(text="Do you want to open YouTube?")
        open_button.config(state=tk.NORMAL)
    else:
        ask_label.config(text="")
        open_button.config(state=tk.DISABLED)

# Function to open YouTube when the button is clicked
def open_youtube():
    webbrowser.open("https://www.youtube.com")

# Initialize the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a Tkinter window
root = tk.Tk()
root.title("Face Detection")

# Create a label for displaying the webcam feed
label = tk.Label(root)
label.pack()

# Create a label for asking the user
ask_label = tk.Label(root, text="")
ask_label.pack()

# Create a button for opening YouTube
open_button = tk.Button(root, text="Visit Site", command=open_youtube, state=tk.DISABLED)
open_button.pack()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Start face detection and asking the user
detect_faces_and_ask()

# Run the Tkinter main loop
root.mainloop()

# Release the webcam when the application is closed
cap.release()
cv2.destroyAllWindows()
