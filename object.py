import cv2
from roboflow import Roboflow

# Replace with your actual API key
rf = Roboflow(api_key="HuoHipicKjXja6vmRU5W")

# Replace with the name of your project
project = rf.workspace().project("interlinked-swx7a")

# Replace with the appropriate model version
model = project.version(2).model

# Replace with the path to your input image
input_image_path = "test_image.jpeg"

# Perform prediction
prediction = model.predict(input_image_path, confidence=40, overlap=30).json()

# Save the annotated image
prediction_image_path = "prediction.jpg"
model.predict(input_image_path, confidence=40, overlap=30).save(prediction_image_path)

# Read and display the predicted image using OpenCV
predicted_image = cv2.imread(prediction_image_path)
resized = cv2.resize(predicted_image, (500,500), interpolation = cv2.INTER_AREA)
cv2.imshow('Resized', resized)
# cv2.imshow("Predicted Image", predicted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
