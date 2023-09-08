import yaml
from yolov5.models import YOLO
from yolov5.utils import general as g_utils

# Load the YAML configuration file
with open("tdata.yaml") as file:
    data = yaml.safe_load(file)

# Define the YOLO model configuration
model_config = {
    "nc": data["nc"],  # Number of classes
    "names": data["names"],  # Class names
    "pretrained": False,  # Set to True if you want to use pretrained weights
    "cfg": data["model"]["cfg"],  # Path to model config file (e.g., yolov8n.yaml)
}

# Create the YOLO model
model = YOLO(**model_config)

# Train the model using the training and validation data from the YAML file
train_data = g_utils.datasets.LoadImages(data["train"])
val_data = g_utils.datasets.LoadImages(data["val"])
model.train(train_path=train_data, val_path=val_data, epochs=100)

# Save the trained model weights
model.save("model.weights")
