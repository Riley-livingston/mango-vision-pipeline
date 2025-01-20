import sys
import io
from flask import Flask, request, jsonify
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

# Add TinySAM to the Python path and import
sys.path.append('/app/TinySAM')
from tinysam import sam_model_registry, SamPredictor

# Other imports
from image_processing import segment_objects, perform_batch_visual_search
from mmdet.apis import init_detector
from pinecone import Pinecone, ServerlessSpec

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask app
app = Flask(__name__)

# Set device to CPU
DEVICE = torch.device('cpu')

# Configure logging
import logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

# Data processing and vector representation functions
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define constants for model paths and configuration
CUSTOM_CONFIG_PATH = "/app/custom.py"
CUSTOM_WEIGHTS_PATH = "/app/epoch_50.pth"
CHECKPOINT_PATH = "/app/TinySAM/weights/tinysam.pth"
MODEL_TYPE = "vit_t"

# Load rtmDet model
rtmdet_model = init_detector(CUSTOM_CONFIG_PATH, CUSTOM_WEIGHTS_PATH, device=DEVICE)

# Load TinySAM model
# Manually load the checkpoint with map_location set to 'cpu'
model_data = torch.load(CHECKPOINT_PATH, map_location='cpu')

# Create the model instance without loading the checkpoint directly
sam = sam_model_registry[MODEL_TYPE]()

# Manually load the state dict into the model
sam.load_state_dict(model_data)

# Move the model to CPU (redundant here but ensures no CUDA calls)
sam.to('cpu')
mask_predictor = SamPredictor(sam)

# Initialize Pinecone
pc = Pinecone(api_key="982a316c-0b61-4932-b974-3d04c77b3996")
index = pc.Index("resnet-34-dropout-v33-index-hard-namespaced")

# Load ResNet34 model for visual search
resnet34_model_save_path = "/app/resnet34_model_yolo_sam_v6_356_30_dropout_v33_hard_jp.pth"
resnet34_model = models.resnet34(pretrained=False).to(DEVICE)
num_ftrs = resnet34_model.fc.in_features
num_classes = 25

# Adjust the fc layer to match the expected state_dict structure
resnet34_model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Assuming dropout was part of the original model during training
    nn.Linear(num_ftrs, num_classes)
)

resnet34_model.load_state_dict(torch.load(resnet34_model_save_path, map_location=DEVICE))
resnet34_model = resnet34_model.to(DEVICE)
resnet34_model.eval()

# If you still want to use the feature extractor without the final fully connected layer
feature_extractor = nn.Sequential(*list(resnet34_model.children())[:-1]).to(DEVICE)

# Load ResNet18 model for price type class prediction
resnet18_model_save_path = "/app/pkmn_price_type_classifier_resnet18_state_dict_v11_256.pth"
resnet18_model = models.resnet18(pretrained=False).to(DEVICE)
num_ftrs_18 = resnet18_model.fc.in_features
resnet18_model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Add dropout if the model was trained with it
    nn.Linear(num_ftrs_18, 5)  # Assuming 5 classes for price type prediction
)

resnet18_model.load_state_dict(torch.load(resnet18_model_save_path, map_location=DEVICE))
resnet18_model = resnet18_model.to(DEVICE)
resnet18_model.eval()

@app.route('/ping', methods=['GET'])
def ping():
    return "", 200

@app.route('/invocations', methods=['POST'])
def invocations():
    try:
        data = request.json
        encoded_image = data['instances'][0]
        cardGame = data['cardGame']  # Get the cardGame parameter from the request
        class_names = ["firstEditionHolofoil", "firstEditionNormal", "holofoil", "normal", "reverseHolofoil"]

        segmented_images, cropped_objects = segment_objects(encoded_image, rtmdet_model, mask_predictor, DEVICE)
        visual_search_results, price_type_predictions, cropped_image_data = perform_batch_visual_search(
            segmented_images, cropped_objects, index, feature_extractor, resnet18_model, class_names, transform, DEVICE, cardGame)

        return jsonify({
            "visual_search_results": visual_search_results,
            "price_type_predictions": price_type_predictions,
            "cropped_objects": cropped_image_data
        })
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
