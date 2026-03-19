import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def load_rescue_model():
    model = models.resnet50(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model.eval()
    return model

def detect_damage(image, model):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, index = torch.max(probabilities, 0)
    
    classes = ['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
    
    # Generate Heatmap
    heatmap_img = get_heatmap(model, input_tensor, image)
    
    return classes[index], confidence.item(), heatmap_img

def get_heatmap(model, input_tensor, original_image):
    # Target the last layer of ResNet50
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    # Prepare original image for overlay
    img_resized = original_image.resize((224, 224))
    img_array = np.float32(img_resized) / 255
    
    # Create the heatmap overlay
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    return visualization