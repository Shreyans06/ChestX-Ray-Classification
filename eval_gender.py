import torch
import numpy as np
import cv2
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget , BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,deprocess_image,preprocess_image
from PIL import Image
import torch.nn as nn

model = torch.load( os.getcwd() + '/outputs/models/gender')
for param in model.parameters():
    param.requires_grad = True
model.eval()

test_path = os.getcwd() + '/datasets/Gender01/test'

image_url = test_path + '/male/JPCNN004.png'
img = np.array(Image.open(image_url))
img = cv2.resize(img, (224, 224))
img = np.float32(img) / 255
img = cv2.merge([img , img , img])
input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

target_layers = [model.features[-2].denselayer24]
targets = [ClassifierOutputTarget(0)]

with GradCAM(model=model, target_layers=target_layers , use_cuda= True) as cam:
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets )
    cam_image = show_cam_on_image(img, grayscale_cams[0, :])

cam = np.uint8(255*grayscale_cams[0, :])
cam = cv2.merge([cam, cam, cam])
images = np.hstack((np.uint8(255*img), cam , cam_image))
im = Image.fromarray(images)
im.save(os.getcwd() + '/outputs/grad-cam/test.png')