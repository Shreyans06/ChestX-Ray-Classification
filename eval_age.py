import torch
import numpy as np
import cv2
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget , BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,deprocess_image,preprocess_image
from PIL import Image
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data_list = pd.read_csv(os.getcwd() + '/datasets/XPAge01_RGB/XP/testdata.csv')
test_path = os.getcwd() + '/datasets/XPAge01_RGB/XP/JPGs'

model = torch.load( os.getcwd() + '/outputs/models/age')
for param in model.parameters():
    param.requires_grad = True
model.eval()


test_path = os.getcwd() + '/datasets/XPAge01_RGB/XP/JPGs'

image_url = test_path + '/JPCLN102.jpg'

img = np.array(Image.open(image_url))
img = cv2.resize(img, (224, 224))
img = np.float32(img) / 255
img = cv2.merge([img , img , img])
input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print(input_tensor)
input_tensor = input_tensor.to(device)


def test(net ,inp_data):
    correct = 0 
    total = 0
    test_loss = 0
    y_pred = []
    y_true = []
    
    net.eval()
    
    with torch.no_grad():
        
        outputs  = net(inp_data).to(device)
        pred_y = outputs[:,0]
        y_pred += pred_y.cpu().tolist()
            
    return y_pred

plt.title(test_data_list['age'][2])
plt.suptitle(test(model , input_tensor ))

# images = np.uint8(255 * input_tensor)
# im = Image.fromarray(images)
plt.imshow(img)
plt.savefig(os.getcwd() + '/outputs/' + 'ROC_Gender',dpi=300)
# target_layers = [model.features[-2].denselayer24]
# targets = [ClassifierOutputTarget(0)]

# with GradCAM(model=model, target_layers=target_layers , use_cuda= True) as cam:
#     grayscale_cams = cam(input_tensor=input_tensor, targets=targets )
#     cam_image = show_cam_on_image(img, grayscale_cams[0, :])

# cam = np.uint8(255*grayscale_cams[0, :])
# cam = cv2.merge([cam, cam, cam])
# images = np.hstack((np.uint8(255*img), cam , cam_image))
# im = Image.fromarray(images)
# im.save(os.getcwd() + '/outputs/grad-cam/test.png')