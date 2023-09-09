import torch
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torch.utils.data import Dataset ,  DataLoader
import torchvision.transforms as transforms
import pandas as pd
import torch.nn as nn
from torch import optim
import torchvision.models
from adabelief_pytorch import AdaBelief
import copy
from sklearn.metrics import classification_report

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math 

def get_data(path , data_list ):
    
    files = os.listdir(path)

    images = []
    targets = []
    data = pd.DataFrame()
    # images += [path + '/' + each_image for each_image in files]
    # targets += [ data_list.loc[data_list['filenames'] == images[i].split('/')[-1] , 'age'].values.tolist() for i in range(len(images)) ]
    # for index, row in data_list.iterrows():
    #     data['images'] = path + '/' + row['filenames']
    #     data['age'] = row['age']
    # print(targets)
    
    # data['images'] = images
    # data['age'] = targets
    data_list['filenames'] = data_list['filenames'].apply(lambda x : path + '/' + x)
    # print(data)
    return data_list 


def read_image_file(path)-> torch.Tensor:
        image_locations_tensor = []
        
        for index , rows in path.iterrows():
            img = Image.open(rows.values[0])
            image_locations_tensor += [transform(img)]
            
        return torch.stack(image_locations_tensor)
        
def read_label_file(target)-> torch.Tensor:
        labels = target.values.tolist()
        labels_tensor = torch.tensor(labels)
        return labels_tensor

class InputData(Dataset):
    def __init__(self, path , target , transform  = None):
        self.path  = path
        self.targets_columns = target
        self.transform = transform
        self.data , self.targets = self._load_data(self.path , self.targets_columns)

    
    def _load_data(self , path , target):
        image_file = path
        data = read_image_file(path)

        label_file = target
        targets = read_label_file(target)
        
        return data , targets
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.path.iloc[idx,0]
        img = Image.open(image)
        
        if self.transform:
            data = self.transform(img)
        
        target = int(self.targets_columns.iloc[idx,0])
            
        return (data , target)
    
def train(net , train_data_loader , criterion , optimizer , num_epochs = 3):
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = 99999999

    net.train()
    for iteration in range(num_epochs):
        y_predicted = []
        y_true = []
        for i , data in enumerate(train_data_loader):
            
            inp_data , labels = data
            inp_data , labels = inp_data.to(device) , labels.to(device)
     
            outputs = net(inp_data)
            loss_val = criterion(outputs , labels)

            optimizer.zero_grad()

            loss_val.backward()

            optimizer.step() 

            if (i + 1) % 2 == 0:
                print(f" Epoch = [{iteration + 1} / {num_epochs}] Step = [{i + 1} / {len(train_data_loader)}]")
            
            net.eval()
            # print(net(inp_data)[:,0].cpu().tolist())
            y_predicted += net(inp_data)[:,0].cpu().tolist()
            y_true += labels.cpu().tolist()
        # print(len(y_predicted) , len(y_true))
        if mean_squared_error(y_true, y_predicted) < best_loss:
            best_loss = loss_val.item()
            best_model_wts = copy.deepcopy(net.state_dict())

    return best_model_wts

def test(net ,test_data_loader , criterion):
    correct = 0 
    total = 0
    test_loss = 0
    y_pred = []
    y_true = []
    
    net.eval()
    
    with torch.no_grad():
        for data in test_data_loader:
            inp_data , labels = data
            inp_data , labels = inp_data.to(device) , labels.to(device)
            
            outputs  = net(inp_data)
            
            loss_val = criterion(outputs , labels)
            
            test_loss += loss_val.item()
            
            pred_y = outputs[:,0]
            total += labels.size(0)
            
            y_pred += pred_y.cpu().tolist()
            y_true += labels.cpu().tolist()
            
            # correct += (pred_y == labels).sum().item()
            
    # accuracy = correct / total
    # print(f"Test Accuracy of the model is : {accuracy * 100 : .2f} %")
    print(f"Overall test loss of the model is : {test_loss} ")
    return y_true, y_pred

class L1LossFlat(nn.SmoothL1Loss):
    def forward(self, input:torch.Tensor, target:torch.Tensor) -> float:
        return super().forward(input.view(-1), target.view(-1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_list = pd.read_csv(os.getcwd() + '/datasets/XPAge01_RGB/XP/trainingdata.csv')
test_data_list = pd.read_csv(os.getcwd() + '/datasets/XPAge01_RGB/XP/testdata.csv')

# data_list = pd.concat([train_data_list ,test_data_list])
# print(data_list)

train_path = os.getcwd() + '/datasets/XPAge01_RGB/XP/JPGs'
test_path = os.getcwd() + '/datasets/XPAge01_RGB/XP/JPGs'

# print(train_path)
# print(test_path)

train_data_set = get_data(train_path , train_data_list)
test_data_set = get_data(test_path , test_data_list)

# print(train_images)
# data_set.drop(data_set[data_set['age'] >= 24000].index, inplace = True)
# data_set['age'] = data_set['age'].apply(lambda x : x[0] if len(x) > 0 else None)
# print(data_set)


transform =  transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.Grayscale(num_output_channels=3),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
])

train_data  = InputData(train_data_set[['filenames']] , train_data_set[['age']] , transform)
test_data  = InputData(test_data_set[['filenames']] , test_data_set[['age']] , transform)

print(train_data.data.shape)
print(train_data.targets.shape)
print(test_data.targets.shape)
print(test_data.data.shape)

train_data_loader = DataLoader(train_data , batch_size = 5 , shuffle = True , num_workers = 1)
test_data_loader = DataLoader(test_data , batch_size = 5 , shuffle = True , num_workers = 1)


# # resnet = torchvision.models.resnet18()
# # resnet.fc = torch.nn.Linear(resnet.fc.in_features, 2)

# # resnet = nn.DataParallel(resnet)
# # resnet.to(device)
# # # resnet.conv1 =  torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# # criterion = nn.CrossEntropyLoss()
# # optimizer = optim.Adam(resnet.parameters() , lr = 0.03)
class AgeModel(nn.Module):
   def __init__(self):
        super().__init__()
        layers = list(torchvision.models.resnet34(pretrained=True).children())[:-2]
        layers += [nn.AvgPool2d(), Flatten()]
        layers += [nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        layers += [nn.Dropout(p=0.50)]
        layers += [nn.Linear(1024, 512, bias=True), nn.ReLU(inplace=True)]
        layers += [nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        layers += [nn.Dropout(p=0.50)]
        layers += [nn.Linear(512, 16, bias=True), nn.ReLU(inplace=True)]
        layers += [nn.Linear(16,1)]
        self.agemodel = nn.Sequential(*layers)
    
   def forward(self, x):
        return self.agemodel(x).squeeze(-1)
   
weights = torchvision.models.ResNet152_Weights.DEFAULT
model = torchvision.models.resnet152(weights = weights)

# preprocess = weights.transforms()

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs , 1)
model = model.to(device)
criterion = L1LossFlat()
optimizer = AdaBelief(model.parameters(), lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = True , rectify = False)
# optimizer = optim.Adam(model.parameters() , lr = 0.01)


best_model = train(model , train_data_loader , criterion , optimizer , 100)
model.load_state_dict(best_model)

y_true, y_pred = test(model , test_data_loader , criterion )
print(y_true[0:5])
print(y_pred[0:5])

print(mean_absolute_error(y_true, y_pred))
print(mean_squared_error(y_true, y_pred))
print(math.sqrt(mean_squared_error(y_true, y_pred)))
print(r2_score(y_true, y_pred))

# print(classification_report(y_true , y_pred , labels = range(0, 2)))
    
# model =


# model= nn.DataParallel(model)
# model.to(device)
# print(device)