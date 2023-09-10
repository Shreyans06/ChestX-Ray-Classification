import torch
import os
import matplotlib.pyplot as plt
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
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def get_data(path , type ="train"):
    
    directories = os.listdir(path)

    images = []
    targets = []
    class_map = {}
    for idx , x in enumerate(directories):
            if type == "train":
                class_map[x] = idx
            image_list = os.listdir(path + '/' + x)
            images += [path + '/' + x + '/' + each_image for each_image in image_list]
            targets += [idx for _ in  image_list]
    if type == "test":
         return images , targets
        
    return images , targets , class_map


def read_image_file(path)-> torch.Tensor:
        image_locations_tensor = []
        
        for index , rows in path.iterrows():
            img = Image.open(rows.values[0])
            image_locations_tensor += [transform(img)]
            
        return torch.stack(image_locations_tensor)
        
def read_label_file(target)-> torch.Tensor:
        labels = target[0]
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
        for i , data in enumerate(train_data_loader):
            
            inp_data , labels = data
            inp_data , labels = inp_data.to(device) , labels.to(device)
     
            outputs = net(inp_data)
            loss_val = criterion(outputs , labels)

            optimizer.zero_grad()

            loss_val.backward()

            optimizer.step() 

            if (i + 1) % 2 == 0:
                print(f" Epoch = [{iteration + 1} / {num_epochs}] Step = [{i + 1} / {len(train_data_loader)}] Loss = {loss_val.item()} ")
        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            best_model_wts = copy.deepcopy(net.state_dict())

    return best_model_wts

def test(net ,test_data_loader , criterion):
    correct = 0 
    total = 0
    test_loss = 0
    y_pred = []
    y_true = []
    y_pred_probs = []
    
    net.eval()
    
    with torch.no_grad():
        for data in test_data_loader:
            inp_data , labels = data
            inp_data , labels = inp_data.to(device) , labels.to(device)
            
            outputs  = net(inp_data)
            
            loss_val = criterion(outputs , labels)
            
            test_loss += loss_val.item()
            

            sm = torch.nn.Softmax()
            probabilities = sm(outputs).cpu() 
        
            y_pred_probs += probabilities[:,1].tolist()

            pred_y = torch.max(outputs , 1)[1].data.squeeze()
            total += labels.size(0)
            
            y_pred += pred_y.cpu().tolist()
            y_true += labels.cpu().tolist()
            
            correct += (pred_y == labels).sum().item()
            
    accuracy = correct / total
    print(f"Test Accuracy of the model is : {accuracy * 100 : .2f} %")
    print(f"Overall test loss of the model is : {test_loss} ")
    return y_true, y_pred , y_pred_probs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = os.getcwd() + '/datasets/Gender01/train'
test_path = os.getcwd() + '/datasets/Gender01/test'

train_images , train_targets , class_map = get_data(train_path)
test_images , test_targets = get_data(test_path , "test")

transform = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.Grayscale(num_output_channels=3),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
])
train_data  = InputData(pd.DataFrame(train_images) , pd.DataFrame(train_targets) , transform)
test_data  = InputData(pd.DataFrame(test_images) , pd.DataFrame(test_targets) , transform)


train_data_loader = DataLoader(train_data , batch_size = 12 , shuffle = True , num_workers = 1)
test_data_loader = DataLoader(test_data , batch_size = 12 , shuffle = True , num_workers = 1)


model = torchvision.models.densenet161()

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs , 2)
model = model.to(device)
criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(model.parameters() , lr = 0.01)


best_model = train(model , train_data_loader , criterion , optimizer , 100)
model.load_state_dict(best_model)

torch.save(model , os.getcwd() + "/outputs/models/gender")

y_true, y_pred , y_pred_probs= test(model , train_data_loader , criterion )

print(classification_report(y_true , y_pred , labels = range(0, 2)))


ns_probs = [0 for _ in range(len(y_true))]
ns_auc_score = roc_auc_score(y_true, ns_probs)
dn_auc_score = roc_auc_score(y_true , y_pred_probs)

print('Random: ROC AUC=%.3f' % (ns_auc_score))
print('DenseNet: ROC AUC=%.3f' % (dn_auc_score))

ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_true, y_pred_probs)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random')
plt.plot(lr_fpr, lr_tpr, linestyle='-', label='DenseNet')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# show the legend
plt.legend()

plt.savefig(os.getcwd() + '/outputs/ROCs/' + 'ROC_Gender',dpi=300)

