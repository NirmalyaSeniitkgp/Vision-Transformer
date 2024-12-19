import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
import pickle

import Vision_Transformer
from Vision_Transformer import VisionTransformer

# Selecting the Device
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_device()

# Types of Transformation in Test Data
transform_1 = transforms.ToTensor()
transform_2 = transforms.Normalize((0.5),(0.5))

transform = transforms.Compose([transform_1,transform_2])

# Following are Hyper Parameters
img_size = 28
patch_size = 4
in_chans = 1
embed_dim = 16
num_heads = 8
hidden = 2048
num_layers = 4
num_classes = 10

batch_size = 20

# Gerenation of Model
model = VisionTransformer(img_size,patch_size,in_chans,embed_dim,num_heads,hidden,num_layers,num_classes)
model.to(device)

# Inserting the State Dict to the Model
model.load_state_dict(torch.load('/home/idrbt/Desktop/VIT_18_12_2024/model_layers_4_batch_512_epoch_200.pth',map_location=torch.device('cuda')))

# Keep Model in the Evaluation Mode
model.eval()

# MNIST Test Data Set and Test Data Loader
test_dataset = datasets.MNIST(root='/home/idrbt/Desktop/VIT_18_12_2024/Data', train = False,
        transform = transform, download = True)

test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size)

# Testing Loop
with torch.no_grad():
    number_correct = 0
    number_samples = 0
    for images, labels in test_loader:

        images =images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Calcutating the Predicted Index
        predictions = torch.argmax(outputs, 1)

        # Calculating the Total Number of Images and
        # Total Number of Images Predicted Correctly by the Model
        number_samples +=labels.shape[0]
        number_correct +=(predictions==labels).sum().item()
    
    # Calculation of Accuracy of Model
    acc = (number_correct/number_samples)*100.0
    print(f'accuracy={acc}')


