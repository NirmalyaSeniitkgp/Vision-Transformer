import torch
import torch.nn as nn
import torch.optim as optim
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

# Types of Transformation in Train Data
transform_1 = transforms.ToTensor()
transform_2 = transforms.Normalize((0.5),(0.5))
transform_3 = transforms.RandomRotation(15)
transform = transforms.Compose([transform_1,transform_2,transform_3])

# Following are Hyper Parameters
img_size = 28
patch_size = 4
in_chans = 1
embed_dim = 16
num_heads = 8
hidden = 2048
num_layers = 4
num_classes = 10

learning_rate = 0.0001
batch_size = 512
num_epochs = 200

# Gerenation of Model
model = VisionTransformer(img_size,patch_size,in_chans,embed_dim,num_heads,hidden,num_layers,num_classes)

# Initiatization of Model
for params in model.parameters():
    if params.dim()>1:
        nn.init.xavier_uniform_(params)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Keep the Model in Training Mode
model.train()
model.to(device)

# MNIST Training Data Set and Train Data Loader
train_dataset = datasets.MNIST(root='/home/idrbt/Desktop/VIT_18_12_2024/Data', train = True,
        transform = transform, download = True)

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=4,drop_last=True)


# Training Loop
loss_each_epoch = []
for i in range(num_epochs):
    total_loss = 0
    for batch_number, (images, labels) in enumerate(train_loader):

        # print('******'*30)
        # print(images.dtype)
        # print(labels.dtype)
        # print('******'*30)

        images = images.to(device)
        labels = labels.to(device)

        # Removing the Previous Gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass and update
        loss.backward()
        optimizer.step()

        # Calculation of Total Loss
        total_loss = total_loss + loss.item()

        # print information
        print(f'epoch={i+1}, step={batch_number+1}, loss={loss.item():.3f}')

    # Append Loss of Each Epoch
    loss_each_epoch.append((total_loss)/(batch_number+1))


# Saving the Model
torch.save(model.state_dict(),'/home/idrbt/Desktop/VIT_18_12_2024/model_layers_4_batch_512_epoch_200.pth')

# Saving the Loss values of Each Epoch in a File
f = open('loss_values.txt','wb')
pickle.dump(loss_each_epoch,f)
f.close()

