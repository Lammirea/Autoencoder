# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:48:07 2022

@author: ПользовательHP
"""

"""
Задача научить автоэнкодер убирать задымленность с фотографии.
Для этого можно подбирать параметры сверточных слоев, возможно добавить еще слои.
Можно также попробовать изменить функцию потерь.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, utils

def im_plot(loader):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    print(images.shape)
    img = utils.make_grid(images, nrow =3) 
    img = img.numpy().transpose((1, 2, 0)) 
    plt.imshow(img)


if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


transformations = transforms.Compose([
    transforms.Resize(235),
    transforms.CenterCrop(230),
    transforms.ToTensor(),
    ])

train_hazy_set = datasets.ImageFolder('.\Datasets\Kaggle_Dense-Haze CVPR 2019\hazy', transform = transformations)
val_hazy_set = datasets.ImageFolder('.\Datasets\Kaggle_Dense-Haze CVPR 2019\hazy_val', transform = transformations)

train_normal_set = datasets.ImageFolder('.\Datasets\Kaggle_Dense-Haze CVPR 2019\GT', transform = transformations)
val_normal_set = datasets.ImageFolder('.\Datasets\Kaggle_Dense-Haze CVPR 2019\GT_val', transform = transformations)

train_hazy_loader = torch.utils.data.DataLoader(train_hazy_set, batch_size=15, shuffle=False)
val_hazy_loader = torch.utils.data.DataLoader(val_hazy_set, batch_size =15, shuffle=False)

train_normal_loader = torch.utils.data.DataLoader(train_normal_set, batch_size=15, shuffle=False)
val_normal_loader = torch.utils.data.DataLoader(val_normal_set, batch_size =15, shuffle=False)

plt.figure()
im_plot(train_hazy_loader)
plt.figure()
im_plot(train_normal_loader)
plt.figure()
im_plot(val_hazy_loader)
plt.figure()
im_plot(val_normal_loader)


# Вся важная часть начинается с этого момента (при публикации надо удалить этот коммент!)
#надо найти denoising convAutoencoder и с его помощью решить эту задачу
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2,output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2,output_padding=0)
       

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.sigmoid(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
              
        return x


model = ConvAutoencoder()
print(model)
model.to(device)

# dataiter = iter(val_hazy_loader)
# images, _ = dataiter.next()
# outputs = model(images)

# print(images.size())
# print(outputs.size())

#Loss function
criterion = nn.MSELoss().cuda()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Epochs
n_epochs = 20 #пока самый нормальный вариант, так как нет переобучения и фотографии получаются на выходе более чёткими

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0

    #Training
    for data_hazy, data_normal in zip(train_hazy_loader, train_normal_loader):
        images, _ = data_hazy
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        gt_images, _ = data_normal
        #эту часть я добавила сама
        gt_images = gt_images.to(device)
        optimizer.zero_grad()
        outputs = model(gt_images)
        #-------------------------------------------------------
        loss = criterion(outputs, gt_images.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*images.size(0)
    
    train_loss = train_loss/len(train_hazy_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


#test images
dataiter = iter(val_hazy_loader)
images, _ = dataiter.next()
output = model(images.to(device))
img = utils.make_grid(output, nrow = 3) # метод делает сетку из картинок
img = img.cpu().numpy().transpose((1, 2, 0)) # транспонируем для отображения в картинке
plt.figure()
plt.imshow(img)

dataiter = iter(val_normal_loader)
images, _ = dataiter.next()
output = model(images.to(device))
img = utils.make_grid(output, nrow = 3) # метод делает сетку из картинок
img = img.cpu().numpy().transpose((1, 2, 0)) # транспонируем для отображения в картинке
plt.figure()
plt.imshow(img)

