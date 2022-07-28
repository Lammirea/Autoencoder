# -*- coding: utf-8 -*-
"""
Задача научить автоэнкодер убирать задымленность с фотографии.
Для этого можно подбирать параметры сверточных слоев, возможно добавить еще слои.
Можно также попробовать изменить функцию потерь.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, utils

def im_plot(loader):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    print(images.shape)
    img = utils.make_grid(images, nrow = 2) 
    img = img.numpy().transpose((1, 2, 0)) 
    plt.imshow(img)
    

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# device = torch.device('cpu')
transformations = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(236),
    transforms.ToTensor(),
    ])

# transformations = transforms.ToTensor()

train_hazy_set = datasets.ImageFolder('.\Datasets\Kaggle_Dense-Haze CVPR 2019\hazy', transform = transformations)
val_hazy_set = datasets.ImageFolder('.\Datasets\Kaggle_Dense-Haze CVPR 2019\hazy_val', transform = transformations)

train_normal_set = datasets.ImageFolder('.\Datasets\Kaggle_Dense-Haze CVPR 2019\GT', transform = transformations)
val_normal_set = datasets.ImageFolder('.\Datasets\Kaggle_Dense-Haze CVPR 2019\GT_val', transform = transformations)

train_hazy_loader = torch.utils.data.DataLoader(train_hazy_set, batch_size=10, shuffle=False)
val_hazy_loader = torch.utils.data.DataLoader(val_hazy_set, batch_size =10, shuffle=False)

train_normal_loader = torch.utils.data.DataLoader(train_normal_set, batch_size=10, shuffle=False)
val_normal_loader = torch.utils.data.DataLoader(val_normal_set, batch_size =10, shuffle=False)

plt.figure()
im_plot(train_hazy_loader)
plt.figure()
im_plot(train_normal_loader)
plt.figure()
im_plot(val_hazy_loader)
plt.figure()
im_plot(val_normal_loader)

#эта штука делает градации фото на красный, зелёный и синие оттенки в rgb формате
# rgb_img = np.squeeze(images[3])
# channels = ['red channel', 'green channel', 'blue channel']

# fig = plt.figure(figsize = (36, 36)) 
# for idx in np.arange(rgb_img.shape[0]):
#     ax = fig.add_subplot(1, 3, idx + 1)
#     img = rgb_img[idx]
#     ax.imshow(img, cmap='gray')
#     ax.set_title(channels[idx])
#     width, height = img.shape
#     thresh = img.max()/2.5
#     for x in range(width):
#         for y in range(height):
#             val = round(img[x][y], 2) if img[x][y] !=0 else 0
#             ax.annotate(str(val), xy=(y,x),
#                     horizontalalignment='center',
#                     verticalalignment='center', size=8,
#                     color='white' if img[x][y]<thresh else 'black')
            
# Вся важная часть начинается с этого момента (при публикации надо удалить этот коммент!)
#надо найти denoising convAutoencoder и с его помощью решить эту задачу
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()  
    # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  
        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        # transpose layer, a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 3, stride=2)  # kernel_size=3 to get to a 7x7 image output
        # two more transpose layers with a kernel of 2
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(32, 3, 3, padding=1)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # transpose again, output should have a sigmoid applied
        x = F.sigmoid(self.conv_out(x))
        
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
criterion = nn.MSELoss()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

torch.cuda.empty_cache()

#Epochs
n_epochs = 20 #пока самый нормальный вариант, так как нет переобучения и фотографии получаются на выходе более чёткими

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0

    #Training
    for data_hazy, data_normal in zip(train_hazy_loader, train_normal_loader):
        images, _ = data_normal
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        gt_images, _ = data_hazy
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
output = output.detach().numpy()#тут ошибка появляется
img = utils.make_grid(output, nrow = 2) # метод делает сетку из картинок
img = img.cpu().numpy().transpose((1, 2, 0)) # транспонируем для отображения в картинке
plt.figure()
plt.imshow(img)

dataiter = iter(val_normal_loader)
images, _ = dataiter.next()
output = model(images.to(device))
output = output.detach().numpy()
img = utils.make_grid(output, nrow = 2) # метод делает сетку из картинок
img = img.cpu().numpy().transpose((1, 2, 0)) # транспонируем для отображения в картинке
plt.figure()
plt.imshow(img)

