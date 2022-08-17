# -*- coding: utf-8 -*-

import os 
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.utils as utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

torch.cuda.empty_cache()

class converter(): 
    
    def __init__(self, IMG_SIZE,LABEL_DIR,LABEL_NAME):

        self.IMG_SIZE = IMG_SIZE
        self.LABEL_DIR = LABEL_DIR
        self.LABEL_NAME = LABEL_NAME
    
        self.training_data=[]
    
    def make_training_data(self):
        
        NUM_IMAGES = len(os.listdir(self.LABEL_DIR))
        
        for f in range(1, NUM_IMAGES+1):
            f = "{:02d}".format(f) + '_' + self.LABEL_NAME + '.png'
            path=os.path.join(self.LABEL_DIR,f)
            img=cv2.imread(path)
            img=cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
            self.training_data.append(np.array(img))

            
        np.save(f'{self.LABEL_NAME}.npy',self.training_data)
   

REBUILD_DATA=True
        
img_size=256

hazy_dir='./Datasets/Kaggle_Dense-Haze CVPR 2019/hazy/hazy' 
gt_dir='./Datasets/Kaggle_Dense-Haze CVPR 2019/GT/GT' 
    
if (REBUILD_DATA):
    convert=converter(img_size, gt_dir, 'GT')
    convert.make_training_data()
    
    convert=converter(img_size, hazy_dir, 'hazy')
    convert.make_training_data()


origin = np.load('GT.npy',allow_pickle=True)
hazy_im = np.load('hazy.npy',allow_pickle=True)

origin.shape, hazy_im.shape

for i in range(0,len(origin),5):
    
    fig=plt.figure(figsize=(15, 5))
    
    ax = plt.subplot(121)
    plt.imshow(origin[i])
    
    ax = plt.subplot(122)
    plt.imshow(hazy_im[i])
    plt.show()
    

patch_loader = torch.utils.data.DataLoader(dataset=origin,batch_size=1,shuffle=False)

for data in patch_loader:
    print(data.size())
    print(type(data))
    break

X_orig=torch.Tensor([origin[i] for i in range(len(origin))])
X_hazy=torch.Tensor([hazy_im[i] for i in range(len(hazy_im))])

X_orig=X_orig/255
X_hazy=X_hazy/255

print('X_orig: ',X_orig.size())

X_orig_T=np.transpose(X_orig,(0,3,1,2))
X_hazy_T=np.transpose(X_hazy,(0,3,1,2))
print('X_orig_T: ',X_orig_T.shape)

X_orig_flat=X_orig_T.reshape(-1,1,img_size,img_size)
X_hazy_flat=X_hazy_T.reshape(-1,1,img_size,img_size)
print('X_orig_flat: ',X_orig_flat.shape)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()  
        self.enc1 = nn.Conv2d(1, 256, 3, padding=1)
        self.enc2 = nn.Conv2d(256, 128, 3, padding=1)
        self.enc3 = nn.Conv2d(128, 64, 3, padding=1)
        self.enc4 = nn.Conv2d(64, 32, 3, padding=1)
        
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.dec2 = nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.dec3 = nn.ConvTranspose2d(64, 128, 2, stride=2)
        self.dec4 = nn.ConvTranspose2d(128, 256, 2, stride=2)
        self.out = nn.Conv2d(256,1, 3, padding=1)
        
        # batch normalization
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # encoder
        x = self.pool(self.bn1(F.relu(self.enc1(x))))
        x = self.pool(self.bn2(F.relu(self.enc2(x))))
        x = self.pool(self.bn3(F.relu(self.enc3(x))))
        x = self.pool(self.bn4(F.relu(self.enc4(x))))
        
        # decoder
        x = self.bn4(F.relu(self.dec1(x)))  
        x = self.bn3(F.relu(self.dec2(x)))
        x = self.bn2(F.relu(self.dec3(x)))
        x = self.bn1(F.relu(self.dec4(x)))
        x = torch.sigmoid(self.out(x))
        return x
    
    #     # conv layer (depth from 1 --> 32), 3x3 kernels
    #     self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  
    #     self.bn1 = nn.BatchNorm2d(32)
    #     # conv layer (depth from 32 --> 16), 3x3 kernels
    #     self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
    #     self.bn2 = nn.BatchNorm2d(16)
    #     # conv layer (depth from 16 --> 8), 3x3 kernels
    #     self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
    #     self.bn3 = nn.BatchNorm2d(8)
        
    #     # pooling layer to reduce x-y dims by two; kernel and stride of 2
    #     self.pool = nn.MaxPool2d(2, 2)
        
    #     ## decoder layers ##
    #     self.t_conv1 = nn.ConvTranspose2d(8, 8, 3, stride=2)
    #     self.t_bn1 = nn.BatchNorm2d(8)
    #     # two more transpose layers with a kernel of 2
    #     self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
    #     self.t_bn2 = nn.BatchNorm2d(16)
    #     self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
    #     self.t_bn3 = nn.BatchNorm2d(32)
    #     # one, final, normal conv layer to decrease the depth
    #     self.conv_out = nn.Conv2d(32, 1, 3, padding=1)


    # def forward(self, x):
    #     ## encode ##
    #     x = self.pool(self.bn1(F.relu(self.conv1(x))))
    #     x = self.pool(self.bn2(F.relu(self.conv2(x))))
    #     x = self.pool(self.bn3(F.relu(self.conv3(x))))
    #     # x = self.pool(self.bn4(F.relu(self.conv4(x))))
        
    #     ## decode ##
    #     x =self.t_bn1(F.relu(self.t_conv1(x)))
    #     x = self.t_bn2(F.relu(self.t_conv2(x)))
    #     x = self.t_bn3(F.relu(self.t_conv3(x)))
    #     # x = self.t_bn4(F.relu(self.t_conv4(x)))

    #     x = torch.sigmoid(self.conv_out(x))
        
    #     return x


model = ConvAutoencoder()
print(model)
model.to(device)

train_orig_loader = torch.utils.data.DataLoader(dataset=X_orig_flat,batch_size=1,shuffle=False)
train_hazy_loader = torch.utils.data.DataLoader(dataset=X_hazy_flat,batch_size=1,shuffle=False)

for train_orig, train_hazy in zip(train_orig_loader, train_hazy_loader):
    orig_image = Variable(train_orig).cuda()
    hazy_image = Variable(train_hazy).cuda()
    output = model(hazy_image)
    
    print('Image Dim: ',orig_image.size())
    print('Hazy Image Dim: ',hazy_image.size())
    print('Output Dim: ',output.size())
    break

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999))
n_epochs = 1000

for epoch in range(1,n_epochs+1):
    
    rand_idx=torch.randperm(X_orig.size()[0])
    X_orig_iter=X_orig[rand_idx]
    X_hazy_iter=X_hazy[rand_idx]

    X_orig_iter1=np.transpose(X_orig_iter,(0,3,1,2))
    X_hazy_iter1=np.transpose(X_hazy_iter,(0,3,1,2))

    X_orig_iter2=X_orig_iter1.reshape(-1,1,img_size,img_size)
    X_hazy_iter2=X_hazy_iter1.reshape(-1,1,img_size,img_size)

    train_orig_loader = torch.utils.data.DataLoader(dataset=X_orig_iter2,batch_size=1,shuffle=False)
    train_hazy_loader = torch.utils.data.DataLoader(dataset=X_hazy_iter2,batch_size=1,shuffle=False)

    for train_orig, train_hazy in zip(train_orig_loader, train_hazy_loader):
        orig_image = Variable(train_orig).cuda()
        hazy_image = Variable(train_hazy).cuda()
        
        optimizer.zero_grad()
        output = model(hazy_image)
        loss=criterion(output,orig_image)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch}/{n_epochs} :\tTraining Loss:{loss:.6f}') 
    
torch.save(model,'dehaze_autoencoder'+n_epochs.__str__()+'.pkl')
model = torch.load('dehaze_autoencoder'+n_epochs.__str__()+'.pkl')


dehazed_output=[]
for train_hazy in tqdm(train_hazy_loader):
    hazy_image = Variable(train_hazy).cuda()
    
    output = model(hazy_image)
    
    output=output.cpu()
    output=output.detach()
    dehazed_output.append(output)
    

X_dehazed=dehazed_output

X_dehazed=torch.stack(X_dehazed)
print(X_dehazed.size())

X_dehazed=X_dehazed.view(-1,1,256,256)
print(X_dehazed.size())

X_dehazed=X_dehazed.view(-1,3,256,256)
print(X_dehazed.size())

X_dehazed=X_dehazed.permute(0,2,3,1)
print(X_dehazed.shape)

for i in range(0,len(X_orig),10):
    fig=plt.figure(figsize=(15, 5))
    ax = plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(X_orig[i])
    
    ax = plt.subplot(132)
    plt.title('Hazy Image')
    plt.imshow(X_hazy[i])
    
    ax = plt.subplot(133)
    plt.title('Dehazed Image')
    plt.imshow(X_dehazed[i])
    plt.show()
