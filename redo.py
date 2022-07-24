import os
import pandas as pd 
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
# import tensorboard
import torch, torchvision, matplotlib, sklearn 
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TF
import cv2

class Resize_Pad():
  '''
  1. receive a tensor and resize longer sides into 128 
  2. maintain aspect ratio
  3. pad
  '''   
  def __call__(self, sample):
    longer, channel = max(sample.shape), np.argmax(sample.shape)
    # print(sample.shape)
    # print(f'longer: {longer} channel: {channel}')
    r = 128 / longer
    if channel == 1: 
      dim = (128, int(sample.shape[2] * r))
    else: 
      dim = (int(sample.shape[1] * r) ,128)
    sample = TF.Resize(dim)(sample)
    canvas = torch.zeros((3, 128, 128))
    if channel == 1: # long side is the LENGTH -> COLUMN SIDE
      canvas[:, :,:sample.shape[2]] = sample
    else:
      canvas[: ,:sample.shape[1],:] = sample
    return canvas
   


class RetrievalDataset(Dataset):

    def __init__(self, gt_csv, transform=None):
      self.gt_df = pd.read_csv(gt_csv)
      self.root_dir = os.getcwd()
      self.transform = transform

      self.gallery_path = os.path.join(self.root_dir, 'pa2_data/val/gallery/')

      self.query_path = os.path.join(self.root_dir, 'pa2_data/val/query/')

      self.gallery2idx = {}

      # storing the image path -> idx (we got idx from it filename)
      for gallery_image_path in os.listdir(self.gallery_path):
        self.gallery2idx[os.path.join(self.gallery_path, gallery_image_path)] = int(os.path.splitext(gallery_image_path)[0])

    def get_gallery_imgs(self):
      '''
      1. create a batch of images of size (N, C, H, W)
      2. enumerate the images in the gallery folder 
      '''
      batch = torch.zeros(len(os.listdir(self.gallery_path)), 3, 128, 128)
      for gallery_image_path in os.listdir(self.gallery_path):
        batch[int(os.path.splitext(gallery_image_path)[0])] = self.transform(Image.open(os.path.join(self.gallery_path, gallery_image_path)))
      return batch

    def __len__(self):
      return len(self.gt_df)

    def __getitem__(self, id):
      '''
      return query image and gt gallery image index
      1. given an id, u need to return image, gallery_idx 
      2. image = Image.open(df.iloc[id])
      3. gallery_idx got from a mapping(path => id (from enumerate))
      '''
      if id > self.gt_df.shape[0]:
        return None, None
      img = self.transform(Image.open(os.path.join(self.query_path,
      self.gt_df.iloc[id,0].split("/")[-1])))

      gallery_idx = self.gallery2idx[
      os.path.join(self.gallery_path, self.gt_df.iloc[id, 1].split("/")[-1])
      ]
      return img, gallery_idx

class ResBlock(nn.Module):
    def __init__(self,input_layer,  output_layer, stride):
        super(ResBlock, self).__init__()
        self.output_layer = output_layer
        self.input_layer = input_layer
        self.stride = stride
        
        self.conv1 = nn.Conv2d(in_channels=input_layer, out_channels=self.output_layer, kernel_size=3, stride=self.stride, bias = False, padding = 1)
        self.bn1 = nn.BatchNorm2d(output_layer)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=output_layer, out_channels=self.output_layer, kernel_size=3, stride=1, bias = False, padding = 1)
        self.bn2 = nn.BatchNorm2d(output_layer)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=input_layer, out_channels=self.output_layer, kernel_size=1, stride=self.stride, bias = False),
            nn.BatchNorm2d(output_layer)
        )
        
        self.relu3 = nn.ReLU()

    def forward(self, x):
        identity = x
        
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        if self.output_layer == self.input_layer:
            x += identity
        else:
            x += self.downsample(identity)
        
        x = self.relu3(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, bias = False, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        
        self.max_poll = self.pool = nn.MaxPool2d((3,3), stride = 2, padding = 1)
        
        #64 layer
        self.layer1 = nn.Sequential(
            ResBlock(64, 64, 1),
            ResBlock(64,64,1)
        )
        
        #128 layer
        self.layer2 = nn.Sequential(
            ResBlock(64,128,2),
            ResBlock(128,128,1)
        )
        
        #256 Layer
        self.layer3 = nn.Sequential(
            ResBlock(128,256,2),
            ResBlock(256,256,1)
        )
        
        #512 Layer
        self.layer4 = nn.Sequential(
            ResBlock(256,512,1),
            ResBlock(512,512,1)
        )
        
        #Average Pooling --> 512 * 1 * 1 ==> Should be conv to 512 * 1
        self.avg_pool = nn.AvgPool2d((8,8), stride=1, padding=0)

        #fc layer
        self.fc1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(256,230),
        )
        
        self.fc3_val = nn.Sequential(
            nn.Linear(256, 49),
        )
        
        self.fc3_triplet = nn.Sequential(
            nn.Linear(256,256)
        )

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.max_poll(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avg_pool(x)
        x = torch.squeeze(x,2)
        x = torch.squeeze(x,2)

        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    def forward_train(self, x):
        x = self.forward(x)
        x = self.fc3(x)
        return x
    
    def forward_validation_id(self, x):
        x = self.forward(x)
        x = self.fc3_val(x)
        return x
    
    def forward_triplet(self, x):
        x = self.forward(x)
        x = self.fc3_triplet(x)
        return x


def main():
  transform = TF.Compose([
    TF.ToTensor(), # for PIL aI
    TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    Resize_Pad(),
  ])

  retrieval_data = RetrievalDataset("pa2_data/val/gt.csv", transform)

  retrieval_dataloader = DataLoader(retrieval_data, batch_size=64, shuffle=False)

  # convert dataloader to iterable object 
  dataiter = iter(retrieval_dataloader)
  train_features, train_labels = dataiter.next()

if __name__ == "__main__":
  main()


