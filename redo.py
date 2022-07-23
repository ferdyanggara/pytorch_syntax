import os
from re import L
import pandas as pd 
import numpy as np
from PIL import Image
# import tensorboard
import torch, torchvision, matplotlib, sklearn 
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
   

transform = TF.Compose([
  TF.ToTensor(), # for PIL aI
  TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  Resize_Pad(),
])

class RetrievalDataset(Dataset):

    def __init__(self, csv_file, root_dir='pa2_data/imgs/train/', transform=None):
      """
      Args:
          csv_file (string): Path to the csv file with annotations.
          root_dir (string): Directory with all the images.
          transform (callable, optional): Optional transform to be applied
              on a sample.
      """
      self.landmarks_frame = pd.read_csv(csv_file)
      self.root_dir = root_dir
      self.transform = transform

    def get_gallery_imgs():
      pass
    def getitem():
      pass

    def __len__(self):
      pass
      # return len(self.)

    def __getitem__(self, idx):
      img_name = os.path.join(self.root_dir,
                              self.landmarks_frame.iloc[idx, 0])
      image = Image.open(img_name)

      if self.transform:
          sample = self.transform(sample)

      return sample