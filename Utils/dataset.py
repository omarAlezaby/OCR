from torch.utils.data import Dataset, sampler, DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
from PIL import Image
import random

class Passwords_data(Dataset):
    def __init__ (self, csv_Path, imgs_path, transformers = None):
        with open(csv_Path, 'r') as csv_file:
            # images names and labels in matching indices
            reader = csv.reader(csv_file)
            self.imgs = []
            self.lables = []
            for i, row in enumerate(reader):
                if i == 0: continue
                self.imgs.append(row[0])
                self.lables.append(row[1])
                
            self.imgs_file = imgs_path
            self.transformers = transformers
    
    def __getitem__(self, index):
        # image
        img_path = self.imgs_file + '/' + self.imgs[index]
        img = Image.open(img_path).convert('L')
        #print(img.size)
        
        # image augmentation
        if self.transformers != None:
            img = self.transformers(img)
        img = transforms.ToTensor()(img)
        # lable
        lable = self.lables[index]
        
        return (img_path, img, lable)
    
    def __len__(self):
        return len(self.imgs)


# resize and normalize grayscale image
class ResizeNormalize (object):
    def __init__(self, img_size):
        self.img_size = img_size # imgH, imgW
    
    def __call__(self, img):
        img = transforms.Resize(size=self.img_size)(transforms.ToPILImage()(img))
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5) # normalize gray scale
        return img


# adjust patch images to have same shape
class AlignBatch(object):
    def __init__(self, imgH = 32, imgW = 100, keep_ratio = True, min_ratio = 1, padding = False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.padding = padding
        
    def __call__(self, batch):
        img_paths, imgs, lables = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        
        if(self.keep_ratio):
            max_ratio = 0
            for img in imgs:
                #print(img.shape)
                _, h, w = img.shape
                if max_ratio < (w/h): max_ratio = (w/h)
            if self.padding:
                imgs_padded = []
                for i, img in enumerate(imgs):
                    _, h, w = img.shape
                    w_new = h * max_ratio
                    pad = int((w_new - w) / 2)
                    #print(img.shape)
                    imgs_padded.append(F.pad(img, (pad, pad), "constant"))
                imgs = imgs_padded
            
            imgW = int(max_ratio * imgH)
            
        resizer = ResizeNormalize((imgH, imgW))
        imgs = [resizer(img).unsqueeze(0) for img in imgs]
        imgs = torch.cat(imgs, 0)
        return img_paths, imgs, lables
                    

# dataloder sampler to compine images with the same shape
class MatchingSampler(sampler.Sampler):
    
    def __init__ (self, data_source, batch_size):
        self.data_source = data_source
        self.number_sampels = len(data_source)
        self.batch_size = batch_size
        
    def __iter__ (self):
        n_batches = len(self)//self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batches):
            random_start = random.randint(0, len(self)-self.batch_size)
            b_indx = random_start + torch.arange(0, self.batch_size)
            index[i*self.batch_size : (i+1) * self.batch_size] = b_indx
        if tail:
            random_start = random.randint(0, len(self)-tail)
            b_indx = random_start + torch.arange(0, tail-1)
            index[n_batches * self.batch_size:] = b_indx
            
        return iter(index)
        
    def __len__(self):
        return self.number_sampels