from torch.utils.data import Dataset, sampler, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
from PIL import Image
import random

class BiDrictionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BiDrictionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
    
    def forward(self, input):
        recurrent, _ = self.rnn(input)
        t_steps, b_size, h_num = recurrent.shape
        recurrent = recurrent.view(t_steps*b_size, h_num) # prepare the linear layer input
        
        output = self.embedding(recurrent)
        output = output.view(t_steps, b_size, -1)
        
        return output


class CRNN(nn.Module):
    def __init__(self, imgH, inChannal, nClasses, nHidden, nLSTMs = 2):
        super(CRNN, self).__init__()
        assert imgH == 32 , 'the image input hight must be 32'
        
        ks = [3, 3, 3, 3, 3, 3, 2] # kernal Size
        ps = [1, 1, 1, 1, 1, 1, 0] # padding
        ss = [1, 1, 1, 1, 1, 1, 1] # stride
        fn = [64, 128, 256, 256, 512, 512, 512] # filters number
        
        cnn = nn.Sequential()
        
        
        def conv_layer(layNum, b_n = False):
            nIn = inChannal if layNum == 0 else fn[layNum-1]
            nOut = fn[layNum]
            # Conv Layer
            cnn.add_module(f'conv{layNum}', nn.Conv2d(nIn, nOut, ks[layNum], ss[layNum], ps[layNum]))
            # btach normalization
            if b_n:
                cnn.add_module(f'batchnorm{layNum}', nn.BatchNorm2d(nOut))
            # non Linearity (ReLU)
            cnn.add_module(f'relu{layNum}', nn.ReLU(inplace=True))
            
        # Cnn Arch
        conv_layer(0)
        cnn.add_module(f'pooling{0}', nn.MaxPool2d(2, 2))  # 64 x 16
        conv_layer(1)
        cnn.add_module(f'pooling{1}', nn.MaxPool2d(2, 2))  # 128 x 8
        conv_layer(2, b_n=True)
        conv_layer(3)
        # the irregular shape of stride and padding beacause of the shape of some char like (i, ..)
        cnn.add_module(f'pooling{2}', nn.MaxPool2d((2, 2), (2, 1), (0, 1))) # 256 x 4
        conv_layer(4, b_n=True)
        conv_layer(5)
        cnn.add_module(f'pooling{3}', nn.MaxPool2d((2, 2), (2, 1), (0, 1))) # 512 x 2
        conv_layer(6, b_n=True) #  512 x 1

        self.cnn = cnn

        # Rnn Arch
        rnn = nn.Sequential(
            BiDrictionalLSTM(512, nHidden, nHidden),
            BiDrictionalLSTM(nHidden, nHidden, nClasses)
        )

        self.rnn = rnn
        
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input):
        # cnn pass
        conv = self.cnn(input)
        b, c, h, w = conv.shape
        
        assert h == 1, 'the hight after cnn must equal 1'
        
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1) # sequance, batch, features
        
        # rnn pass 
        rnn = self.rnn(conv)
        
        output = self.softmax(rnn)
        
        return output
