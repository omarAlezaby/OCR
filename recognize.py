import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import PIL
import time
import collections
import argparse
import statistics

from Utils.dataset import *
from Utils.logger import *
from Utils.utils import * 
from model import *

import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--use_cuda", type=bool, default=True, help="use gpu")
    parser.add_argument("--n_workers", type=int, default=4, help="number of workers to load data form disk")    
    parser.add_argument("--alphapet", type=str, default='!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~', help="alphapet chars")
    parser.add_argument("--checkpoint_path", type=str, default='checkpoints/best_checkpoint.pth', help="the path for the weights file")
    parser.add_argument("--fixed_seed", type=int, default=12, help="the seed for random functionalities")
    parser.add_argument("--imgs_folder", type=str, default='test_demo', help="imges folder path ")
    opt = parser.parse_args()
    print(opt)

    # model variables 
    inChannels = 1
    imgH = 32
    nHidden = 256
    nClasses = len(opt.alphapet)+1
    
    # checck for cuda devices
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() and opt.use_cuda else 'cpu')
    print(f'detect cuda device? {cuda}')

    alphapet = opt.alphapet
    print (f'alphapet is {alphapet}')
    print(f'number of classes id {len(alphapet)} + blank\n\n')

    # fix seeds
    random.seed(opt.fixed_seed)
    np.random.seed(opt.fixed_seed)
    torch.manual_seed(opt.fixed_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # validation dataloder
    val_dataset = Passwords_File(opt.imgs_folder)

    align = AlignBatch(recognize=True)
    imgs_dataloder = DataLoader(val_dataset, batch_size=opt.batch_size, collate_fn=align, 
                        shuffle= False, num_workers=opt.n_workers)


    # create the model
    crnn = CRNN(imgH, inChannels, nClasses, nHidden)
    crnn.apply(weights_init)
    # load checkpoint
    checkpoint = torch.load(opt.checkpoint_path)
    crnn.load_state_dict(checkpoint['state_dic'])

    # string label converter 
    converter = strLabelConverter(alphapet, ignore_case=False)


    # move to gpu 
    if cuda and opt.use_cuda :
        crnn = crnn.to(device)

    # eval mode
    crnn.eval()

    out_file = open(opt.imgs_folder + '/prediction.csv','w')
    writer = csv.writer(out_file)
    writer.writerow(['img', 'password'])

    interface_time = []

    for batch_i, (img_path, imgs) in enumerate(imgs_dataloder):
        
        tick = time.time()
        with torch.no_grad():

            # move to device and create variables
            imgs = Variable(imgs.to(device), requires_grad=False)

            # pass to the network
            preds = crnn(imgs)
            preds_size = Variable(torch.IntTensor([preds.shape[0]] * imgs.shape[0]))
            
            # get the nework prediction
            _, preds = preds.max(2)
            preds = preds.transpose(1,0).contiguous().view(-1)
            words_preds, lables_preds = converter.decode(preds.cpu(), preds_size)

            for img, password in zip(img_path, words_preds):
                print(img, password)
                writer.writerow([img, password])
        tock = time.time()

        interface_time.append(tock-tick)

    print(f'interface time for single image is {statistics.mean(interface_time)}')



    out_file.close()

