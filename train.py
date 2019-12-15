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
import os

from utils.dataset import *
from utils.logger import *
from utils.utils import * 
from model import *


# train the model for one epoch
def train(model, criterion, optimizer, logger, train_dataloder, batch_size, epoch_num):
    
    model.train()

    epoch_loss = 0
    samples_num = 0
    
    for batch_i, (_, imgs, targets) in enumerate(train_dataloder):
        batches_done = len(train_dataloder) * epoch_num + batch_i
        samples_num += imgs.shape[0]
        
        # move to device and create variables
        imgs = Variable(imgs.to(device))
        targets, lenghts = converter.encode(targets)
        targets = Variable(targets.to(device), requires_grad=False)
        t_lens = Variable(lenghts, requires_grad=False)
        
        # pass to the network
        preds = model(imgs)
        preds_size = Variable(torch.IntTensor([preds.shape[0]] * imgs.shape[0]))
        
        # loss
        #print(preds_size.shape)
        loss = criterion(preds, targets.cpu(), preds_size, t_lens)
        epoch_loss += loss * imgs.shape[0]
        logger.scalar_summary('loss_batches', loss, batches_done)
        print(f'Epoch {epoch_num}, Batch {batch_i}/{len(train_dataloder)} : Loss = {loss}')
        
        # optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # compute epoch loss
    epoch_loss /= samples_num
    logger.scalar_summary('loss_epochs', epoch_loss, epoch_num)
    
    return epoch_loss


# compute model accuracy
def val(model, criterion, logger, val_dataloder, epoch_num, batch_size=16, test_display=4, log_name='val'):
    
    model.eval()
    
    nCorrect_words = 0
    val_loss = 0
    samples_num = 0
    
    for batch_i, (_, imgs, targets) in enumerate(val_dataloder):
        samples_num += imgs.shape[0]
        
        # move to device and create variables
        imgs = Variable(imgs.to(device), requires_grad=False)
        targets_encoded, lenghts = converter.encode(targets)
        targets_encoded = Variable(targets_encoded.to(device), requires_grad=False)
        t_lens = Variable(lenghts, requires_grad=False)
        
        with torch.no_grad():
            # pass to the network
            preds = model(imgs)
            preds_size = Variable(torch.IntTensor([preds.shape[0]] * imgs.shape[0]))

            # loss
            loss = criterion(preds, targets_encoded.cpu(), preds_size, t_lens)
            val_loss += loss * imgs.shape[0]
            
            # get the nework prediction
            _, preds = preds.max(2)
            preds = preds.transpose(1,0).contiguous().view(-1)
            words_preds, lables_preds = converter.decode(preds, preds_size)
            
            for word_pred, target in zip(words_preds, targets):
                if word_pred == target:
                    nCorrect_words += 1
    
    # display some of the network prediction
    row_preds, _ = converter.decode(preds, preds_size, raw=True)[:test_display]

    for row_pred, word_pred, gt in zip(row_preds, words_preds, targets):
        print(f'{row_pred} => {word_pred}, Ground Truth is {gt}')
    
    #compute loss and accurcy
    word_accurcy = nCorrect_words / samples_num
    val_loss /= samples_num
    logger.scalar_summary(log_name + '_loss', val_loss, epoch_num)
    logger.scalar_summary(log_name + '_WordAccurcy', word_accurcy, epoch_num)
    
    return val_loss, word_accurcy



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_num", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--use_cuda", type=bool, default=True, help="use gpu")
    parser.add_argument("--n_workers", type=int, default=4, help="number of workers to load data form disk")
    parser.add_argument("--lr", type=int, default=.001, help="learning rate value")
    parser.add_argument("--test_display", type=int, default=4, help="number of examples outputs to display")
    parser.add_argument("--val_each", type=int, default=1, help="number of epoch to validate after")
    parser.add_argument("--weights_file", type=str, default='weights/crnn.pth', help="the path for the weights file")
    parser.add_argument("--fixed_seed", type=int, default=12, help="the seed for random functionalities")
    parser.add_argument("--imgs_folder", type=str, default='data/train_passwordv3', help="imges folder path ")
    parser.add_argument("--train_file", type=str, default='data/password_train.csv', help="training csv file path")
    parser.add_argument("--val_file", type=str, default='data/password_val.csv', help="val csv file path")
    opt = parser.parse_args()
    print(opt)

    # model variables 
    inChannels = 1
    imgH = 32
    nHidden = 256

    # create used files
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # intialize logger file connection
    logger = Logger('logs')

    # check for cuda devices 
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() and opt.use_cuda else 'cpu')
    print(f'detect cuda device? {cuda}\n\n')

    # extract alphabet from training files
    with open(opt.train_file) as csv_file:
        reader = csv.reader(csv_file)
        char_ststistics = {}
        for i, row in enumerate(reader):
            if(i == 0): continue
            for c in row[1]:
                if c in char_ststistics: char_ststistics[c] += 1
                else: char_ststistics[c] = 1

    alphabet = ''
    for c in sorted(char_ststistics):
        alphabet += c

    nClasses = len(alphabet) + 1
    print (f'alphapet is {alphabet}')
    print(f'number of classes id {nClasses-1} + blank\n\n')

    # fix seeds for detemastic accuracies
    random.seed(opt.fixed_seed)
    np.random.seed(opt.fixed_seed)
    torch.manual_seed(opt.fixed_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # train Augmentation
    train_transforms = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.RandomAffine(degrees=0, translate=(.03,.03)),
                transforms.RandomAffine(degrees=0, scale=(.95,1.05)),
                transforms.RandomAffine(degrees=0, shear=20),
                transforms.RandomRotation(degrees=3, expand=True)]),
            transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.3)],  p=0.5)])

    # train dataloder
    train_dataset = Passwords_data(opt.train_file, opt.imgs_folder, transformers=train_transforms)

    align = AlignBatch()
    train_dataloder = DataLoader(train_dataset, batch_size=opt.batch_size, collate_fn=align, 
                        shuffle= True, num_workers=opt.n_workers)

    # dataloader to compute train accurcy
    train_val_dataset = Passwords_data(opt.train_file, opt.imgs_folder)

    align = AlignBatch()
    train_val_dataloder = DataLoader(train_val_dataset, batch_size=opt.batch_size, collate_fn=align, 
                        shuffle= False, num_workers=opt.n_workers)

    # validation dataloder
    val_dataset = Passwords_data(opt.val_file, opt.imgs_folder)

    align = AlignBatch()
    val_dataloder = DataLoader(val_dataset, batch_size=opt.batch_size, collate_fn=align, 
                        shuffle= False, num_workers=opt.n_workers)

    # create model
    crnn = CRNN(imgH, inChannels, nClasses, nHidden)
    crnn.apply(weights_init)

    # use pretrained weights
    if opt.weights_file.find('checkpoints/') == -1 :
        # use pretrained weights from english alphabet model
        model_dict = crnn.state_dict() # state of the current model
        pretrained_dict = torch.load(opt.weights_file) # state of the pretrained model
        # remove the classifier from the state
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k != 'rnn.1.embedding.weight' and k != 'rnn.1.embedding.bias'}
        # get the classifier weight from new model
        classifier_dict = {k: v for k, v in model_dict.items() if k == 'rnn.1.embedding.weight' or k == 'rnn.1.embedding.bias'} 
        pretrained_dict.update(classifier_dict) # update without classifier
        crnn.load_state_dict(pretrained_dict)

    else:
        # use previous checkpoint
        checkpoint = torch.load(opt.weights_file)
        crnn.load_state_dict(checkpoint['state_dic'])

    # display model structure
    print(crnn)

    # string label converter 
    converter = strLabelConverter(alphabet, ignore_case=False)

    # loss function 
    criterion = nn.CTCLoss()

    # optimizer
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr)

    # move to gpu 
    if cuda and opt.use_cuda:
        crnn = crnn.to(device)
        criterion = criterion.to(device)

    # start training
    best_acc = 0
    for epoch in range(opt.epochs_num):
        #train
        tick = time.time()
        train_loss = train(crnn, criterion, optimizer, logger, 
                        train_dataloder, opt.batch_size, epoch)
        tock = time.time()
        
        print(f'Epoch {epoch} finished in {(tock - tick) / 60} minutes')
        print(f'Epoch {epoch} training_loss = {train_loss}')

        # compute training accuracy
        train_acc = val(crnn, criterion, logger, train_val_dataloder,
                        epoch, opt.batch_size, opt.test_display, log_name='train_val')
        
        # evaluate
        if epoch % opt.val_each == 0:
            val_loss, val_accurcy = val(crnn, criterion, logger, val_dataloder,
                                        epoch, opt.batch_size, opt.test_display)
            print(f'Epoch {epoch} val_loss = {val_loss}, word_accuracy = {val_accurcy}')
            
            # save best checkpoint
            if best_acc <= val_accurcy:
                best_acc = val_accurcy
                checkpoint = {'input_hight':32,
                            'output_size':len(alphabet)+1,
                            'alphapet':alphabet,
                            'train_transforms':train_transforms,
                            'optim_dic':optimizer.state_dict(),
                            'state_dic':crnn.state_dict(),
                            'epoch':epoch
                            }
                torch.save(checkpoint,'checkpoints/best_checkpoint.pth')

        # save last epoch
        checkpoint = {'input_hight':32,
                    'output_size':len(alphabet)+1,
                    'alphapet':alphabet,
                    'train_transforms':train_transforms,
                    'optim_dic':optimizer.state_dict(),
                    'state_dic':crnn.state_dict(),
                    'epoch':epoch
                    }
        torch.save(checkpoint,'checkpoints/last_checkpoint.pth')
        
    print(f'the best accurcay is {best_acc}')
