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
import sklearn.metrics as metrics
import argparse
import os


from utils.dataset import *
from utils.logger import *
from utils.utils import * 
from model import *



# computer accuracy and F1 score
def evaluate(model, criterion, val_dataloder, test_display=4):
    model.eval()
    
    nCorrect_words = 0
    val_loss = 0
    samples_num = 0
    y_pred = []
    y_targets = []
    worng_samples = []
    worng_preds = []
    
    for batch_i, (img_path, imgs, targets) in enumerate(val_dataloder):
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
            words_preds, lables_preds = converter.decode(preds.cpu(), preds_size)
            
            for i, (word_pred, target) in enumerate(zip(words_preds, targets)):
                if word_pred == target:
                    nCorrect_words += 1
                else:
                    worng_samples.append(img_path[i])
                    worng_preds.append(word_pred)
                    
                # char level evalutaion
                target_lable = [converter.dict[c] for c in target]
                pred_lable = [converter.dict[c] for c in word_pred]
                if len(target_lable) > len(pred_lable):
                    pred_lable.extend([len(alphabet)]*abs(len(target_lable) - len(pred_lable)))
                elif len(target_lable) < len(pred_lable):
                    target_lable.extend([len(alphabet)]*abs(len(target_lable) - len(pred_lable)))
                assert len(pred_lable) == len(target_lable), f'not matched{len(target_lable)}, {len(pred_lable)}'
                y_pred.extend(pred_lable)
                y_targets.extend(target_lable)
                
    
    # display some of the network prediction
    row_preds, _ = converter.decode(preds, preds_size, raw=True)[:test_display]

    for row_pred, word_pred, gt in zip(row_preds, words_preds, targets):
        print(f'{row_pred} => {word_pred}, Ground Truth is {gt}')
    
    #compute loss and accurcy
    word_accurcy = nCorrect_words / samples_num
    val_loss /= samples_num
    
    # compute char accuracy
    char_acc = 0
    for c_p, c_t in zip(y_pred, y_targets):
        if(c_p == c_t): char_acc += 1
    char_acc /= len(y_pred)
    
    # compute prec,  recall, f1_score, weighted mode beacause of the unbalanced char apperance
    prec = metrics.precision_score(y_targets, y_pred, average='weighted')
    recall = metrics.recall_score(y_targets, y_pred, average='weighted')
    f1_score = metrics.f1_score(y_targets, y_pred, average='weighted')
    
    return val_loss, word_accurcy, char_acc, prec, recall, f1_score, y_pred, y_targets, worng_samples, worng_preds



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--use_cuda", type=bool, default=True, help="use gpu")
    parser.add_argument("--n_workers", type=int, default=4, help="number of workers to load data form disk")
    parser.add_argument("--test_display", type=int, default=4, help="number of examples outputs to display")
    parser.add_argument("--checkpoint_path", type=str, default='checkpoints/best_checkpoint.pth', help="the path for the weights file")
    parser.add_argument("--fixed_seed", type=int, default=12, help="the seed for random functionalities")
    parser.add_argument("--imgs_folder", type=str, default='data/train_passwordv3', help="imges folder path ")
    parser.add_argument("--train_file", type=str, default='data/password_train.csv', help="training csv file path")
    parser.add_argument("--val_file", type=str, default='data/password_val.csv', help="val csv file path")
    parser.add_argument("--display_wrong", type=bool, default=False, help="display every wrong image with it' worng prediciton")
    parser.add_argument("--display_confusion", type=bool, default=False, help="diplay confusion matrix for predicted characters")
    opt = parser.parse_args()
    print(opt)

    # model variables 
    inChannels = 1
    imgH = 32
    nHidden = 256
    
    # checck for cuda devices
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() and opt.use_cuda else 'cpu')
    print(f'detect cuda device? {cuda}')


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

    # validation dataloder
    val_dataset = Passwords_data(opt.val_file, opt.imgs_folder)

    align = AlignBatch()
    val_dataloder = DataLoader(val_dataset, batch_size=opt.batch_size, collate_fn=align, 
                        shuffle= False, num_workers=opt.n_workers)

    # create the model
    crnn = CRNN(imgH, inChannels, nClasses, nHidden)
    crnn.apply(weights_init)

    # load checkpoint
    checkpoint = torch.load(opt.checkpoint_path)
    crnn.load_state_dict(checkpoint['state_dic'])

    # string label converter 
    converter = strLabelConverter(alphabet, ignore_case=False)

    # loss function 
    criterion = nn.CTCLoss()

    # move to gpu 
    if cuda and opt.use_cuda :
        crnn = crnn.to(device)
        criterion = criterion.to(device)


    val_loss, word_accurcy, char_acc, prec, recall, f1_score, y_preds, y_targets, wrong_smaples, wrong_pred = evaluate(crnn, criterion, val_dataloder, opt.test_display)

    print(f'loss: {val_loss}, word_acc: {word_accurcy}, char_acc: {char_acc}, prec: {prec}, recall: {recall}, f1_score: {f1_score}')

    # display chars confusion matrix
    if opt.display_confusion :
        confusion_matrix(alphabet, y_targets, y_preds)

    # diplay the images of wrong predictions and the wrong lable
    if opt.display_wrong :
        display_imgs(wrong_smaples, wrong_pred)