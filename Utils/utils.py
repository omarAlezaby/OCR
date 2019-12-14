from torch.utils.data import Dataset, sampler, DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
from PIL import Image
import random
import collections

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.abc.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t]), [i - 1 for i in t]
            else:
                char_list = []
                lables_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                        lables_list.append(t[i] - 1)
                return ''.join(char_list), lables_list
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            lables = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                text, lable = self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw)
                texts.append(text)
                lables.append(lable)
                index += l
            return texts, lables


def confusion_matrix(alphapet, y_targets, y_pred):
    # plot confusion matrix
    confusion_matrix = np.zeros((len(alphapet)+1, len(alphapet)+1))
    for c_p, c_t in zip(y_pred, y_targets):
        confusion_matrix[c_t, c_p] += 1
        
    df = pd.DataFrame(confusion_matrix, index=[c for c in (alphapet+'-')], columns=[c for c in (alphapet+'-')])
    plt.figure(figsize=(30,30))
    sn.set(font_scale=1)
    sn.heatmap(df, annot=True)

def display_imgs(paths, preds):
    # display wrong samples\
    for path, pred in zip(paths, preds):
        plt.title(pred)
        plt.imshow(np.array(Image.open(path).convert('L')), cmap='gray')
        plt.show()