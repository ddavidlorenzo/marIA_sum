import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import add_special_tokens


class GPT2SumDataset(Dataset):


    __modes__ = set(["train", "val", "test"])
    __pad_mask = None

    def __init__(self,
                 root_dir,
                 max_input,
                 sep_token_id,
                 pad_token_id,
                 mode='train',
                 length=None):
        if mode not in self.__modes__:
            raise ValueError(f'Invalid type of data, try again with one of {self.__modes__}')

        self.root_dir = f'{root_dir}/{mode}/'
        self.__sep_token_id = sep_token_id
        self.__pad_token_id = pad_token_id
        self.max_input = max_input

        # Set fixed pad mask, constraint to `max_input` tokens
        self.pad_mask = pad_token_id

        self.idxs = os.listdir(self.root_dir)
        self.mode = mode
        if length == None:
            self.len = len(self.idxs)
        else:
            self.len = length
    
    @property
    def max_input(self):
        return self.__max_input

    @max_input.setter
    def max_input(self, max_input):
        if max_input < 1:
            raise ValueError(f'The maximum number of tokens must be greater than 1, not {max_input}')
        self.__max_input = max_input

    @property
    def pad_mask(self):
        if not self.__pad_mask:
            print(f'Pad mask not set')
            self.pad_mask = self.pad_token_id
        return self.__pad_mask[:]

    @pad_mask.setter
    def pad_mask(self, pad_token_id):
        self.__pad_mask = [pad_token_id]*self.max_input

    @property
    def sep_token_id(self):
        return self.__sep_token_id

    @property
    def pad_token_id(self):
        return self.__pad_token_id


    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        file_name = os.path.join(self.root_dir,str(idx)+'.json')
        with open(file_name,'r') as f:
              data = json.load(f)
        text = self.pad_mask
        content = data['article'] + [self.sep_token_id] + data['abstract']
        text[:len(content)] = content

        if len(text) > self.max_input:
            raise Exception(f'file {file_name} has {len(text)} tokens!')

        text = torch.tensor(text)

        return {'article': text, 'sum_idx': len(data['article'])}