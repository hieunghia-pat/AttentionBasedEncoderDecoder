from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import os
import cv2 as cv
import numpy as np
import os
import json

from torch.utils.data.dataloader import DataLoader
from data_utils.utils import collate_fn, preprocess_sentence
from data_utils.vocab import Vocab

import config

class OCRDataset(Dataset):
    def __init__(self, dirs, image_size, out_level, vocab=None):
        super(OCRDataset, self).__init__()
        self.size = image_size
        self.out_level = out_level
        self.vocab = vocab if vocab is not None else Vocab(dir, out_level)
        self.get_groundtruth(dirs)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        '''
        samples: [{"image": image file, "tokens": ["B", "a", "o", " ", "g", "ồ", "m"], "gt": "Bao gồm"}, ... ]
        '''

        sample = self.samples[index]
        img_path, label = sample["image"], sample["label"]
        img = cv.imread(img_path)

        img = img / 255.

        # resize the image 
        img_h, img_w, _ = img.shape 
        w, h = self.size 
        if w == -1: # keep h, scale w according to h
            scale = img_w / img_h 
            w = round(scale * h)

        if h == -1: # keep w, scale h according to w
            scale = img_w / img_h 
            h = round(w / scale)

        img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
            
        # Channels-first
        img = np.transpose(img, (2, 0, 1))
        # As pytorch tensor
        img = torch.from_numpy(img).float()

        tokens = torch.ones(self.max_len, dtype=int) * self.vocab.padding_idx
        for idx, token in enumerate([self.vocab.sos_token] + label):
            tokens[idx] = self.vocab.stoi[token]
        shifted_right_tokens = torch.ones(self.max_len, dtype=int) * self.vocab.padding_idx
        for idx, token in enumerate(label + [self.vocab.eos_token]):
            shifted_right_tokens[idx] = self.vocab.stoi[token]
        
        return img, tokens, shifted_right_tokens

    def get_groundtruth(self, image_dirs):
        self.max_len = 0
        self.samples = []

        for image_dir in image_dirs:
            for folder in os.listdir(image_dir):
                labels = json.load(open(os.path.join(image_dir, folder, "label.json")))
                for img_file, label in labels.items():
                    label = preprocess_sentence(label, self.vocab.out_level)
                    self.samples.append({"image": os.path.join(image_dir, folder, img_file), "label": label})
                    if self.max_len < len(label) + 2:
                        self.max_len = len(label) + 2

    def get_folds(self, k=5) -> List[DataLoader]:
        fold_size = len(self) // k
        splits = [fold_size]*(k-1) + [len(self) - fold_size*(k-1)]
        subdatasets = random_split(self, splits, torch.Generator().manual_seed(13))

        loaders = []
        for subdataset in subdatasets:
            loaders.append(DataLoader(subdataset, 
                                        batch_size=config.batch_train, 
                                        shuffle=True, 
                                        collate_fn=collate_fn))

        return loaders

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, imgs, tokens, shifted_right_tokens, padding_idx=0, device="cuda"):
        self.imgs = imgs.to(device)
        self.tokens = tokens.to(device)
        self.shifted_right_tokens = shifted_right_tokens.to(device)
        self.ntokens = (self.shifted_right_tokens != padding_idx).sum()
