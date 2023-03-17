import torch
import torch.nn as nn

import os
import json

from collections import defaultdict
import unicodedata

from PIL import Image

import random

import sentencepiece as spm

class COCOCaption(torch.utils.data.Dataset):
    '''
        COCO Caption Dataset. 
        This only contains the basic loading functionality. 
        You will need to implement the preprocessing and augmentation yourself.
    '''
    def __init__(self, root="/scratch/datasets/cp598/coco/", split='train'):
        image_dir = os.path.join(root, f"{split}2017")

        caption_file = os.path.join(root, "annotations", f"captions_{split}2017.json")
        
        with open(caption_file) as f:
            captions = json.load(f)
            
        # (image_id, list[caption])
        captions_per_image = defaultdict(list)

        for ann in captions['annotations']:
            caption = ann['caption'].lower()
            caption = unicodedata.normalize("NFKD", caption)
            caption = "".join([chr for chr in caption if not unicodedata.combining(chr)])
            captions_per_image[ann['image_id']].append(caption)
        
        
        image_locations = {
            target["id"]: os.path.join(image_dir, target['file_name'])
            for target in captions['images']
        }
        
        self.data = [
            (im_id, image_locations[im_id], captions_per_image[im_id])
            for im_id in sorted(captions_per_image.keys())
        ]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        im_id, im_loc, im_captions = self.data[idx]
        
#         im = cv2.cvtColor(cv2.imread(im_loc), cv2.COLOR_BGR2RGB)
        im = Image.open(im_loc).convert('RGB')
        
        return {
            "image_id": im_id,
            "image": im,
            "captions": im_captions
        }
    
class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, root="/scratch/datasets/cp598/coco/",
                 split='train', transform=None,
                 tokenizer="data/tokenizer/coco_vocab_2017.model", 
                 max_caption_length=30):
        self._dset = COCOCaption(root, split)
        self.split = split
        self.transform = transform
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer)
        
        self.max_caption_length = max_caption_length
        
        # idx of control character
        # since sentencepiece internally does subword regularization
        # so the control character could accidentally get parsed into different tokens
        self.pad_idx = self.tokenizer.piece_to_id("<unk>")
        self.sos_idx = self.tokenizer.piece_to_id("[SOS]")
        self.eos_idx = self.tokenizer.piece_to_id("[EOS]")
        
        
        
    def __len__(self):
        return len(self._dset)
    
    def __getitem__(self, idx):
        datum = self._dset[idx]

        if self.split == 'train':        
            datum['caption_to_use'] = random.choice(datum['captions'])
        else:
            datum['caption_to_use'] = datum['captions'][0]
        
        if self.transform:
            datum = self.transform(datum)
            
        datum['caption_token'] = torch.LongTensor(
            [self.sos_idx, *self.tokenizer.encode(datum['caption_to_use']), self.eos_idx][:self.max_caption_length])
        
        datum["caption_token_reverse"] = datum['caption_token'].flip(0)
        datum['caption_token_length'] = len(datum['caption_token'])
            
        return datum
    

class collate_fn:
    '''
        Collate function for the CaptionDataset
    '''
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, data):
        processed_data = {}
        
        for key in data[0]:
            processed_data[key] = [datum[key] for datum in data]
        
        # stack images
        processed_data['image'] = torch.stack(processed_data['image'])
        
        # pad caption tokens
        processed_data['caption_token'] = nn.utils.rnn.pad_sequence(processed_data['caption_token'], batch_first=True, padding_value=self.pad_idx)
        processed_data['caption_token_reverse'] = nn.utils.rnn.pad_sequence(processed_data['caption_token_reverse'], batch_first=True, padding_value=self.pad_idx)
        processed_data['caption_token_length'] = torch.LongTensor(processed_data['caption_token_length'])
        return processed_data