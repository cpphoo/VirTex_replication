import torch
import torchvision
import torch.nn as nn

class RandomHorizontalFlip(nn.Module):
    '''
    Randomly flip the image horizontally with a probability
    Also replace left with right and vice versa in the caption
    '''
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, inp):
        # assume input is a dictionary 
        # image is a pil image
        if torch.rand(1) < self.p:
            inp['image'] = torchvision.transforms.functional.hflip(inp['image'])
            inp['caption_to_use'] = inp['caption_to_use'].replace("left", "[temp]").replace("right", "left").replace("[temp]", "right")
        return inp
    
    
class CaptionIntact(nn.Module):
    '''
        Transform the images but do not change the caption
    '''
    def __init__(self, transform):
        super().__init__()
        self.transform = transform
        
    def forward(self, inp):
        inp['image'] = self.transform(inp['image'])
        
        return inp
    

IMAGE_CROP_SIZE = 224
MAX_CAPTION_LENGTH = 30

normalize = torchvision.transforms.Normalize(
    mean=(0.485, 0.456, 0.406), 
    std=(0.229, 0.224, 0.225)
)

default_train_transform = nn.Sequential(
    CaptionIntact(torchvision.transforms.RandomResizedCrop(size=IMAGE_CROP_SIZE, scale=(0.2, 1.0))), # random resized crop
    RandomHorizontalFlip(), # this will change the caption and image
    CaptionIntact(torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)), # color jitter
    CaptionIntact(torchvision.transforms.ToTensor()), # to tensor
    CaptionIntact(normalize), # normalize
)


default_test_transform = nn.Sequential(
    CaptionIntact(torchvision.transforms.Resize(IMAGE_CROP_SIZE)), # smallest resize
    CaptionIntact(torchvision.transforms.CenterCrop(size=IMAGE_CROP_SIZE)), # centercrop
    CaptionIntact(torchvision.transforms.ToTensor()), # to tensor
    CaptionIntact(normalize), # normalize
)
