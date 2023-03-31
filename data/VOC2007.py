import torch

import os

from collections import defaultdict

from PIL import Image


class VOC2007(torch.utils.data.Dataset):
    def __init__(self, root='/home/cp598/scratch/datasets/cp598/VOC2007', split='trainval', transforms=None):
        self.root = os.path.join(root, f"VOC2007_{split}")
        self.split = split
        self.transforms = transforms
        
        self.ann_paths = [
            os.path.join(self.root, "ImageSets", "Main", i) for i in os.listdir(
                os.path.join(self.root, "ImageSets", "Main")) if i.endswith(f"_{split}.txt") 
        ]
        
        self.classes = sorted(
            [os.path.basename(p).split('_')[0] for p in self.ann_paths]
        )
        
        self.class_to_idx = {i: ind for ind, i in enumerate(self.classes)}
        
        self.image_labels = defaultdict(
            lambda: -torch.ones(len(self.classes), dtype=torch.long)
        )
        
        for ind_c, cl in enumerate(self.classes):
            with open(os.path.join(self.root, "ImageSets", "Main", f"{cl}_{split}.txt"), "r") as f:
                for line in f:
                    # label should be {-1, 0, 1}
                    # -1 is not present, 1 is present, 0 is ignored
                    img_name, label = line.strip().split()
                    label = int(label)
                        
                    self.image_labels[img_name][ind_c] = label
                    
        self.dset_list = [
            (os.path.join(self.root, "JPEGImages", f"{i}.jpg"), self.image_labels[i]) for i in self.image_labels
        ]
                    

    def __len__(self):
        return len(self.dset_list)
        
    def __getitem__(self, idx):
        img_path, label = self.dset_list[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transforms:
            image = self.transforms({"image":image})["image"]
        return image, label