# VirTex
This repo contains a replication of the CVPR 2021 paper VirTeX: Learning Visual Representations from Textual Annotations (https://arxiv.org/abs/2010.07734arxiv.org/abs/2006.06666). To start, please install all the packages specified in `requirements.txt`. Also you would need the COCO Caption Dataset (https://cocodataset.org/#download) and VOC2007 dataset (http://host.robots.ox.ac.uk/pascal/VOC/). 

### Step 0: Generate the Vocabulary for the COCO Caption Dataset  
``` bash
cd data/
python generate_vocabulary.py --data {path to coco caption annotations}
```

### Step 1: Train a visual representation using the bidirectional captioning model
```bash
python train.py 
--save_dir {directory_to_save_the_model} \
--wandb_name {name_for_wandb_trakcing} \
--data_dir {where_to_find_the_COCO_Caption_Dataset} \
--max_steps 500000 \
--batch_size 64 \
--visual_lr 0.2 \
--textual_lr 0.001 \
--warmup_steps 10000 \
--gradient_clip_val 10.0 \
--checkpoint_freq 5000 \
--val_check_interval 1000 \
--seed 1 \
```


### Step 3: Linear Probing on VOC2007
```bash
python downstream_VOC2007.py --model_path {checkpoint_file} --data_dir {where_to_find_VOC2007}
```  

## Notes
1. This repo is tested on 4 NVidia A6000 gpus. 
2. On VOC2007, our replication is able to achieve mAP of 88.0 (reported results in the table 1 of the paper is 88.7)