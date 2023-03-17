import data
from models import BiDirectional_Captioning_Model

import torch 

import pytorch_lightning as pl


from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import argparse 


def main(args):
    pl.seed_everything(args.seed)
    logger = WandbLogger(project="VirTeX", name=args.wandb_name)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                            dirpath=args.save_dir, save_last=True, 
                            save_top_k=-1, every_n_train_steps=args.checkpoint_freq)

    trainset = data.CaptionDataset(
        root=args.data_dir,
        split='train', 
        transform=data.default_train_transform,
        tokenizer=args.tokenizer, 
        max_caption_length=args.max_caption_length)

    valset = data.CaptionDataset(
        root=args.data_dir,
        split='val', 
        transform=data.default_test_transform,
        tokenizer=args.tokenizer, 
        max_caption_length=args.max_caption_length)


    trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=args.batch_size, 
                                            num_workers=args.num_workers,
                                            drop_last=True, 
                                            shuffle=True, 
                                            pin_memory=True,
                                            collate_fn=data.collate_fn(trainset.pad_idx))

    valloader = torch.utils.data.DataLoader(valset, 
                                            batch_size=args.batch_size, 
                                            num_workers=args.num_workers,
                                            drop_last=False,
                                            shuffle=False, 
                                            pin_memory=True,
                                            collate_fn=data.collate_fn(trainset.pad_idx))
    trainer = pl.Trainer(max_steps=args.max_steps, 
                        logger=logger,
                        accelerator="gpu", 
                        strategy="ddp", 
                        precision=16, 
                        val_check_interval=args.val_check_interval,
                        check_val_every_n_epoch=None,
                        sync_batchnorm=True,
                        deterministic=True,
                        gradient_clip_algorithm="norm",
                        gradient_clip_val=args.gradient_clip_val,
                        callbacks=[lr_monitor, checkpoint_callback])
    
    model = BiDirectional_Captioning_Model(
        vocab_size=len(trainset.tokenizer),
        visual_dim=args.visual_dim, 
        transformer_dimension=args.transformer_dimension, 
        n_heads=args.n_heads, 
        feedforward_dimension=args.feedforward_dimension,
        dropout=args.dropout, 
        padding_idx=trainset.pad_idx, 
        sos_idx=trainset.sos_idx, 
        eos_idx=trainset.eos_idx,
        max_caption_length=args.max_caption_length,
        optim_params={
            "visual_lr": args.visual_lr,
            "textual_lr": args.textual_lr,
            "wd": args.wd,
            "warmup_steps": min(args.warmup_steps, args.max_steps),
            "total_steps": args.max_steps})

    trainer.fit(model, 
                train_dataloaders=trainloader, 
                val_dataloaders=valloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    #directory to checkpoint and save the model 
    parser.add_argument("--save_dir", type=str, help="directory to save the model")
    parser.add_argument("--data_dir", type=str, default="/scratch/datasets/cp598/coco/", 
                        help="Directory to look for the COCO Caption Dataset")
    parser.add_argument("--wandb_name", type=str, default="VirTex", 
                        help="name of the wandb run (for tracking purposes)")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/coco_vocab_2017.model", 
                        help="Path to the tokenizer model")
    

    # model hyperparameters
    parser.add_argument("--max_caption_length", type=int, default=30, help="max caption length")
    parser.add_argument("--visual_dim", type=int, default=2048, 
                        help="visual feature dimension. This changes depending on the backbone. Currently only ResNet50 is implemented")
    parser.add_argument("--transformer_dimension", type=int, default=1024, help="Hidden dimension of the transformer")
    parser.add_argument("--n_heads", type=int, default=16, help="Number of heads in the transformer")
    parser.add_argument("--feedforward_dimension", type=int, default=4096, help="dimension of the dense bottleneck in transformer layer")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate for the transformer")
    
    

    # training hyperparameters
    parser.add_argument('--max_steps', type=int, default=50000, help='max steps to train')
    parser.add_argument("--batch_size", type=int, default=64, help="batch size per gpus")
    parser.add_argument("--num_workers", type=int, default=4, help="workers for the dataloader")
    parser.add_argument("--visual_lr", type=float, default=0.2, help="learning rate for the visual backbone")
    parser.add_argument("--textual_lr", type=float, default=0.001, help="learning rate for the textual backbone")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay for the optimizer")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="warmup steps for the learning rate linear wamrup scheduler")
    parser.add_argument("--gradient_clip_val", type=float, default=10.0, help="gradient clipping value")

    
    # miscellaneous
    parser.add_argument('--checkpoint_freq', type=int, default=1000, help='frequency of saving checkpoints')
    parser.add_argument('--val_check_interval', type=int, default=1000, help='frequency of validation')
    parser.add_argument("--seed", type=int, default=1, help="random seed")

    args = parser.parse_args()
    main(args)