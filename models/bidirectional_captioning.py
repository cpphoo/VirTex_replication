import pytorch_lightning as pl

import torch
import torch.nn as nn
import torchvision 

import numpy as np
from .optimizer import Lookahead


# Visual Backbone: ResNet50
class resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50()
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity() 
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x
    
# Text Backbone: Word Embedding + Positional Embedding
class Token_Positional_Embedding(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=1024, dropout=0.1, 
                 padding_idx=0, max_caption_length=30):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        
        self.token_embedding = nn.Embedding(
            vocab_size, 
            embedding_dim=embed_dim, 
            padding_idx=padding_idx)
        
        
        self.position_embedding = nn.Embedding(max_caption_length, embedding_dim=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, tokens):
        
        b, l = tokens.shape
        # sum the positions and token embedding
        position_indices = torch.arange(l).expand((b, l)).to(tokens.device)
        position_features = self.position_embedding(position_indices)
        token_features = self.token_embedding(tokens)
        
        out = position_features + token_features
        
        # layer norm then dropout
        out = self.dropout(self.norm(out))
        
        mask = (tokens != self.padding_idx).unsqueeze(-1).type(out.dtype)
        
        return out*mask

# textual backbone: bidirectional transformer
class bidirectional_transformer(nn.Module):
    def __init__(self, 
                 vocab_size=10000,
                 visual_dim=2048, 
                 transformer_dimension=1024, 
                 n_heads=16, 
                 feedforward_dimension=4096,
                 dropout=0.1, padding_idx=0, max_caption_length=30):
        super().__init__()
        
        # First create token embeddings that is shared
        self.token_embedding = Token_Positional_Embedding(
            vocab_size=vocab_size,
            embed_dim=transformer_dimension, 
            dropout=dropout, 
            padding_idx=padding_idx, 
            max_caption_length=max_caption_length
        )
        
        # project the visual features to the transformer feature dimension
        self.visual_projector = nn.Linear(visual_dim, transformer_dimension)
        
        # Forward transformer
        self.forward_transformer = nn.TransformerDecoderLayer(
            d_model=transformer_dimension, 
            nhead=n_heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=False,
        )
        
        # backward transformer
        self.backward_transformer = nn.TransformerDecoderLayer(
            d_model=transformer_dimension, 
            nhead=n_heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=False,
        )
        
        # classifier head
        self.clf = nn.Linear(transformer_dimension, vocab_size)
        
        self.apply(self._init_weights)
        
    
    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                
                
    def forward(self, visual_features, forward_tokens, token_length, backward_tokens=None):
        b, d_visual, H, W = visual_features.shape
        b, l = forward_tokens.shape
        
        # change the dimension to b, Hxw, d_visual
        visual_features = visual_features.reshape(b, d_visual, -1).transpose(2, 1)
        projected_visual_features = self.visual_projector(visual_features)
        
        forward_token_features = self.token_embedding(forward_tokens)
        
        # making the model causal so need to mask out "future"
        attention_mask = torch.triu(torch.full((l, l), float("-inf"), device=visual_features.device), diagonal=1)
        
        # mask corresponding to padding
        padding_mask = (torch.arange(1, l+1, device=visual_features.device).expand((b, l)) > token_length.unsqueeze(1))
        
        forward_logits = self.clf(self.forward_transformer(
            tgt=forward_token_features,
            memory=projected_visual_features,
            tgt_mask=attention_mask,
            tgt_key_padding_mask=padding_mask
        ))
        
        if backward_tokens is None:
            backward_logits = None
        else:
            backward_token_features = self.token_embedding(backward_tokens)
            backward_logits = self.clf(self.backward_transformer(
                tgt=backward_token_features,
                memory=projected_visual_features,
                tgt_mask=attention_mask,
                tgt_key_padding_mask=padding_mask
            ))
        
        
        return forward_logits, backward_logits
    
class BiDirectional_Captioning_Model(pl.LightningModule):
    def __init__(self, vocab_size=10000,
                 visual_dim=2048, 
                 transformer_dimension=1024, 
                 n_heads=16, 
                 feedforward_dimension=4096,
                 dropout=0.1, padding_idx=0, 
                 sos_idx=1, 
                 eos_idx=2,
                 max_caption_length=30,
                 optim_params={
                     "visual_lr": 0.2,
                     "textual_lr": 0.001,
                     "wd": 1e-4,
                     "warmup_steps": 10000,
                     "total_steps": 500000}
                ):
        super().__init__()
        self.save_hyperparameters()
        self.visual_backbone = resnet50()
        self.textual_head = bidirectional_transformer(vocab_size,
                 visual_dim, 
                 transformer_dimension, 
                 n_heads, 
                 feedforward_dimension,
                 dropout, padding_idx, max_caption_length)
        
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        self.optim_params = optim_params
        
    def forward(self, batch):
        visual_features = self.visual_backbone(batch['image'])
        
        forward_logits, backward_logits = self.textual_head(visual_features, 
                                                            batch['caption_token'], 
                                                            batch['caption_token_length'],
                                                            batch['caption_token_reverse'])
        return forward_logits, backward_logits

    def extract_visual_features(self, X):
        return self.visual_backbone(X)
    
    def training_step(self, batch, batch_idx):
        batch_size = batch['image'].shape[0]
        forward_logits, backward_logits = self.forward(batch)
        
        forward_loss = self.loss(
            forward_logits[:, :-1].contiguous().view(-1, self.vocab_size),
            batch['caption_token'][:, 1:].reshape(-1)
        )
        
        backward_loss = self.loss(
            backward_logits[:, :-1].contiguous().view(-1, self.vocab_size),
            batch['caption_token_reverse'][:, 1:].reshape(-1)
        )
        
        loss = forward_loss + backward_loss
        self.log("train/loss_forward", forward_loss, batch_size=batch_size,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/loss_backward", backward_loss, batch_size=batch_size,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/loss", loss, batch_size=batch_size,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch['image'].shape[0]
        forward_logits, backward_logits = self.forward(batch)
        
        forward_loss = self.loss(
            forward_logits[:, :-1].contiguous().view(-1, self.vocab_size),
            batch['caption_token'][:, 1:].reshape(-1)
        )
        
        backward_loss = self.loss(
            backward_logits[:, :-1].contiguous().view(-1, self.vocab_size),
            batch['caption_token_reverse'][:, 1:].reshape(-1)
        )
        
        loss = forward_loss + backward_loss
        self.log("val/loss_forward", forward_loss, batch_size=batch_size,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/loss_backward", backward_loss, batch_size=batch_size,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/loss", loss, batch_size=batch_size,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    
    def configure_optimizers(self):

        def generate_lr_schedule(total_steps, warmup_steps):
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    multiplicative_factor = float(current_step) / float(max(1, warmup_steps))
                else:
                    ratio = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    multiplicative_factor = np.cos(np.pi * ratio * 0.5) ** 2

                return max(0.0, multiplicative_factor)
            return lr_lambda

        # def generate_cosine_decay_with_warmup(total_steps, warmup_steps):
        #     return np.concatenate(
        #         [np.linspace(0, 1, warmup_steps)[:-1],
        #         np.cos(np.linspace(0, 1, total_steps - warmup_steps + 1)*np.pi / 2) **2]).tolist()
        # no weight decay layer norm or bias for transformer
        no_wd_group = []
        wd_group = []
        for name, p in self.textual_head.named_parameters():
            if 'norm' in name or 'bias' in name:
                no_wd_group.append(p)
            else:
                wd_group.append(p)
        
        
        # different lr for backbone: 0.2
        # transformer 0.001
        # cosine for 500k with 10k warmup
        
        base_optimizer = torch.optim.SGD([
            # visual backbone
            {'params': self.visual_backbone.parameters(), 'lr': self.optim_params["visual_lr"]},
            # textual head
            {'params': wd_group, 'lr': self.optim_params["textual_lr"]},
            {'params': no_wd_group, 'weight_decay': 0.0, 'lr': self.optim_params["textual_lr"]} 
        ], weight_decay=self.optim_params["wd"], momentum=0.9)
        
        lr_factor = generate_lr_schedule(
            total_steps=self.optim_params["total_steps"], 
            warmup_steps=self.optim_params["warmup_steps"])
        
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": 'step',
                "frequency": 1
            }
        }