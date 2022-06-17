# Transformer Classification 


import pytorch_lightning as pl 
import torch 
import torch.nn as nn 
from models import *

class TransformClassification(pl.LightningModule):
    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers, lr, warmup, max_iters, dropout  = 0.0, input_dropout = 0.0):
        super().__init__()
        self.save_hyperparameters() 
        self._create_model()

    

    def _create_model(self):
        self.input = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        )

        self.positional_encoding = PositionalEncodings(d_modle = self.hparams.model_dim)

        self.Transformer = TransformerEncoder(num_layers=self.hparams.num_layers,
                                              input_dim=self.hparams.model_dim,
                                              dim_feedforward=2*self.hparams.model_dim,
                                              num_heads=self.hparams.num_heads,
                                              dropout=self.hparams.dropout)

        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes)
        )
    
    def forward(self,x , mask = None, add_positional_encoding= True):
        x = self.input(x)
        x =  PositionalEncodings(x)
        x = self.Transformer(x, mask = mask)
        x = self.output_net(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
    