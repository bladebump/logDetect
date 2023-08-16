from typing import Any
import torch
import torch.nn as nn
from .base import BaseClassifier

class TransformereEncoderClassifier(BaseClassifier):
    def __init__(self,vocab_size,num_classes,lr,embed_size=756):
        super(TransformereEncoderClassifier, self).__init__(vocab_size,num_classes,lr)
        self.embed = nn.Embedding(vocab_size,embed_size)

        # one layer transformer
        encoderlayer = nn.TransformerEncoderLayer(embed_size, 3, 2048, 0.1)
        self.transformer = nn.TransformerEncoder(encoderlayer, 1)
        
        self.linear = nn.Linear(embed_size,num_classes)
        self.gblobel_avg_pool = nn.AdaptiveAvgPool1d(1)


    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is not None:
            attention_mask = (attention_mask == 0)
        x = self.embed(input_ids)
        x = x.permute(1,0,2)
        x = self.transformer(x , src_key_padding_mask=attention_mask)
        x = self.linear(x)
        x = x.permute(1,2,0)
        x = self.gblobel_avg_pool(x)
        return x.squeeze(2)
    
    
if __name__ == "__main__":
    model = TransformereEncoderClassifier(1000,out_size=6)
    x = torch.randint(0,1000,(10,100))
    y = model(x)
    print(y.shape)
