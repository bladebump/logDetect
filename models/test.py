from typing import Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy

class TestModule(pl.LightningModule):
    def __init__(self,vocab_size,out_size,embed_size=756):
        super(TestModule, self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)

        # one layer transformer
        encoderlayer = nn.TransformerEncoderLayer(embed_size, 3, 2048, 0.1)
        self.transformer = nn.TransformerEncoder(encoderlayer, 1)
        
        self.linear = nn.Linear(embed_size,out_size)
        self.gblobel_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.acc = Accuracy(task='multiclass',num_classes=out_size)

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
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def one_step(self,batch,batch_idx,out_str):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
        outputs = self.forward(input_ids, attention_mask)
        lossfn = nn.CrossEntropyLoss()
        loss = lossfn(outputs, labels)
        self.log(out_str + "_loss", loss)
        acc = self.acc(outputs, labels)
        self.log(out_str + '_acc', acc)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.one_step(batch,batch_idx,'train')
    
    def validation_step(self, batch, batch_idx):
        return self.one_step(batch,batch_idx,'val')
    
    def test_step(self, batch, batch_idx):
        return self.one_step(batch,batch_idx,'test')
    
if __name__ == "__main__":
    model = TestModule(1000,out_size=6)
    x = torch.randint(0,1000,(10,100))
    y = model(x)
    print(y.shape)
