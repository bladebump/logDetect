from .base import BaseClassifier
import torch
from torch import nn

class LstmClassifier(BaseClassifier):
    def __init__(self, vocab_size,num_classes,lr,embedding_dim=756,**kwargs):
        super().__init__(vocab_size=vocab_size,num_classes=num_classes,lr=lr,**kwargs)

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 128, batch_first=True,bidirectional=True,dropout=0.1,num_layers=2)
        self.linear = nn.Linear(128 * 2, num_classes)

    def forward(self, input_ids,attention_mask,**kwargs):
        x = self.embed(input_ids)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x
    
if __name__ =="__main__":
    model = LstmClassifier(100,6,1e-3)
    input_ids = torch.randint(0,100,(10,100))
    output = model(input_ids,None)
    print(output.shape)