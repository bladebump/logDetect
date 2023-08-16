import torch
import torch.nn as nn
from .base import BaseClassifier

class LstmPlusTransformerModule(BaseClassifier):
    def __init__(self,vocab_size,num_classes,lr,embed_size=756):
        super(LstmPlusTransformerModule, self).__init__(vocab_size,num_classes,lr)

        self.embed = nn.Embedding(vocab_size,embed_size)

        # one layer transformer
        encoderlayer = nn.TransformerEncoderLayer(embed_size, 3, 2048, 0.1)
        self.transformer = nn.TransformerEncoder(encoderlayer, 1)

        # Lstm decoder
        self.decoder = nn.LSTM(embed_size,embed_size,1,batch_first=True)
        
        self.linear = nn.Linear(embed_size,num_classes)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is not None:
            attention_mask = (attention_mask == 0)
        x = self.embed(input_ids)
        x = x.permute(1,0,2)
        x = self.transformer(x , src_key_padding_mask=attention_mask)
        x = x.permute(1,0,2)
        x, _ = self.decoder(x)
        x = x[:,-1,:]
        x = self.linear(x)
        return x
    
if __name__ == "__main__":
    model = LstmPlusTransformerModule(1000,2,1e-3)
    x = torch.randint(0,1000,(2,512))
    y = model(x)
    print(y.shape)
