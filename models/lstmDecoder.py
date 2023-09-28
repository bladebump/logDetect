import torch
import torch.nn as nn
from .base import BaseClassifier
from transformers import AutoModel

class LstmPlusTransformerModule(BaseClassifier):
    def __init__(self,vocab_size,num_classes,lr,embed_size=756):
        super(LstmPlusTransformerModule, self).__init__(vocab_size,num_classes,lr)

        self.model = AutoModel.from_pretrained('codeBERTa')

        self.decoder = nn.LSTM(self.model.config.hidden_size,self.model.config.hidden_size,1,batch_first=True)
        
        self.linear = nn.Linear(self.model.config.hidden_size,num_classes)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.model(input_ids, attention_mask=attention_mask)
        x, _ = self.decoder(x[0])
        x = x[:,-1,:]
        x = self.linear(x)
        return x
    
if __name__ == "__main__":
    model = LstmPlusTransformerModule(1000,2,1e-3)
    x = torch.randint(0,1000,(2,512))
    y = model(x)
    print(y.shape)
