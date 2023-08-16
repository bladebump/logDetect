from .base import BaseClassifier
import torch
from torch import nn

class textCnnClassifier(BaseClassifier):
    def __init__(self, vocab_size,num_classes,lr,embed_size=756,hidden_features=128,**kwargs):
        super(textCnnClassifier, self).__init__(vocab_size,num_classes,lr,**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, hidden_features, (k, embed_size)) for k in [3, 4, 5]
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_features * len(self.convs), num_classes)

    def forward(self, input_ids,**kwargs):
        x = self.embedding(input_ids)
        x = x.unsqueeze(1)
        x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
    
if __name__ == "__main__":
    model = textCnnClassifier(100,10,0.01)
    print(model)

    x = torch.randint(0,100,(10,100))
    print(x.shape)
    y = model(x)
    print(y.shape)