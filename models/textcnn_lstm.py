from .base import BaseClassifier
import torch
from torch import nn

class textCnnAndLstmClassifier(BaseClassifier):
    def __init__(self, vocab_size,num_classes,lr,embedding_dim=756,conv_features=128,**kwargs):
        super(textCnnAndLstmClassifier, self).__init__(vocab_size,num_classes,lr,**kwargs)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.text_conv = nn.ModuleList([nn.Conv2d(1, conv_features, (k, embedding_dim)) for k in [3, 4, 5]])
        self.lstm = nn.LSTM(embedding_dim, conv_features, 2, batch_first=True, bidirectional=True)

        self.total_features = conv_features * 2 + conv_features * 3
        self.out = nn.Linear(self.total_features, num_classes)

    def forward(self, input_ids,**kwargs):
        x = self.embedding(input_ids)
        x_cnn = x.unsqueeze(1)
        x_cnn = [nn.functional.relu(conv(x_cnn)).squeeze(3) for conv in self.text_conv]
        x_cnn = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x_cnn]
        x_cnn = torch.cat(x_cnn, 1)
        x_lstm,_ = self.lstm(x)
        x_lstm = x_lstm[:, -1, :]
        x = torch.cat([x_cnn, x_lstm], 1)
        x = self.out(x)
        return x

if __name__ == "__main__":
    model = textCnnAndLstmClassifier(100,10,0.01)
    x = torch.randint(0,100,(10,100))
    y = model(x)
    print(y.shape)
        
