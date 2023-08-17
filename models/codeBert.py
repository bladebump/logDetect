from .base import BaseClassifier
import torch
from torch import nn
from transformers import AutoModel

class CodeBertClassifier(BaseClassifier):
    def __init__(self, vocab_size, num_classes, lr, **kwargs):
        super().__init__(vocab_size, num_classes, lr, **kwargs)

        self.model = AutoModel.from_pretrained('codeBERTa')
        self.fc = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)
        output = self.fc(output[1])
        return output


if __name__ == "__main__":
    model = CodeBertClassifier(100, 10, 1e-4)
    input_ids = torch.randint(0, 100, (32, 128))
    attention_mask = torch.randint(0, 2, (32, 128))
    output = model(input_ids, attention_mask)
    print(output.shape)