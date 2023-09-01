import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC, AveragePrecision, ConfusionMatrix

class BaseClassifier(pl.LightningModule):
    def __init__(self, vocab_size, num_classes,lr,**kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.lr = lr

        self.acc = Accuracy(task='multiclass',num_classes=num_classes)
        self.f1 = F1Score(task='multiclass',num_classes=num_classes,average='macro')
        self.precision = Precision(task='multiclass',num_classes=num_classes,average='macro')
        self.recall = Recall(task='multiclass',num_classes=num_classes,average='macro')
        self.auroc = AUROC(task='multiclass',num_classes=num_classes)
        self.ap = AveragePrecision(task='multiclass',num_classes=num_classes)
        self.confusion_matrix = ConfusionMatrix(task='multiclass',num_classes=num_classes)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids,attention_mask=None,**kwargs):
        pass

    def one_step(self,batch,batch_idx,out_str):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
        outputs = self.forward(input_ids, attention_mask=attention_mask)
        loss = self.loss(outputs, labels)
        self.log(out_str + "_loss", loss)
        acc = self.acc(outputs, labels)
        self.log(out_str + '_acc', acc)
        f1 = self.f1(outputs, labels)
        self.log(out_str + '_f1', f1)
        precision = self.precision(outputs, labels)
        self.log(out_str + '_precision', precision)
        recall = self.recall(outputs, labels)
        self.log(out_str + '_recall', recall)
        auroc = self.auroc(outputs, labels)
        self.log(out_str + '_auroc', auroc)
        ap = self.ap(outputs, labels)
        self.log(out_str + '_ap', ap)
        confusion_matrix = self.confusion_matrix(outputs, labels)
        
        # log confusion matrix
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                self.log(out_str + f'_confusion_matrix_{i}_{j}', confusion_matrix[i][j])
                
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.one_step(batch,batch_idx,'train')
    
    def validation_step(self, batch, batch_idx):
        return self.one_step(batch,batch_idx,'val')
    
    def test_step(self, batch, batch_idx):
        return self.one_step(batch,batch_idx,'test')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer