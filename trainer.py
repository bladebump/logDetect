from datachange import load_match_data
from models.test import TestModule
from models.lstmDecoder import LstmPlusTransformerModule
from models.retnet import RetNetClassifier
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import wandb
import torch

debug = False
model_name = "RetNetClassifier"

if not debug:
    wandb.login(key='7636d0cc1edf410cae67d21d09968d70d6791a89')
    wandb_logger = WandbLogger(project="logDetect",name=f"tokenizer-codeBERTa-name-{model_name}",save_dir="logs")

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("codeBERTa")
    tokenizer.pad_token = tokenizer.eos_token
    train_data, val_data = load_match_data(tokenizer,max_length=512)
    train_data = DataLoader(train_data,batch_size=2,shuffle=True)
    val_data = DataLoader(val_data,batch_size=2,shuffle=True)
    
    if model_name == "LstmPlusTransformerModule":
        model = LstmPlusTransformerModule(tokenizer.vocab_size,out_size=6)
    elif model_name == "TestModule":
        model = TestModule(tokenizer.vocab_size,out_size=6)
    elif model_name == "RetNetClassifier":
        model = RetNetClassifier(tokenizer.vocab_size,num_classes=6)

    if torch.cuda.is_available():
        trainer = pl.Trainer(max_epochs=1,enable_model_summary=True,logger=wandb_logger,devices=[0],accelerator='gpu')
    else:
        trainer = pl.Trainer(max_epochs=1,enable_model_summary=True,logger=wandb_logger)
    trainer.fit(model, train_data,val_dataloaders=val_data)
