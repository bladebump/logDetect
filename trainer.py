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
config = wandb.config
config.batch_size = 32
config.model_name = "RetNetClassifier"
config.lr = 1e-3
config.tokenizer_name = "codeBERTa"
config.max_length = 512
config.num_classes = 6

if not debug:
    wandb.login(key='7636d0cc1edf410cae67d21d09968d70d6791a89')
    wandb_logger = WandbLogger(project="logDetect",name=f"tokenizer-{config.tokenizer_name}-name-{config.model_name}",save_dir="logs")

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("codeBERTa")
    tokenizer.pad_token = tokenizer.eos_token
    train_data, val_data = load_match_data(tokenizer,max_length=config.max_length)
    
    if config.model_name == "LstmPlusTransformerModule":
        model = LstmPlusTransformerModule(tokenizer.vocab_size,out_size=config.num_classes)
    elif config.model_name == "TestModule":
        model = TestModule(tokenizer.vocab_size,out_size=config.num_classes)
    elif config.model_name == "RetNetClassifier":
        model = RetNetClassifier(tokenizer.vocab_size,num_classes=config.num_classes)

    if torch.cuda.is_available():
        trainer = pl.Trainer(max_epochs=1,enable_model_summary=True,logger=wandb_logger,devices=[0],accelerator='gpu')
    else:
        config.batch_size = 2
        trainer = pl.Trainer(max_epochs=1,enable_model_summary=True,logger=wandb_logger)
        
    train_data = DataLoader(train_data,batch_size=config.batch_size,shuffle=True,drop_last=True)
    val_data = DataLoader(val_data,batch_size=config.batch_size,shuffle=True,drop_last=True)
    trainer.fit(model, train_data,val_dataloaders=val_data)
