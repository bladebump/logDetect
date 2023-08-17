from datachange import load_match_data
from models import TransformereEncoderClassifier,LstmPlusTransformerModule,RetNetClassifier,LstmClassifier,textCnnClassifier,textCnnAndLstmClassifier,CodeBertClassifier
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import wandb
import torch
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rune_name",type=str)
    parser.add_argument("--model_name",type=str,default="TestModule")
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--max_length",type=int,default=512)
    parser.add_argument("--num_classes",type=int,default=6)
    parser.add_argument("--tokenizer_name",type=str,default="codeBERTa")
    parser.add_argument("--debug",type=bool,default=False)
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--seed",type=int,default=42)
    args = parser.parse_args()

    debug = args.debug

    if not debug:
        wandb.login(key='7636d0cc1edf410cae67d21d09968d70d6791a89')
        wandb_logger = WandbLogger(project="logDetect",name=args.rune_name,save_dir="logs")
        config = wandb_logger.experiment.config
        config.batch_size = args.batch_size
        config.model_name = args.model_name
        config.lr = args.lr
        config.tokenizer_name = args.tokenizer_name
        config.max_length = args.max_length
        config.num_classes = args.num_classes
        config.epochs = args.epochs
        config.seed = args.seed

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_data, val_data = load_match_data(tokenizer,max_length=config.max_length,random_seed=config.seed)
    
    if config.model_name == "LstmPlusTransformerModule":
        model = LstmPlusTransformerModule(tokenizer.vocab_size,num_classes=config.num_classes,lr=config.lr)
    elif config.model_name == "TransformereEncoderClassifier":
        model = TransformereEncoderClassifier(tokenizer.vocab_size,num_classes=config.num_classes,lr=config.lr)
    elif config.model_name == "RetNetClassifier":
        model = RetNetClassifier(tokenizer.vocab_size,num_classes=config.num_classes,lr=config.lr)
    elif config.model_name == "LstmClassifier":
        model = LstmClassifier(tokenizer.vocab_size,num_classes=config.num_classes,lr=config.lr)
    elif config.model_name == "textCnnClassifier":
        model = textCnnClassifier(tokenizer.vocab_size,num_classes=config.num_classes,lr=config.lr)
    elif config.model_name == "textCnnAndLstmClassifier":
        model = textCnnAndLstmClassifier(tokenizer.vocab_size,num_classes=config.num_classes,lr=config.lr)
    elif config.model_name == "CodeBertClassifier":
        model = CodeBertClassifier(tokenizer.vocab_size,num_classes=config.num_classes,lr=config.lr)

    if torch.cuda.is_available():
        trainer = pl.Trainer(max_epochs=config.epochs,enable_model_summary=True,logger=wandb_logger,devices=[0],accelerator='gpu')
    else:
        config.batch_size = 2
        trainer = pl.Trainer(max_epochs=config.epochs,enable_model_summary=True,logger=wandb_logger)

    train_data = DataLoader(train_data,batch_size=config.batch_size,shuffle=True,drop_last=True)
    val_data = DataLoader(val_data,batch_size=config.batch_size,shuffle=True,drop_last=True)
    trainer.fit(model, train_data,val_dataloaders=val_data)
