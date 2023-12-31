from datachange import load_match_data,load_bigdata,make_weights_for_balanced_classes
from models import TransformereEncoderClassifier,LstmPlusTransformerModule,RetNetClassifier,LstmClassifier,textCnnClassifier,textCnnAndLstmClassifier,CodeBertClassifier
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import wandb
import torch
import argparse
import os

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--rune_name",type=str)
    parser.add_argument("--model_name",type=str,default="textCnnClassifier")
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--max_length",type=int,default=512)
    parser.add_argument("--num_classes",type=int,default=6)
    parser.add_argument("--tokenizer_name",type=str,default="codeBERTa")
    parser.add_argument("--debug",type=int,default=0)
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--data_name",type=str,default="match")
    parser.add_argument("--use_mutillabel",type=int,default=1)
    parser.add_argument("--device_id",type=int,default=0)
    parser.add_argument("--limit_train_batches",type=float,default=0.1)
    parser.add_argument("--use_weight",type=int,default=0)
    parser.add_argument("--use_gpu",type=int,default=1)
    args = parser.parse_args()

    debug = bool(args.debug)

    if not debug:
        wandb.login(key='7636d0cc1edf410cae67d21d09968d70d6791a89')
        wandb_logger = WandbLogger(project="logDetect-fine",name=args.rune_name + "-" + args.model_name,save_dir="logs")
        config = wandb_logger.experiment.config
        config.batch_size = args.batch_size
        config.model_name = args.model_name
        config.lr = args.lr
        config.tokenizer_name = args.tokenizer_name
        config.max_length = args.max_length
        config.num_classes = args.num_classes
        config.epochs = args.epochs
        config.seed = args.seed
        config.data_name = args.data_name
        config.use_mutillabel = bool(args.use_mutillabel)
        config.limit_train_batches = args.limit_train_batches
        config.use_weight = bool(args.use_weight)
        config.use_gpu = bool(args.use_gpu)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    if config.data_name == "match":
        train_data, val_data = load_match_data(tokenizer,max_length=config.max_length,random_seed=config.seed)
    elif config.data_name == "bigdata":
        use_mutillabel = config.use_mutillabel
        train_data, val_data = load_bigdata(tokenizer,max_length=config.max_length,random_seed=config.seed,use_mtilabel=use_mutillabel)
    
    if not config.use_mutillabel:
        config.num_classes = 2
    
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

    if torch.cuda.is_available() and config.use_gpu:
        print("Using GPU")
        trainer = pl.Trainer(max_epochs=config.epochs,enable_model_summary=True,logger=wandb_logger,devices=[args.device_id],accelerator='gpu',limit_train_batches=config.limit_train_batches,callbacks=[EarlyStopping(monitor='val_f1',mode='max'),ModelCheckpoint(dirpath=f"saves/{args.model_name}-{args.tokenizer_name}",monitor='val_f1',mode='max',save_top_k=1,filename='{val_f1:.5f}')])
    else:
        trainer = pl.Trainer(max_epochs=config.epochs,enable_model_summary=True,logger=wandb_logger,accelerator = 'cpu',limit_train_batches=config.limit_train_batches,callbacks=[EarlyStopping(monitor='val_loss')])

    if config.use_weight:
        weights = make_weights_for_balanced_classes(train_data['label'],config.num_classes)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,len(weights))
        train_data = DataLoader(train_data,batch_size=config.batch_size,shuffle=False,drop_last=True,sampler=sampler)
    else:
        train_data = DataLoader(train_data,batch_size=config.batch_size,shuffle=True,drop_last=True)
    val_data = DataLoader(val_data,batch_size=config.batch_size,shuffle=True,drop_last=True)
    trainer.fit(model, train_data,val_dataloaders=val_data)
