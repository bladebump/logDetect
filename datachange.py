import datasets
from tokenizers import Tokenizer
import pandas as pd
from pathlib import Path

def load_data(tokenzier:Tokenizer, max_length:int=512):
    """
    Load the data and return it as a tuple.
    """
    data = datasets.load_dataset("text",data_dir="data")
    data = data['train'].map(lambda x: tokenzier.encode(x['text'],truncation=True,max_length=max_length),batched=True)
    trainer_datasets, val_datasets = data.train_test_split(test_size=0.1)
    return trainer_datasets, val_datasets

def load_files_to_dict(filepath:Path):
    if filepath.is_dir():
        label = 0
        files = filepath.glob("*.csv")
        total = []
        for file in files:
            with open(file,'r',encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    total.append({'text':line,'label':label})
            label += 1
    return total

def load_match_data(tokenzier:Tokenizer, max_length:int=512):
    """
    Load the data and return it as a tuple.
    """
    data = load_files_to_dict(Path("data/match/train"))
    data = datasets.Dataset.from_list(data)
    data.set_format('pt')
    data = data.map(lambda x: tokenzier(x['text'],truncation=True,max_length=max_length,padding="max_length"),batched=True)
    data = data.train_test_split(test_size=0.1)
    return data['train'], data['test']

if __name__ == "__main__":
    load_match_data()