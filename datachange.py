import datasets
from tokenizers import Tokenizer
import pandas as pd
from pathlib import Path
import pandas as pd

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

def load_match_data(tokenzier:Tokenizer, max_length:int=512,random_seed:int=42):
    """
    Load the data and return it as a tuple.
    """
    data = load_files_to_dict(Path("data/match/train"))
    data = datasets.Dataset.from_list(data)
    data.set_format('pt')
    data = data.map(lambda x: tokenzier(x['text'],truncation=True,max_length=max_length,padding="max_length"),batched=True)
    data = data.train_test_split(test_size=0.2,seed=random_seed)
    return data['train'], data['test']

def load_bigdata_to_dict(filepath:Path):
    if filepath.is_dir():
        files = filepath.glob("*.csv")
        total = []
        for file in files:
            temp_df = pd.read_csv(file)
            temp_df.rename(columns={'攻击标签':'label','全文本':'text'},inplace=True)
            total.append(temp_df)
    total = pd.concat(total)
    return total

def load_bigdata(tokenzier:Tokenizer, max_length:int=512,random_seed:int=42):
    """
    Load the data and return it as a tuple.
    """
    cache_path = Path(f"bigdata/data-{random_seed}-{max_length}")
    if cache_path.exists():
        data = datasets.load_from_disk(cache_path)
    else:
        data = load_bigdata_to_dict(Path("bigdata"))
        data = datasets.Dataset.from_pandas(data)
        data.set_format('pt')
        data = data.map(lambda x: tokenzier(x['text'],truncation=True,max_length=max_length,padding="max_length"),batched=True)
        data = data.train_test_split(test_size=0.2,seed=random_seed)
        data.save_to_disk(cache_path)
    return data['train'], data['test']

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("codeBERTa")
    
    
    data_train,data_test = load_bigdata(tokenizer)
    from collections import Counter
    print(Counter(data_train['label']))