import datasets
from tokenizers import Tokenizer
import pandas as pd
from pathlib import Path
import pandas as pd
import torch

label2id = {0:0,1:1,2:1,3:1,4:1,5:1}

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

def load_bigdata_to_dict(filepath:Path,user_mtilabel:bool=True):
    if filepath.is_dir():
        files = filepath.glob("*.csv")
        total = []
        for file in files:
            temp_df = pd.read_csv(file)
            temp_df.rename(columns={'攻击标签':'label','全文本':'text'},inplace=True)
            temp_df['label'] = temp_df['label'].astype(int)
            # remove label 4
            temp_df = temp_df[temp_df['label'] != 4]
            
            # change label 5 to 4
            temp_df['label'] = temp_df['label'].apply(lambda x: 4 if x == 5 else x)
            
            if not user_mtilabel:
                temp_df['label'] = temp_df['label'].apply(lambda x: label2id[x])
            if file.name.startswith("white"):
                # 选取1000000个
                temp_df = temp_df.sample(n=2000000)
            total.append(temp_df)
    total = pd.concat(total)
    return total

def load_bigdata(tokenzier:Tokenizer, max_length:int=512,random_seed:int=42,use_mtilabel:bool=True):
    """
    Load the data and return it as a tuple.
    """
    cache_path = Path(f"bigdata/data-{random_seed}-{max_length}-{use_mtilabel}")
    if cache_path.exists():
        data = datasets.load_from_disk(cache_path)
    else:
        data = load_bigdata_to_dict(Path("bigdata"),user_mtilabel=use_mtilabel)
        data = datasets.Dataset.from_pandas(data)
        data = data.shuffle()

        data.set_format('pt')
        data = data.map(lambda x: tokenzier(x['text'],truncation=True,max_length=max_length,padding="max_length"),batched=True,num_proc=20)
        data = data.train_test_split(test_size=0.2,seed=random_seed)
        data.save_to_disk(cache_path)
    return data['train'], data['test']

def make_weights_for_balanced_classes(labels,num_classes):
    count = [0] * num_classes
    for item in labels:
        count[item] += 1
    weight_per_class = [0.] * num_classes
    N = float(sum(count))
    for i in range(num_classes):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(labels)
    for idx,val in enumerate(labels):
        weight[idx] = weight_per_class[val]
    return weight

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("codeBERTa")
    
    
    data_train,data_test = load_bigdata(tokenizer,use_mtilabel=True)
    
    weights = make_weights_for_balanced_classes(data_train['label'],5)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,len(weights))
    dataloader = torch.utils.data.DataLoader(data_train,sampler=sampler,batch_size=32)
    for batch in dataloader:
        print(batch['label'])
        break