
import pandas as pd
from enum import Enum, auto
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import MultiLabelBinarizer

# enum of datasets
class DatasetEnum(Enum):
    GoEmotions = auto()
    TwitterData = auto()

DATASET_DIR = 'dataset/'
GOEMOTIONS_DATASET_DIR = DATASET_DIR + 'GoEmotionsSplit/'
TWITTER_DATASET_DIR = DATASET_DIR + 'TwitterDataSplit/'

def load_goemotions():
    # Load GoEmotions dataset
    goemotions_train = pd.read_csv(GOEMOTIONS_DATASET_DIR + '/train.tsv', sep='\t', header=None, index_col=False)
    goemotions_val = pd.read_csv(GOEMOTIONS_DATASET_DIR + '/dev.tsv', sep='\t', header=None, index_col=False)
    goemotions_test = pd.read_csv(GOEMOTIONS_DATASET_DIR + '/test.tsv', sep='\t', header=None, index_col=False)
    # drop last column (tweet id)
    goemotions_train = goemotions_train.drop(goemotions_train.columns[2], axis=1)
    goemotions_val = goemotions_val.drop(goemotions_val.columns[2], axis=1)
    goemotions_test = goemotions_test.drop(goemotions_test.columns[2], axis=1)
    # rename columns
    goemotions_train.columns = ['text', 'emotions']
    goemotions_val.columns = ['text', 'emotions']
    goemotions_test.columns = ['text', 'emotions']
    return goemotions_train, goemotions_val, goemotions_test

def load_twitter_data():
    # Load TwitterData dataset
    goemotions_train = pd.read_csv(TWITTER_DATASET_DIR + '/train.txt', sep=';', header=None, index_col=False)
    goemotions_val = pd.read_csv(TWITTER_DATASET_DIR + '/val.txt', sep=';', header=None, index_col=False)
    goemotions_test = pd.read_csv(TWITTER_DATASET_DIR + '/test.txt', sep=';', header=None, index_col=False)
    # rename columns
    goemotions_train.columns = ['text', 'emotions']
    goemotions_val.columns = ['text', 'emotions']
    goemotions_test.columns = ['text', 'emotions']
    return goemotions_train, goemotions_val, goemotions_test

DATA_LOADERS = {
    DatasetEnum.GoEmotions: load_goemotions,
    DatasetEnum.TwitterData: load_twitter_data
}

def load_dataset(dataset: DatasetEnum):
    return DATA_LOADERS[dataset]()

class EmotionsData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = dataframe['text']
        label_series = dataframe['emotions'].apply(lambda x: x.split(','))
        mlb = MultiLabelBinarizer()
        self.targets = mlb.fit_transform(label_series)
        self.label_order = mlb.classes_
        self.max_len = max_len
        self.nclasses = len(self.label_order)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        # normalize whitespace
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
def create_data_loader_from_dataframe(dataframe, tokenizer, max_len, **loader_params):
    ds = EmotionsData(
        dataframe=dataframe,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        **loader_params
    )

def create_data_loader(dataset: DatasetEnum, tokenizer, max_len, **loader_params):
    train, val, test = load_dataset(dataset)
    train_loader = create_data_loader_from_dataframe(train, tokenizer, max_len, **loader_params)
    val_loader = create_data_loader_from_dataframe(val, tokenizer, max_len, **loader_params)
    test_loader = create_data_loader_from_dataframe(test, tokenizer, max_len, **loader_params)
    return train_loader, val_loader, test_loader