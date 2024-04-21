
import pandas as pd
from enum import Enum, auto
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import MultiLabelBinarizer
import json

# enum of datasets
class DatasetEnum(Enum):
    GoEmotions = auto()
    TwitterData = auto()
    GoEmotionsCleaned = auto()
    TwitterDataCleaned = auto()

DATASET_DIR = 'dataset/'
GOEMOTIONS_DATASET_DIR = DATASET_DIR + 'GoEmotionsSplit/'
GOEMOTIONS_LABEL_MAPPING_PATH = GOEMOTIONS_DATASET_DIR + 'label_mapping.json'
TWITTER_DATASET_DIR = DATASET_DIR + 'TwitterDataSplit/'
GOEMOTIONSCLEAN_DATASET_DIR = DATASET_DIR + 'GoEmotionsCleaned/'
TWITTERCLEAN_DATASET_DIR = DATASET_DIR + 'TwitterDataCleaned/'

def load_goemotions(k_hot_encode=False):
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
    if k_hot_encode:
        # binarize emotions
        goemotions_train['emotions'] = goemotions_train['emotions'].apply(lambda x: x.split(','))
        goemotions_val['emotions'] = goemotions_val['emotions'].apply(lambda x: x.split(','))
        goemotions_test['emotions'] = goemotions_test['emotions'].apply(lambda x: x.split(','))
        mlb = MultiLabelBinarizer()
        goemotions_train = goemotions_train.join(pd.DataFrame(mlb.fit_transform(goemotions_train.pop('emotions')),
                                                          columns=mlb.classes_))
        goemotions_val = goemotions_val.join(pd.DataFrame(mlb.fit_transform(goemotions_val.pop('emotions')),
                                                      columns=mlb.classes_))
        goemotions_test = goemotions_test.join(pd.DataFrame(mlb.fit_transform(goemotions_test.pop('emotions')),
                                                        columns=mlb.classes_))
        # apply label mapping
        with open(GOEMOTIONS_LABEL_MAPPING_PATH, 'r') as f:
            label_mapping = json.load(f)
            goemotions_train = goemotions_train.rename(columns=label_mapping)
            goemotions_val = goemotions_val.rename(columns=label_mapping)
            goemotions_test = goemotions_test.rename(columns=label_mapping)
    return goemotions_train, goemotions_val, goemotions_test

def load_twitter_data(k_hot_encode=False):
    # Load TwitterData dataset
    goemotions_train = pd.read_csv(TWITTER_DATASET_DIR + '/train.txt', sep=';', header=None, index_col=False)
    goemotions_val = pd.read_csv(TWITTER_DATASET_DIR + '/val.txt', sep=';', header=None, index_col=False)
    goemotions_test = pd.read_csv(TWITTER_DATASET_DIR + '/test.txt', sep=';', header=None, index_col=False)
    # rename columns
    goemotions_train.columns = ['text', 'emotions']
    goemotions_val.columns = ['text', 'emotions']
    goemotions_test.columns = ['text', 'emotions']
    if k_hot_encode:
        # binarize emotions
        goemotions_train['emotions'] = goemotions_train['emotions'].apply(lambda x: x.split(','))
        goemotions_val['emotions'] = goemotions_val['emotions'].apply(lambda x: x.split(','))
        goemotions_test['emotions'] = goemotions_test['emotions'].apply(lambda x: x.split(','))
        mlb = MultiLabelBinarizer()
        goemotions_train = goemotions_train.join(pd.DataFrame(mlb.fit_transform(goemotions_train.pop('emotions')),
                                                          columns=mlb.classes_))
        goemotions_val = goemotions_val.join(pd.DataFrame(mlb.fit_transform(goemotions_val.pop('emotions')),
                                                      columns=mlb.classes_))
        goemotions_test = goemotions_test.join(pd.DataFrame(mlb.fit_transform(goemotions_test.pop('emotions')),
                                                        columns=mlb.classes_))
    return goemotions_train, goemotions_val, goemotions_test

def load_goemotions_cleaned():
    # Load GoEmotions dataset
    goemotions_train = pd.read_csv(GOEMOTIONSCLEAN_DATASET_DIR + 'train.tsv', sep='\t', index_col=False)
    goemotions_val = pd.read_csv(GOEMOTIONSCLEAN_DATASET_DIR + 'val.tsv', sep='\t', index_col=False)
    goemotions_test = pd.read_csv(GOEMOTIONSCLEAN_DATASET_DIR + 'test.tsv', sep='\t', index_col=False)
    return goemotions_train, goemotions_val, goemotions_test

def load_twitter_data_cleaned():
    # Load TwitterData dataset
    goemotions_train = pd.read_csv(TWITTERCLEAN_DATASET_DIR + 'train.txt', sep='\t', index_col=False)
    goemotions_val = pd.read_csv(TWITTERCLEAN_DATASET_DIR + 'val.txt', sep='\t', index_col=False)
    goemotions_test = pd.read_csv(TWITTERCLEAN_DATASET_DIR + 'test.txt', sep='\t', index_col=False)
    return goemotions_train, goemotions_val, goemotions_test

DATA_LOADERS = {
    DatasetEnum.GoEmotions: load_goemotions,
    DatasetEnum.TwitterData: load_twitter_data,
    DatasetEnum.GoEmotionsCleaned: load_goemotions_cleaned,
    DatasetEnum.TwitterDataCleaned: load_twitter_data_cleaned
}

def load_dataset(dataset: DatasetEnum, **kwargs):
    return DATA_LOADERS[dataset](**kwargs)

def compute_max_tokens(dataframe_list, tokenizer):
    max_len = 0
    for dataframe in dataframe_list:
        for text in dataframe['text']:
            tokens = tokenizer.encode(text)
            max_len = max(max_len, len(tokens))
    return max_len

'''
used for dataset in pytorch, tokenizes text once during initialization (best suited for small datasets)
'''
class EmotionsData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=None, truncation=True):
        self.tokenizer = tokenizer
        self.text = dataframe['text']
        self.targets = dataframe.drop(columns=['text']).to_numpy()
        self.ids = []
        self.mask = []
        self.token_type_ids = []
        if max_len is None:
            self.max_len = max([len(self.tokenizer.encode(text)) for text in self.text])
        else:
            self.max_len = max_len
        self.nclasses = len(self.targets[0])
        self.truncation = truncation
        # tokenize text
        for i, text in enumerate(self.text):
            # normalize whitespace
            text = str(text)
            text = " ".join(text.split())

            inputs = self.tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=self.truncation,
                padding='max_length',
                return_token_type_ids=True
            )
            self.ids.append(inputs['input_ids'])
            self.mask.append(inputs['attention_mask'])
            self.token_type_ids.append(inputs['token_type_ids'])
        self.ids = torch.tensor(self.ids, dtype=torch.long)
        self.mask = torch.tensor(self.mask, dtype=torch.long)
        self.token_type_ids = torch.tensor(self.token_type_ids, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.float)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):

        return {
            'ids': self.ids[index],
            'mask': self.mask[index],
            'token_type_ids': self.token_type_ids[index],
            'targets': self.targets[index]
        }
    
def create_data_loader_from_dataframe(dataframe, tokenizer, max_len=None, truncation=True, **loader_params):
    ds = EmotionsData(
        dataframe=dataframe,
        tokenizer=tokenizer,
        max_len=max_len,
        truncation=truncation
    )

    return DataLoader(
        ds,
        **loader_params
    )

def create_data_loader(dataset: DatasetEnum, tokenizer, max_len, dataset_loader_param_dict={}, **loader_params):
    train, val, test = load_dataset(dataset, **dataset_loader_param_dict)
    train_loader = create_data_loader_from_dataframe(train, tokenizer, max_len, **loader_params)
    val_loader = create_data_loader_from_dataframe(val, tokenizer, max_len, **loader_params)
    test_loader = create_data_loader_from_dataframe(test, tokenizer, max_len, **loader_params)
    return train_loader, val_loader, test_loader