
import pandas as pd
from enum import Enum, auto
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import MultiLabelBinarizer
import json
from deprecated import deprecated

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

LABEL_COLS = ['admiration','amusement','disapproval','disgust','embarrassment','excitement','fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'anger', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness','surprise','neutral','annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment']
GOEMOTIONS_LABELS = ['admiration', 'amusement', 'disapproval', 'disgust',
       'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
       'love', 'nervousness', 'anger', 'optimism', 'pride', 'realization',
       'relief', 'remorse', 'sadness', 'surprise', 'neutral', 'annoyance',
       'approval', 'caring', 'confusion', 'curiosity', 'desire',
       'disappointment']

GOEMOTIONS_EKMAN_MAPPING = {
    "ekman_joy": ["admiration", "amusement", "approval", "caring","desire", "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"],
    "ekman_anger": ["anger","annoyance", "disapproval"],
    "ekman_surprise": ["confusion", "curiosity", "surprise","realization"],
    "ekman_sadness": ["disappointment", "embarrassment", "grief", "remorse", "sadness"],
    "ekman_disgust": ["disgust"],
    "ekman_fear": ["fear","nervousness"]
}

# twitter has different labels: joy, sadness, anger, surprise, fear, love
GOEMOTIONS_TWITTER_MAPPING = {
    "twitter_joy": ["admiration", "amusement", "approval","desire", "excitement", "gratitude", "joy", "optimism", "pride", "relief","realization"],
    "twitter_anger": ["anger","annoyance", "disapproval"],
    "twitter_surprise": ["confusion", "curiosity", "surprise"],
    "twitter_sadness": ["disappointment", "embarrassment", "grief", "remorse", "sadness", "disgust"],
    "twitter_love": ["love", "caring"],
    "twitter_fear": ["fear","nervousness"]
}

GOEMOTIONS_EMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]

# execute the or operation on the dataset with the columns specified in the array
def _or(dataset, array):
    # column as long as the dataset with all 0s as a base for or operations
    value = pd.Series([0]*len(dataset))
    for column in array:
        value = value | dataset[column]
    return value 

# taking in input a dataset, returns a new dataset where the emotions of each entry have been mapped according to 'mapping'
def goemotions_apply_emotion_mapping(dataset, drop_original=True, mapping=GOEMOTIONS_EKMAN_MAPPING,isDataframe=True):
    # dataframe == false => we want to map the values of a tensor so we change it into a dataframe first
    if not(isDataframe):
        dataset = pd.DataFrame(dataset,columns=GOEMOTIONS_LABELS)
    for twitter, goemotion in mapping.items():
        dataset[twitter] = _or(dataset, goemotion)
    if drop_original:
        # drop goemotion columns
        dataset = dataset.drop(columns=GOEMOTIONS_EMOTIONS)
        # we must drop every entry whose only label was neutral
        if isDataframe:
            dataset = dataset.loc[dataset[mapping.keys()].sum(axis=1) > 0]
        else:
            dataset = dataset.values
    return dataset

def load_goemotions(k_hot_encode=False):
    # Load GoEmotions dataset
    goemotions_train = pd.read_csv(GOEMOTIONS_DATASET_DIR + '/train.tsv', sep='\t', header=None, index_col=False)
    goemotions_val = pd.read_csv(GOEMOTIONS_DATASET_DIR + '/dev.tsv', sep='\t', header=None, index_col=False)
    goemotions_test = pd.read_csv(GOEMOTIONS_DATASET_DIR + '/test.tsv', sep='\t', header=None, index_col=False)
    # drop last column (comment id)
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
    goemotions_train = pd.read_csv(TWITTERCLEAN_DATASET_DIR + 'train.tsv', sep='\t', index_col=False)
    goemotions_val = pd.read_csv(TWITTERCLEAN_DATASET_DIR + 'val.tsv', sep='\t', index_col=False)
    goemotions_test = pd.read_csv(TWITTERCLEAN_DATASET_DIR + 'test.tsv', sep='\t', index_col=False)
    return goemotions_train, goemotions_val, goemotions_test

DATA_LOADERS = {
    DatasetEnum.GoEmotions: load_goemotions,
    DatasetEnum.TwitterData: load_twitter_data,
    DatasetEnum.GoEmotionsCleaned: load_goemotions_cleaned,
    DatasetEnum.TwitterDataCleaned: load_twitter_data_cleaned
}

def load_dataset(dataset: DatasetEnum, **kwargs):
    return DATA_LOADERS[dataset](**kwargs)

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
        # normalize whitespace
        self.text = self.text.apply(lambda x: " ".join(str(x).split())).tolist()
        # tokenize text
        inputs = self.tokenizer.batch_encode_plus(
            self.text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=self.truncation,
            padding='max_length' if max_len is not None else 'longest',
            return_token_type_ids=True
        )
        self.ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        self.mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        self.token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)
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
    
def create_data_loader_from_dataframe(dataframe, tokenizer, max_len=None, truncation=True, batch_size=8, shuffle=True, num_workers=0):
    ds = EmotionsData(
        dataframe=dataframe,
        tokenizer=tokenizer,
        max_len=max_len,
        truncation=truncation
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

def create_data_loader(dataset: DatasetEnum, tokenizer, max_len, dataset_loader_param_dict={}, batch_size=8, shuffle=True, num_workers=0):
    train, val, test = load_dataset(dataset, **dataset_loader_param_dict)
    train_loader = create_data_loader_from_dataframe(train, tokenizer, max_len, batch_size, shuffle, num_workers)
    val_loader = create_data_loader_from_dataframe(val, tokenizer, max_len, batch_size, shuffle, num_workers)
    test_loader = create_data_loader_from_dataframe(test, tokenizer, max_len, batch_size, shuffle, num_workers)
    return train_loader, val_loader, test_loader

# separate class because no tokenization is needed
class Llama_EmotionsData(Dataset):
    def __init__(self, dataframe) -> None:
        self.text = dataframe['text']
        self.targets = dataframe.drop(columns=['text']).to_numpy()
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index], self.targets[index]