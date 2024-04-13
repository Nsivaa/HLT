
import pandas as pd
from enum import Enum, auto

# enum of datasets
class Dataset(Enum):
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
    Dataset.GoEmotions: load_goemotions,
    Dataset.TwitterData: load_twitter_data
}

def load_dataset(dataset: Dataset):
    return DATA_LOADERS[dataset]()