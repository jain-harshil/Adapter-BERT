from helper import *
from transformers.modeling_bert import BertConfig
import copy
from config import *
from modeling_adapters import *
from modeling_mcqa import * 
from adapter import *
from trainer import *
import trainer
from transformers import BertTokenizer, BertConfig, BertModel
from adapter import AdapterBertModel, AdapterBertForSequenceClassification, ParallelAdapterBertForSequenceClassification
import helper
import torch
import numpy as np
import logging
import torch
import os
from config import RunConfig, ParallelAdapterBertConfig, BottleneckAdapterBertConfig
import torch
import sys
from torch.utils.data import TensorDataset, random_split
import pandas as pd

logger = logging.getLogger(__name__)

texts = []
labels = []

def load(filepath):
    df = pd.read_csv(filepath)
    global texts
    global labels
    for i in range (len(df)):
        a = []
        a.append(df.iloc[i]['FirstWord'])
        a.append(df.iloc[i]['SecondWord'])
        texts.append(a)
    for i in range (len(df)):
        labels.append(df.iloc[i]['Relation'])

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

c = RunConfig()

# Loading tokenizer, configuration, and model
tokenizer = BertTokenizer.from_pretrained(c.pretrained_transformer)

filepath = "./datasets/CNItalianDataFiltered.csv"

load(filepath)

finaldict = featurize_texts(texts,tokenizer,labels,max_length = 128,add_special_tokens = True, is_text_pair = False, has_toktype_ids = True)

dataset = TensorDataset(finaldict["input_ids"],finaldict["attention_mask"],finaldict["token_type_ids"],finaldict["label"])

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# torch.save(train_dataset,'train_conceptnet_mbert.td')
# torch.save(val_dataset,'dev_conceptnet_mbert.td')
