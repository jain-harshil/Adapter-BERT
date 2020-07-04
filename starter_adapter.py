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

logger = logging.getLogger(__name__)


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

c = RunConfig()
bertonja = BertModel.from_pretrained(c.pretrained_transformer)

train_data = torch.load(c.train_set)
eval_data = torch.load(c.val_set)

if not os.path.exists(c.output_dir):
    os.makedirs(c.output_dir)

# Set seed
trainer.set_seed(c.seed)

# Loading tokenizer, configuration, and model
tokenizer = BertTokenizer.from_pretrained(c.pretrained_transformer)

match = [f for f in os.listdir(c.output_dir) if f.startswith("best")]
if len(match) > 0:
    c.model_name_or_path = os.path.join(c.output_dir, 'best')

#config = ParallelAdapterBertConfig.from_pretrained(c.model_name_or_path)
config = BottleneckAdapterBertConfig.from_pretrained(c.model_name_or_path)

if len(match) == 0:
    config.layers_to_adapt = c.layers_to_adapt
    config.num_labels = 9

if type(config) is ParallelAdapterBertConfig:
    config.adjust()

#model = ParallelAdapterBertForSequenceClassification.from_pretrained(c.model_name_or_path, config=config)
model = AdapterBertForSequenceClassification.from_pretrained(c.model_name_or_path, config=config)
model.to(c.device)

logger.info("Training/evaluation starts...")
params = {"task_type" : "nli", "model_params" : {}, "mcqa_config" : c}

_, _, eval_perf = trainer.train(train_data, eval_data, model, params)
