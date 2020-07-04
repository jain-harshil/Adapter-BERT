import logging
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertPreTrainedModel
from transformers import BertModel
from transformers.modeling_bert import BertLayer
from transformers import RobertaModel
from transformers import XLMRobertaConfig
from transformers import RobertaConfig
from transformers import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers import XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

import config as c

class BertForMultichoiceQA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.last_layer_dropout)
        
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), 
                                        nn.Tanh(),
                                        nn.Linear(config.hidden_size, 1, bias = False))
        self.init_weights()

    def forward(self, batch):
        answer_outputs = []
        for i in range(self.num_labels):
            tokids = batch[3 * i]
            att_masks = batch[3 * i + 1]
            tok_type_ids = batch[3 * i + 2]

            outputs = self.bert(input_ids = tokids, attention_mask = att_masks, token_type_ids=tok_type_ids)
            cls_rep = self.dropout(outputs[1])
            
            lt = self.classifier(cls_rep)
            answer_outputs.append(lt)
        
        logits = torch.cat(answer_outputs, dim = 1)
        outputs = (logits, )

        labels = batch[-1]
        loss_func = CrossEntropyLoss()
        loss = loss_func(logits, labels)
        outputs = (loss, ) + outputs
        
        return outputs


##### XLM-R handling large batches

class LargeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.model_gpu_split = c.large_model_gpu_split_layer

        for i, layer_module in enumerate(self.layer):
            if i >= self.model_gpu_split:
                layer_module = layer_module.cuda(1)
            else:
                layer_module = layer_module.cuda(0)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i == self.model_gpu_split:
                hidden_states = hidden_states.cuda(1)
                attention_mask = attention_mask.cuda(1)
                if encoder_hidden_states:
                    encoder_hidden_states = encoder_hidden_states.cuda(1)
                if encoder_attention_mask:
                    encoder_attention_mask = encoder_attention_mask.cuda(1)

            if i >= self.model_gpu_split and head_mask[i]:
                head_mask[i] = head_mask[i].cuda(1)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

#####

class RobertaForMultichoiceQA(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), 
                                        nn.Tanh(),
                                        nn.Linear(config.hidden_size, 1, bias = False))
        
        if hasattr(c, "large_model_gpu_split_layer"):
            self.roberta.to("cuda:0")
            self.roberta.encoder = LargeEncoder(config)
            self.roberta.pooler.to("cuda:1")
            self.dropout = self.dropout.cuda(1)
            self.classifier = self.classifier.cuda(1)

        self.init_weights()

    def forward(self, batch):
        answer_outputs = []
        for i in range(self.num_labels):
            tokids = batch[2 * i]
            att_masks = batch[2 * i + 1]

            outputs = self.roberta(input_ids = tokids, attention_mask = att_masks)

            # for XLM-R large
            sequence_output = self.dropout(outputs[1])
            
            lt = self.classifier(sequence_output)
            answer_outputs.append(lt)
        
        logits = torch.cat(answer_outputs, dim = 1)
        outputs = (logits, )

        labels = batch[-1]
        if hasattr(c, "large_model_gpu_split_layer"):
            labels = labels.cuda(1)

        loss_func = CrossEntropyLoss()

        loss = loss_func(logits, labels)
        
        outputs = (loss, ) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class XLMRobertaForMultichoiceQA(RobertaForMultichoiceQA):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP