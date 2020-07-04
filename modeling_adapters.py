import torch
from torch import nn
from transformers.modeling_bert import BertIntermediate, BertOutput, BertLayer 
from transformers.modeling_bert import BertEncoder, BertModel, BertForSequenceClassification 
from transformers import RobertaModel, RobertaForSequenceClassification
from transformers import XLMRobertaModel, XLMRobertaForSequenceClassification
#from .modeling_biaffine import BertForBiaffineParsing, RobertaForBiaffineParsing, XLMRobertaForBiaffineParsing
from modeling_mcqa import BertForMultichoiceQA, RobertaForMultichoiceQA, XLMRobertaForMultichoiceQA
#from .modeling_mlm import BertForDynamicMLM, RobertaForDynamicMLM, XLMRobertaForDynamicMLM
from transformers import XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers import XLMRobertaModel
from transformers import XLMRobertaConfig
from transformers import BertConfig, RobertaConfig

def get_nonlin_func(nonlin):
    if nonlin == "tanh":
        return torch.tanh
    elif nonlin == "relu":
        return torch.relu
    elif nonlin == "gelu":
        return nn.functional.gelu
    elif nonlin == "sigmoid":
        return torch.sigmoid
    else:
        raise ValueError("Unsupported nonlinearity!")

### Bottleneck Adapter

class BottleneckAdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.adapter_input_size = config.hidden_size
        self.adapter_latent_size = config.adapter_latent_size
        self.non_linearity = get_nonlin_func(config.adapter_non_linearity)
        self.residual = config.adapter_residual

        # down projection
        self.down_proj = nn.Linear(self.adapter_input_size, self.adapter_latent_size)
        # up projection
        self.up_proj = nn.Linear(self.adapter_latent_size, self.adapter_input_size)

        self.init_weights()

    def init_weights(self):
        """ Initialize the weights -> so that initially we the whole Adapter layer is a near-identity function """
        self.down_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.down_proj.bias.data.zero_()
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.bias.data.zero_()

    def forward(self, x):
        output = self.up_proj(self.non_linearity(self.down_proj(x)))
        if self.residual:
            output = x + output
        return output

### BERT

class AdapterBertConfig(BertConfig):
    def __init__(self,
        layers_to_adapt = list(range(12)),
        adapter_non_linearity = "gelu",
        adapter_latent_size = 64,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.layers_to_adapt = layers_to_adapt
        self.adapter_latent_size = adapter_latent_size
        self.adapter_non_linearity = adapter_non_linearity

class BottleneckAdapterBertConfig(AdapterBertConfig):
    def __init__(self,
        adapter_residual = True,
        add_intermediate_adapter = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.adapter_residual = adapter_residual
        self.add_intermediate_adapter = add_intermediate_adapter

class AdapterBertIntermediate(BertIntermediate):
    def __init__(self, config, layer_index):
        super().__init__(config)
        self.add_adapter = layer_index in config.layers_to_adapt and config.add_intermediate_adapter
        if self.add_adapter:
            self.intermediate_adapter = BottleneckAdapterLayer(config)

    def forward(self, hidden_states):
        # adapter extension
        if self.add_adapter:
            hidden_states = self.intermediate_adapter(hidden_states)

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class AdapterBertOutput(BertOutput):
    def __init__(self, config, layer_index):
        super().__init__(config)
        self.add_adapter = layer_index in config.layers_to_adapt
        if self.add_adapter:
            self.output_adapter = BottleneckAdapterLayer(config)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # adapter extension
        if self.add_adapter:
            hidden_states = self.output_adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class AdapterBertLayer(BertLayer):
    def __init__(self, config, layer_index):
        super().__init__(config)        
        self.intermediate = AdapterBertIntermediate(config, layer_index)
        self.output = AdapterBertOutput(config, layer_index)

class AdapterBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([AdapterBertLayer(config, i) for i in range(config.num_hidden_layers)])

class AdapterBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = AdapterBertEncoder(config)
        self.freeze_original_params(config)

    def freeze_original_params(self, config):
        for param in self.parameters():
            param.requires_grad = False

        for i in range(config.num_hidden_layers):
            if i in config.layers_to_adapt:
                for param in self.encoder.layer[i].intermediate.intermediate_adapter.parameters():
                    param.requires_grad = True
                for param in self.encoder.layer[i].output.output_adapter.parameters():
                    param.requires_grad = True
    
    def unfreeze_original_params(self, config):
        for param in self.parameters():
            param.requires_grad = True

class AdapterBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = AdapterBertModel(config)

# class AdapterBertForBiaffineParsing(BertForBiaffineParsing):
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert = AdapterBertModel(config)

class AdapterBertForMultichoiceQA(BertForMultichoiceQA):
    def __init__(self, config):
        super().__init__(config)
        self.bert = AdapterBertModel(config)

# class AdapterBertForDynamicMLM(BertForDynamicMLM):
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert = AdapterBertModel(config)

### RoBERTa

class AdapterRobertaConfig(RobertaConfig):
    def __init__(self,
        layers_to_adapt = list(range(12)),
        adapter_non_linearity = "gelu",
        adapter_latent_size = 64,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.layers_to_adapt = layers_to_adapt
        self.adapter_latent_size = adapter_latent_size
        self.adapter_non_linearity = adapter_non_linearity

class BottleneckAdapterRobertaConfig(AdapterRobertaConfig):
    def __init__(self,
        adapter_residual = True,
        add_intermediate_adapter = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.adapter_residual = adapter_residual
        self.add_intermediate_adapter = add_intermediate_adapter

class AdapterRobertaModel(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = AdapterBertEncoder(config)
        self.freeze_original_params(config)

    def freeze_original_params(self, config):
        for param in self.parameters():
            param.requires_grad = False

        for i in range(config.num_hidden_layers):
            if i in config.layers_to_adapt:
                for param in self.encoder.layer[i].intermediate.intermediate_adapter.parameters():
                    param.requires_grad = True
                for param in self.encoder.layer[i].output.output_adapter.parameters():
                    param.requires_grad = True
    
    def unfreeze_original_params(self, config):
        for param in self.parameters():
            param.requires_grad = True

class AdapterRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = AdapterRobertaModel(config)

# class AdapterRobertaForBiaffineParsing(RobertaForBiaffineParsing):
#     def __init__(self, config):
#         super().__init__(config)
#         self.roberta = AdapterRobertaModel(config)

class AdapterRobertaForMultichoiceQA(RobertaForMultichoiceQA):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = AdapterRobertaModel(config)

# class AdapterRobertaForDynamicMLM(RobertaForDynamicMLM):
#     def __init__(self, config):
#         super().__init__(config)
#         self.roberta = AdapterRobertaModel(config)

### XLM-R
class BottleneckAdapterXLMRobertaConfig(BottleneckAdapterRobertaConfig):
    config_class = XLMRobertaConfig
    pretrained_config_archive_map = XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP

class AdapterXLMRobertaModel(AdapterRobertaModel):
    config_class = XLMRobertaConfig
    pretrained_config_archive_map = XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP

class AdapterXLMRobertaForSequenceClassification(AdapterRobertaForSequenceClassification):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

# class AdapterXLMRobertaForBiaffineParsing(AdapterRobertaForBiaffineParsing):
#     config_class = XLMRobertaConfig
#     pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

class AdapterXLMRobertaForMultichoiceQA(AdapterRobertaForMultichoiceQA):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

# class AdapterXLMRobertaForDynamicMLM(AdapterRobertaForDynamicMLM):
#     config_class = XLMRobertaConfig
#     pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP