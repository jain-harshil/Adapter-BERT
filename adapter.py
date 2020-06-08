import torch
from torch import nn
from transformers.modeling_bert import BertIntermediate, BertOutput, BertLayer, BertEncoder, BertModel, BertForSequenceClassification 

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
        self.bert.unfreeze_original_params(config)

### Parallel Adapter

class ParallelAdapterBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)

        # parallel, adapter-BERT
        self.parabert = BertModel(config.parabert_config)

        # freezing the pre-trained BERT
        self.freeze_original_params()
    
    def freeze_original_params(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.parabert.parameters():
            param.requires_grad = True

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        ):
        
        outputs_main = super().forward(input_ids, attention_mask, token_type_ids)
        outputs_adapter = self.parabert(input_ids, attention_mask, token_type_ids)
        
        outs_cls = []
        outs_cls.append(outputs_main[1])
        outs_cls.append(outputs_adapter[1])
        concat_cls = torch.cat(outs_cls, dim = 1)

        outs_tok = []
        outs_tok.append(outputs_main[0])
        outs_tok.append(outputs_adapter[0])
        concat_tok = torch.cat(outs_tok, dim = 2)

        outputs = (concat_tok, concat_cls)
        return outputs

class ParallelAdapterBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = ParallelAdapterBertModel(config)
        self.classifier = nn.Linear(config.hidden_size + config.parabert_config.hidden_size, self.config.num_labels)
        
### XLM-R