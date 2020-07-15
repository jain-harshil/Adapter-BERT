from transformers.modeling_bert import BertConfig
import copy
        
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

class ParallelAdapterBertConfig(AdapterBertConfig):
    def __init__(self,
        parabert_num_attention_heads=4,
        parabert_intermediate_size=256,
        enforce_orthogonality = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.parabert_num_attention_heads = parabert_num_attention_heads
        self.parabert_intermediate_size = parabert_intermediate_size
        self.enforce_orthogonality = enforce_orthogonality
    
    def adjust(self):
        self.parabert_config = copy.deepcopy(self)
        self.parabert_config.hidden_act = self.adapter_non_linearity
        self.parabert_config.hidden_size = self.adapter_latent_size
        self.parabert_config.num_hidden_layers = len(self.layers_to_adapt)
        self.parabert_config.num_attention_heads = self.parabert_num_attention_heads
        self.parabert_config.intermediate_size = self.parabert_intermediate_size
    
    def save_pretrained(self, save_directory):
        self.parabert_config = None
        del self.parabert_config

        super().save_pretrained(save_directory)

class RunConfig():
    def __init__(self):

        # pretrained transformer model, supported one listed below
        self.pretrained_transformer = 'bert-base-multilingual-cased'
    
        # ### training setup 
        # self.model_name_or_path = self.pretrained_transformer #"/work/gglavas/models/siqa_copa/siqa_pretrained/mbert_cased/5/best" # self.pretrained_transformer
        # ##
        # self.train_set = "/content/drive/My Drive/Adapter_Code/datasets/train_conceptnet_mbert.td"
        # self.val_set = "/content/drive/My Drive/Adapter_Code/datasets/dev_conceptnet_mbert.td"
        # self.output_dir = "/content/drive/My Drive/Adapter_Code/datasets/conceptnet"

        ### training setup 
        self.model_name_or_path = "/content/drive/My Drive/Adapter_Code/datasets/conceptnet/best" # self.pretrained_transformer
        ##
        self.train_set = "/content/drive/My Drive/Adapter_Code/datasets/train_siqa.td"
        self.val_set = "/content/drive/My Drive/Adapter_Code/datasets/test_siqa.td"
        self.output_dir = "/content/drive/My Drive/Adapter_Code/datasets/siqa_finetune"

        self.train_batch_size = 8
        self.learning_rate = 3e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 10
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = 5000
        self.seed = 42
        self.device = "cuda"
        self.num_evals_early_stop = 10
        self.eval_stop_metric = "Accuracy"
        self.eval_metric_increasing = True
        self.last_layer_dropout = 0.2
        self.hidden_size = 100

        # # adapter params
        self.layers_to_adapt = list(range(8, 12))
        # self.adapter_latent_size = 64
        # self.adapter_non_linearity = "gelu"
        # self.adapter_residual = True
        # self.add_intermediate_adapter = True