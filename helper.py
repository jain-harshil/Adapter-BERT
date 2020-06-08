import codecs
from transformers.data.processors import InputFeatures
import torch

def write_list(path, list):
	f = codecs.open(path,'w',encoding='utf8')
	for l in list:
		f.write(l + "\n")
	f.close()

def load_lines(filepath):
	return [l.strip() for l in list(codecs.open(filepath, "r", encoding = 'utf8', errors = 'replace').readlines())]

def featurize_texts(texts, tokenizer, max_length = 128, add_special_tokens = True, is_text_pair = False, label = None, has_toktype_ids = True):
    instances = []
    for text in texts:
        feats = featurize_text(text, tokenizer, max_length, add_special_tokens, is_text_pair=is_text_pair, label = None, has_toktype_ids = has_toktype_ids)
        instances.append(feats)
    
    token_ids = torch.tensor([x.input_ids for x in instances], dtype = torch.long)
    attention_mask = torch.tensor([x.attention_mask for x in instances], dtype = torch.long)
    token_type_ids = torch.tensor([x.token_type_ids for x in instances], dtype = torch.long)

    input_dict = {"input_ids" : token_ids, "attention_mask" : attention_mask, "token_type_ids" : token_type_ids}
    return input_dict

def featurize_text(text, tokenizer, max_length = 128, add_special_tokens = True, is_text_pair = False, label = None, has_toktype_ids = True):
    if is_text_pair:
        text1, text2 = text
        inputs = tokenizer.encode_plus(text1, text2, add_special_tokens=True, max_length=max_length)
    else:
        inputs = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if has_toktype_ids:
        token_type_ids = inputs["token_type_ids"]
    
    # Zero-pad up to the sequence length.
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    padding_length = max_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    if has_toktype_ids:
        token_type_ids = token_type_ids + ([0] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)

    if has_toktype_ids:
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

    return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids if has_toktype_ids else None, label = label)
