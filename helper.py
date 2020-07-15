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

def load(filepath):
    df = pd.read_csv(filepath)
    texts = []
    labels = []
    for i in range (len(df)):
        a = []
        a.append(df.iloc[i]['FirstWord'])
        a.append(df.iloc[i]['SecondWord'])
        texts.append(a)
    for i in range (len(df)):
        labels.append(df.iloc[i]['Relation'])

def featurize_texts(texts, tokenizer, labels, max_length = 128, add_special_tokens = True, is_text_pair = False, has_toktype_ids = True):
    instances = []
    i = 0
    for text in texts:
        feats = featurize_text(text, tokenizer, labels[i],max_length, add_special_tokens, is_text_pair=is_text_pair, has_toktype_ids = has_toktype_ids)
        i = i+1
        instances.append(feats)
    
    token_ids = torch.tensor([x.input_ids for x in instances], dtype = torch.long)
    attention_mask = torch.tensor([x.attention_mask for x in instances], dtype = torch.long)
    token_type_ids = torch.tensor([x.token_type_ids for x in instances], dtype = torch.long)
    label = torch.tensor([x.label for x in instances], dtype = torch.long)


    input_dict = {"input_ids" : token_ids, "attention_mask" : attention_mask, "token_type_ids" : token_type_ids, "label" : label}
    return input_dict

def featurize_texts_siqa(texts, tokenizer, labels, max_length = 128, add_special_tokens = True, is_text_pair = False, has_toktype_ids = True):
    instances = []
    i = 0
    for text in texts:
        feats = featurize_text_siqa(text, tokenizer, labels[i],max_length, add_special_tokens, is_text_pair=is_text_pair, has_toktype_ids = has_toktype_ids)
        i = i+1
        instances.append(feats)
    print("Printing")
    print((instances[0]).input_ids)
    print((instances[1]).input_ids)
    token_ids = torch.tensor([x.input_ids for x in instances], dtype = torch.long)
    attention_mask = torch.tensor([x.attention_mask for x in instances], dtype = torch.long)
    token_type_ids = torch.tensor([x.token_type_ids for x in instances], dtype = torch.long)
    label = torch.tensor([x.label for x in instances], dtype = torch.long)


    input_dict = {"input_ids" : token_ids, "attention_mask" : attention_mask, "token_type_ids" : token_type_ids, "label" : label}
    return input_dict

def featurize_text_siqa(text,tokenizer, label,max_length = 128, add_special_tokens = True, is_text_pair = False, has_toktype_ids = True):
    choices_inputs = []
    for i in range (0,5,2):
        if is_text_pair:
            text1, text2 = text[i],text[i+1]
            inputs = tokenizer.encode_plus(text1, text2, add_special_tokens=True, max_length=max_length,pad_to_max_length=True)
        else:
            inputs = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length,pad_to_max_length=True)
        choices_inputs.append(inputs)
    

    input_ids = [x["input_ids"] for x in choices_inputs]

    attention_mask = ([x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None)
    if has_toktype_ids:
        token_type_ids = ([x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None)

    return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids if has_toktype_ids else None, label = label)

def featurize_text(text, tokenizer, label,max_length = 128, add_special_tokens = True, is_text_pair = False, has_toktype_ids = True):
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