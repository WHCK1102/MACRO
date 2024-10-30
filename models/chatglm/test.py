import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from typing import Tuple, List



import json
from pprint import pprint

def predict(inputs: List[str]) -> List[Tuple[str, str]]:
    peft_model_path = 'code/chatglm/chatGLM_6B_QLoRA-yz.json'

    config = PeftConfig.from_pretrained(peft_model_path)
    q_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.float32)

    base_model = AutoModel.from_pretrained(config.base_model_name_or_path,
                                        quantization_config=q_config,
                                        trust_remote_code=True,
                                        device_map='auto')

    model = PeftModel.from_pretrained(base_model, peft_model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    
    result = list()
    for input_text in inputs:

        response_before, _ = base_model.chat(tokenizer=tokenizer, query=input_text)

        response_after, _ = model.chat(tokenizer=tokenizer, query=input_text)
        
        result.append((response_before, response_after))
    
    return result

from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu

def nll(outputs: List[List[str]]):
    peft_model_path = 'code/chatglm/chatGLM_6B_QLoRA-yz.json'

    config = PeftConfig.from_pretrained(peft_model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    result = list()
    for label, before, after in outputs:
        label_encode = tokenizer.encode(label)
        before_encode = tokenizer.encode(before)
        after_encode = tokenizer.encode(after)
        print(len(label_encode))
        print(len(before_encode))
        print(len(after_encode))
        score_before = sentence_bleu([label_encode], before_encode)
        score_after = sentence_bleu([label_encode], after_encode)
        result.append([score_before, score_after])
    return result


def save_predict():
    sft_data = list()
    with open('data/sftdata-04.jsonl') as sft_data_file:
        line = sft_data_file.readline()
        while line:
            sft_data.append(json.loads(line))
            line = sft_data_file.readline()
    print(len(sft_data))
    response = predict([one['instruction'] for one in sft_data])
    result = list()
    for sft, (response_before, response_after) in zip(sft_data, response):
        result.append({
            'input': sft['instruction'],
            'output': sft['output'],
            'response_before': response_before,
            'response_after': response_after,
        })
    with open('predict-100.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=4, ensure_ascii=False)


from transformers import PreTrainedTokenizer
def chatglm6b_tokenizer() -> PreTrainedTokenizer:
    peft_model_path = 'saved_files/100'
    config = PeftConfig.from_pretrained(peft_model_path)
    return AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)


def encode(from_path, target_path):
    with open(from_path) as file:
        before: list[dict] = json.load(file)
    print(len(before))
    print(before[0].keys())
    tokenizer = chatglm6b_tokenizer()
    for one in before:
        one['output-encode'] = tokenizer.encode(one['output'])
        one['ground truth-encode'] = tokenizer.encode(one['ground truth'])
    with open(target_path, 'w') as file:
        json.dump(before, file, ensure_ascii=False, indent=4)
    

def xxx():
    after_zh_path = '/home/yz/here/code/chatglm/result/yz/MOOCQA13/after_chatglm6b_MOOCQA13.json'
    after_zh_utf8_path = '/home/yz/here/code/chatglm/result/yz/MOOCQA13/after_chatglm6b_MOOCQA13.json'
    
    before_zh_path = '/home/yz/here/code/chatglm/result/yz/MOOCQA13/before_chatglm6b_MOOCQA13.json'
    before_zh_utf8_path = '/home/yz/here/code/chatglm/result/yz/MOOCQA13/before_chatglm6b_MOOCQA13.json'
    
    with open(after_zh_path) as file:
        after_data = json.load(file)
    with open(after_zh_utf8_path, 'w') as file:
        json.dump(after_data, file, ensure_ascii=False, indent=4)
    
    with open(before_zh_path) as file:
        before_data = json.load(file)
    with open(before_zh_utf8_path, 'w') as file:
        json.dump(before_data, file, ensure_ascii=False, indent=4)
        
    print(len(after_data))
    print(len(before_data))


if __name__ == '__main__':
    data = [{
        'name': '你好'
    }]
    
    file_path = '/home/yz/here/code/chatglm/data/yz/MOOCQA13/MOOCQA13-test.json'    
    with open(file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)