import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from typing import Union
import json
import time
from tqdm import tqdm
import os
import argparse


model_path = 'THUDM/chatglm-6b'

def predict(valid_data_path, save_path, adapter_path: Union[str, None]=None):
    config = PeftConfig.from_pretrained(model_path)
    q_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.float32)
    model = AutoModel.from_pretrained(model_path,
                                        quantization_config=q_config,
                                        trust_remote_code=True,
                                        device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)

    if adapter_path:
        status = 'after train'
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        status = 'before train'
        
    result: list[dict] = json.load(open(save_path)) if os.path.exists(save_path) else list()
    
    valid_data: list[dict] = json.load(open(valid_data_path))
        
    pbar = tqdm(range(len(valid_data)), desc=status, initial=len(result))
    
    try:
        for sample in valid_data[len(result):]:
            query = sample['input']
            gt = sample['output']
            response, history = model.chat(tokenizer=tokenizer, query=query)
            result.append({
                'input': query,
                'output': response,
                'ground truth': gt,
            })
            pbar.update()
    finally:
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=4)
        print(f'saved in {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='valid data path')
    parser.add_argument('--output', type=str, required=True, help='save dir')
    args = parser.parse_args()
    
    valid_data_path = args.data
    result_dir = args.output
    
    os.makedirs(result_dir, exist_ok=True)
    
    dir_name, file_name = os.path.split(valid_data_path)
    file_base_name = os.path.splitext(file_name)[0]
    before_path = os.path.join(result_dir, file_base_name + '-before-train-result.json')
    after_path = os.path.join(result_dir, file_base_name + '-after-train-result.json')
    adapter_dir = os.path.join(result_dir, 'adapter')
    
    checkpoints = [filename for filename in os.listdir(adapter_dir) if os.path.isdir(os.path.join(adapter_dir, filename)) and filename.startswith("checkpoint-")]

    adapter_path = os.path.join(adapter_dir, max(checkpoints)) if checkpoints else adapter_dir
    
    print(f'base model: {model_path}')
    print(f'finetuned checkpoint: {adapter_path}')
    
    print(f"inferring {file_name} on {model_path}")
    predict(valid_data_path, before_path)
    print(f"inferring {file_name} on {adapter_path}")
    predict(valid_data_path, after_path, adapter_path)