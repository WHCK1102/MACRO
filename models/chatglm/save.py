import json
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm

def predict(test_set_path, pre_peft_save_path, post_peft_save_path):
    peft_model_path = '/home/yz/here/code/chatglm/data/yz-r100/moocdata13/chatGLM_6B'
    
    config = PeftConfig.from_pretrained(peft_model_path)
    q_config = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.float32)

    base_model = AutoModel.from_pretrained(config.base_model_name_or_path,
                                           quantization_config=q_config,
                                           trust_remote_code=True,
                                           device_map='auto')

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)

    # 读取JSON文件
    with open(test_set_path, 'r', encoding='utf-8') as file:
        test_set = json.load(file)

    # 对每个条目进行预测，并更新微调前的结果到测试集中
    #for entry in tqdm(test_set, desc="Predicting (Pre-Peft)"):
    #    input_text = entry['input']
    #    response, history = base_model.chat(tokenizer=tokenizer, query=input_text)
    #    # 将原始的output字段值移动到ground truth字段
    #    if 'output' in entry:
    #        entry['ground truth'] = entry.pop('output')
    #    # 将预测结果添加到output字段
    #    entry['output'] = response
#
    # 保存微调前的结果到新的文件
    #with open(pre_peft_save_path, 'w', encoding='utf-8') as file:
    #    json.dump(test_set, file, ensure_ascii=False, indent=4)
#
    # 加载Peft模型
    model = PeftModel.from_pretrained(base_model, peft_model_path)

    # 对每个条目进行预测，并更新微调后的结果到测试集中
    for entry in tqdm(test_set, desc="Predict"):
        input_text = entry['input']
        response, history = model.chat(tokenizer=tokenizer, query=input_text)
        if 'output' in entry:
            entry['ground truth'] = entry.pop('output')
        # 更新output字段为预测结果
        entry['output'] = response
        
    # 保存微调后的结果到新的文件
    with open(post_peft_save_path, 'w', encoding='utf-8') as file:
        json.dump(test_set, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # 测试集JSON文件的路径
    test_set_path = '/home/yz/here/code/chatglm/data/yz-r100/moocdata13/moocdata13-test.json'
    # 微调前结果保存路径
    pre_peft_save_path = '/home/yz/here/code/chatglm/data/yz/moocdata13/moocdata13-test-before.json'
    # 微调后结果保存路径
    post_peft_save_path = '/home/yz/here/code/chatglm/data/yz-r100/moocdata13/moocdata13-test-after.json'
    predict(test_set_path, pre_peft_save_path, post_peft_save_path)
