from transformers import AutoModelForCausalLM, AutoTokenizer
from codebleu import calc_codebleu
import torch
import json
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
import time


checkpoint = "./starcoder2_rag_results/model_0.8/checkpoint"
device = "cuda:0" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint,torch_dtype=torch.bfloat16).to(device)



em_correction = []
code_bleus = []
bleus4_2 = []
lengths = []
start_time = time.time()
with open ("./Tufano_dataset/datasets/50-100/src_source_code_unixAST_embedding_test_remaining_retrival.json",'r') as f1, open(
    "./Tufano_dataset/datasets/50-100/tgt-test_remaining.txt",'r') as f2,open(
        './Tufano_dataset/datasets/50-100/src_source_code_unixAST_embedding_1000_codebase.json','r') as f3,open(
            './Tufano_dataset/datasets/50-100/tgt_1000_codebase.txt','r') as f4:

    buggy_codes = f1.readlines()
    fixed_codes = f2.readlines()
    bug_codebase = f3.readlines()
    fix_codebase = f4.readlines()
    simi_threshold = 0.8
    max_length = 1024
    index = 0
    for buggy_info, fixed_code in zip(buggy_codes,fixed_codes):
        mm = json.loads(buggy_info)
        index += 1
        print(index)
        buggy_code = ' '.join(mm['code_token'])
        inputs = '[BUG] ' + buggy_code.strip() + ' [FIX]'
        for rag_index, rag_simi in zip(mm['retrival_index'],mm['retrival_similarity']):
            if len(tokenizer.tokenize(inputs)) < max_length:
                if rag_simi < simi_threshold:
                    break
                nn_bug = json.loads(bug_codebase[rag_index])
                codebase_buggy_code = ' '.join(nn_bug['code_token'])
                tmp_inputs = '[BUG]' + ' ' + codebase_buggy_code.strip() + ' [FIX]' + ' ' + fix_codebase[rag_index].strip() + ' ' + inputs
                if len(tokenizer.tokenize(tmp_inputs)) >= max_length:
                    break
                else:
                    inputs = tmp_inputs

        lengths.append(len(tokenizer.tokenize(inputs)))
        inputs_ids = tokenizer.encode(inputs,max_length=1024,truncation=True,return_tensors='pt').to(device)
        labels_ids = tokenizer.encode(fixed_code, max_length=120,truncation=True,return_tensors='pt').to(device)
        outputs = model.generate(inputs_ids,max_length=1024)
        outputs_code = tokenizer.decode(outputs[0][len(inputs_ids[0]):],skip_special_tokens=True)

        # print(buggy_code)
        # print(fixed_code)
        # print(outputs_code)
        # print(fixed_code.replace(' ','').replace('\n',''))
        # print(outputs_code.replace(' ',''))
        result = calc_codebleu([fixed_code],[outputs_code],lang="java",weights=(0.25,0.25,0.25,0.25))
        code_bleus.append(result['codebleu'])
        bleu4_2 = sentence_bleu([fixed_code.split(' ')],outputs_code.split(' '))
        bleus4_2.append(bleu4_2)
        if fixed_code.replace(' ','').replace('\n','') == outputs_code.replace(' ','').replace('\n',''):
            em_correction.append(index)
end_time = time.time()
print(em_correction)
print(len(em_correction))
print(len(em_correction)/len(buggy_codes))
print(sum(code_bleus)/len(buggy_codes))
print(sum(bleus4_2)/len(buggy_codes))
print(sum(lengths)/len(buggy_codes))
print(end_time-start_time)

