import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
import deepspeed
import logging
import numpy as np


import torch
from utils import *

# torch.cuda.empty_cache()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
threshold = 0.9

class BugFixDataset(Dataset):
    def __init__(self, tokenizer, bugs_file, fixes_file, bug_codebase_file,fix_codebase_file,max_length=512,rag_threshold=threshold):  
        self.tokenizer = tokenizer
        self.bug_token = '[BUG]'
        self.fix_token = '[FIX]'
        self.bugs = open(bugs_file, "r").readlines()
        self.fixes = open(fixes_file, "r").readlines()
        self.bug_codebase = open(bug_codebase_file, "r").readlines()
        self.fix_codebase = open(fix_codebase_file, "r").readlines()    #txt文档
        assert len(self.bugs) == len(self.fixes)
        self.max_length = max_length
        self.threshold = rag_threshold

    def __len__(self):
        return len(self.bugs)
    
    def __getitem__(self, idx):
        mm = json.loads(self.bugs[idx])
        buggy_code = ' '.join(mm['code_token'])


        bug = self.bug_token + ' ' + buggy_code.strip()
        fix = self.fixes[idx].strip()
        inputs = bug+ ' ' + self.fix_token + ' ' + fix + tokenizer.eos_token
        outputs = fix + tokenizer.eos_token
        
        for rag_index, rag_simi in zip(mm['retrival_index'],mm['retrival_similarity']): 
            if len(tokenizer.tokenize(inputs)) < self.max_length:
                if rag_simi < self.threshold:
                    break
                nn_bug = json.loads(self.bug_codebase[rag_index])
                codebase_buggy_code = ' '.join(nn_bug['code_token'])
                tmp_inputs = self.bug_token + ' ' + codebase_buggy_code.strip() + self.fix_token + ' ' + self.fix_codebase[rag_index].strip() + ' ' + inputs
                if len(tokenizer.tokenize(tmp_inputs)) >= self.max_length:
                    break
                else:
                    inputs = tmp_inputs

        inputs = tokenizer.encode(inputs,return_tensors='pt',truncation=True,max_length=self.max_length)
        outputs = tokenizer.encode(outputs,return_tensors='pt',truncation=True,max_length=self.max_length)
        labels = torch.cat([torch.zeros(1, inputs.size(1) - outputs.size(1)).fill_(-100).long(), outputs], dim=1)   
        attention_mask = torch.ones(inputs.size())

        inputs = torch.cat([inputs, torch.zeros(1, self.max_length - inputs.size(1)).fill_(tokenizer.pad_token_id).long()],dim=1)
        labels = torch.cat([labels, torch.zeros(1, self.max_length - labels.size(1)).fill_(-100).long()], dim=1)
        attention_mask = torch.cat([attention_mask, torch.zeros(1, self.max_length - attention_mask.size(1))], dim=1)
        return {
            'input_ids': inputs.squeeze(),
            'attention_mask': attention_mask.squeeze(),
            'labels': labels.squeeze()
        }
    

      
model_name = "/bigcode/starcoder2-7b"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if 'checkpoint' not in model_name:
    logging.info("adding special tokens")
    new_tokens = {'additional_special_tokens': ['[FIX]','[BUG]'],'pad_token': '[PAD]'}
    tokenizer.add_special_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

logging.info("Loading datasets...")
train_dataset = BugFixDataset(tokenizer, "./Tufano_dataset/datasets/50/src_source_code_unixAST_embedding_train_remaining_retrival.json", "./Tufano_dataset/datasets/50/tgt-train_remaining.txt",'Tufano_dataset/datasets/50/src_source_code_unixAST_embedding_1000_codebase.json','Tufano_dataset/datasets/50/tgt_1000_codebase.txt',)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)





training_args = TrainingArguments(
    output_dir='./starcoder2_rag_results/model_{}'.format(threshold),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_dir='./starcoder2_rag_results/logs',
    logging_steps=3,     
    save_strategy='epoch',      #
    bf16=True,
    deepspeed='./ds_config_trainer.json',
    dataloader_pin_memory=True,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    learning_rate=5e-5,
    warmup_ratio=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

logger.info("Start training...")
trainer.train()
logger.info("Training completed...")


