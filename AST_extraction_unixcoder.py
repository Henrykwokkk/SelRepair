import json
import os
import pickle
import sys
from tree_sitter import Language, Parser
import glob
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

tokenizer = RobertaTokenizer.from_pretrained('microsoft/unixcoder-base')




parsers={}              
for lang in ['java']:
    LANGUAGE = Language('Tree-sitter/build/java.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parsers[lang] = parser



def tree_to_token_index(root_node):
    if (len(root_node.children)==0 or root_node.type=='string' or root_node.type=='comment' or 'comment' in root_node.type):
        return [(root_node.start_point,root_node.end_point)]
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_token_index(child)
        return code_tokens
    
def tree_to_variable_index(root_node,index_to_code):
    if (len(root_node.children)==0 or root_node.type=='string' or root_node.type=='comment' or 'comment' in root_node.type):
        index=(root_node.start_point,root_node.end_point)
        _,code=index_to_code[index]
        if root_node.type!=code:
            return [(root_node.start_point,root_node.end_point)]
        else:
            return []
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_variable_index(child,index_to_code)
        return code_tokens    

def index_to_code_token(index,code):
    start_point=index[0]
    end_point=index[1]
    if start_point[0]==end_point[0]:
        s=code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s=""
        s+=code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1,end_point[0]):
            s+=" "+code[i]
        s+=" "+code[end_point[0]][:end_point[1]]   
    return s

def travel(root_node,index_to_code,tokenizer):
    """Given a AST node, return AST travel sequence using Algo in the paper: https://arxiv.org/pdf/2203.03850.pdf"""
    if (len(root_node.children) == 0 or root_node.type == 'string' or root_node.type == 'comment' or 'comment' in root_node.type):
        index = (root_node.start_point,root_node.end_point)
        code = index_to_code[index][1]
        return tokenizer.tokenize(code)
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += travel(child,index_to_code,tokenizer)
        # remove nodes that have only one children for reducing length
        if len(root_node.children) != 1:
            return ["AST#" + root_node.type.replace("#","") + "#Left"] + code_tokens + ["AST#" + root_node.type.replace("#","") + "#Right"] 
        else:
            return code_tokens

def AST(code,lang,tokenizer):
    """Given a code, return its AST flatten sequence"""
    # if lang == "php":
    #     code = "<?php "+code+"?>" 
    # remove comment
    # try:
    #     code = remove_comments_and_docstrings(code,lang)
    # except:
    #     pass
    # parse source code
    # if lang == "csharp":
    #     tree = parsers["c_sharp"].parse(bytes(code,'utf8'))    
    # else:
    tree = parsers[lang].parse(bytes(code,'utf8'))  
    
    # obtain AST sequence
    root_node = tree.root_node  
    tokens_index = tree_to_token_index(root_node)     
    code = code.split('\n')
    code_tokens = [index_to_code_token(x,code) for x in tokens_index]  
    index_to_code = {}
    for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
        index_to_code[index] = (idx,code)  

    code_tokens = travel(root_node,index_to_code,tokenizer)
    return code_tokens


data_type = ['train','val','test']
token_length = ['50','50-100']
for j in token_length:
    for i in data_type:
        with open('Tufano_dataset/datasets/{}/src-{}.txt'.format(j,i),'r',encoding='utf-8') as f,open(
            'Tufano_dataset/datasets/{}/src_source_code_unixAST_{}.json'.format(j,i),'w',encoding='utf-8') as g:
            datas = f.readlines()
            num = 0
            for data in datas:
                num += 1
                mm = {}
                data = data.replace('\n','').strip()
                mm['code_token'] = data.split(' ')

                unixcoder_seq = AST(data,'java',tokenizer)
                mm['unixcoder_seq'] = unixcoder_seq
                g.write(json.dumps(mm))
                g.write('\n')
                if num % 1000 == 0:
                    print(num)

