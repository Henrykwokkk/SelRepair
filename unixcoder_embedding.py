import torch
import json
from unixcoder import UniXcoder


model_name = "microsoft/unixcoder-base"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UniXcoder(model_name)
model.to(device)

def embedding(info):
    tokens_ids = model.tokenize([info],max_length=1023,mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)

    tokens_embedding,seq_embedding = model(source_ids)

    return seq_embedding
    
data_type = ['train','val','test']
token_length = ['50','50-100']

for i in data_type:
    for j in token_length:
        with open('Tufano_dataset/{}/src_source_code_unixAST_{}.json'.format(j,i),'r',encoding='utf-8') as f,open(
            'Tufano_dataset/{}/src_source_code_unixAST_embedding_{}.json'.format(j,i),'w',encoding='utf-8') as g:
            datas = f.readlines()
            for index,data in enumerate(datas):
                print(index)
                mm = json.loads(data)
                code = ' '.join(mm['code_token'])
                unixcoder_seq = ' '.join(mm['unixcoder_seq'])

                code_embedding = embedding(code)
                unixcoder_seq_embedding = embedding(unixcoder_seq)
                mm['code_embedding'] = code_embedding.tolist()
                mm['unixcoder_seq_embedding'] = unixcoder_seq_embedding.tolist()
                g.write(json.dumps(mm))
                g.write('\n')

