import json
from sklearn.metrics.pairwise import cosine_similarity
import heapq





def get_smallest_indices(lst, n):
    smallest_nums = heapq.nsmallest(n, lst)
    smallest_indices = [lst.index(num) for num in smallest_nums]
    return smallest_indices


def get_largest_indices(lst, n):
    indexed_lst = list(enumerate(lst))

    five_largest_with_indices = heapq.nlargest(n, indexed_lst, key=lambda x: x[1])

    largest_indices, largest_similarities = zip(*five_largest_with_indices)
    return largest_indices,largest_similarities

def similarity_computing(target_data):
    distance_list = []  
    with open('Tufano_dataset/datasets/50/src_source_code_unixAST_embedding_1000_codebase.json','r') as f1:
        rag_datas = f1.readlines()
        for rag_data in rag_datas:
            nn = json.loads(rag_data)
            code_similarity = cosine_similarity(target_data['code_embedding'],nn['code_embedding'])[0,0]
            unixAST_similarity = cosine_similarity(target_data['unixcoder_seq_embedding'],nn['unixcoder_seq_embedding'])[0,0]
            similarity = 0.5*code_similarity+0.5*unixAST_similarity
            distance_list.append(similarity)
    nearest_indices,nearest_similarities = get_largest_indices(distance_list,5) #选择最相似的五条数据
    target_data['retrival_index'] = nearest_indices
    target_data['retrival_similarity'] = nearest_similarities
    return target_data

def similarity_computing_100(target_data):
    distance_list = []  
    with open('Tufano_dataset/datasets/50-100/src_source_code_unixAST_embedding_1000_codebase.json','r') as f1:
        rag_datas = f1.readlines()
        for rag_data in rag_datas:
            nn = json.loads(rag_data)
            code_similarity = cosine_similarity(target_data['code_embedding'],nn['code_embedding'])[0,0]
            unixAST_similarity = cosine_similarity(target_data['unixcoder_seq_embedding'],nn['unixcoder_seq_embedding'])[0,0]
            similarity = 0.5*code_similarity+0.5*unixAST_similarity
            distance_list.append(similarity)
    nearest_indices,nearest_similarities = get_largest_indices(distance_list,5) #选择最相似的五条数据
    target_data['retrival_index'] = nearest_indices
    target_data['retrival_similarity'] = nearest_similarities
    return target_data



data_type = ['train','val','test']
for i in data_type:
    with open('Tufano_dataset/50/src_source_code_unixAST_embedding_{}_remaining.json'.format(i),'r') as f,open(
        'Tufano_dataset/50/src_source_code_unixAST_embedding_{}_remaining_retrival.json'.format(i),'w') as g:
        datas = f.readlines()
        index = 0
        for data in datas:
            index += 1
            mm = json.loads(data)
            data_including_similarity = similarity_computing(mm)
            
            g.write(json.dumps(data_including_similarity))
            g.write('\n')
            if index % 1000 == 0:
                print(index)

        
for i in data_type:
    with open('Tufano_dataset/50-100/src_source_code_unixAST_embedding_{}_remaining.json'.format(i),'r') as f,open(
        'Tufano_dataset/50-100/src_source_code_unixAST_embedding_{}_remaining_retrival.json'.format(i),'w') as g:
        datas = f.readlines()
        index = 0
        for data in datas:
            index += 1
            mm = json.loads(data)
            data_including_similarity = similarity_computing_100(mm)

            g.write(json.dumps(data_including_similarity))
            g.write('\n')
            if index % 1000 == 0:
                print(index)
