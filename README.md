# SelRepair
SelRepair is an automatic program tool based on dual RAG and fine-tuned LLM.
## Requirements
accelerate 0.27.2 

codebleu 0.6.1

transformers 4.41.2

tree-sitter 0.21.3

pytorch 2.0.1

## Data Processing
The data scource refers to [LLMC-APR project](https://github.com/LLMC-APR/STUDY/tree/main/Dataset) and the corresponding [paper](https://ieeexplore.ieee.org/abstract/document/10298532/). The dataset is propoed by [Tufano et al.](https://dl.acm.org/doi/abs/10.1145/3340544). It is processed as follows:

1. AST generation and traversal
```shell
python AST_extraction_unixcoder.py
```

2.Code & AST Embedding by [Unixcoder](https://huggingface.co/microsoft/unixcoder-base)
```shell
python unixcoder_embedding.py
```

3. Hybrid Retriever
```shell
python RAG_construction.py
```
The processed dataset can be downlaoded from [Zenodo](https://zenodo.org/records/13752229)

## Fine-tuning and Validation
1.Fine-tuning
```shell
accelerate launch finetune_trainer_seletion_rag_50.py
accelerate launch finetune_trainer_seletion_rag_50-100.py
```

2.Validation
```shell
python rag_evaluation_50.py

python rag_evaluation_50-100.py
```
