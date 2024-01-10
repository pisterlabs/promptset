"""
@Time    : 2023/12/29 18:25
@Author  : yangzq80@gmail.com
@File    : text_embedding_bge.py
"""

from langchain.embeddings import HuggingFaceBgeEmbeddings

def main():
    print('Hello, World!')

    model_name = "/home/ubuntu/yzq/models/bge-large-zh"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="为这个句子生成表示以用于检索相关文章："
    )
    model.query_instruction = "为这个句子生成表示以用于检索相关文章："

    print(model.aembed_query('句话'))
    

if __name__ == '__main__':
    main()



