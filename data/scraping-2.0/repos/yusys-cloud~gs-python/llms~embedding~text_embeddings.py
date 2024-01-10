"""
@Time    : 2023/12/27 14:12
@Author  : yangzq80@gmail.com
@File    : embeddings.py
"""
# 使用python加载 model  TheBloke/vicuna-7B-1.1-HF

from langchain.embeddings import HuggingFaceBgeEmbeddings


model = HuggingFaceBgeEmbeddings(
        model_name='/home/ubuntu/yzq/models/m3e-base',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}, # set True to compute cosine similarity,
        query_instruction="为这个句子生成表示以用于检索相关文章："
    )
    

model.query_instruction = "为这个句子生成表示以用于检索相关文章："

    

def main():
    print('Hello, World!')
    # 嵌入文档
    embeddings = model.embed_documents(
        [
            "Hi there!",
            "Oh, hello!",
            "What's your name?",
            "My friends call me World",
            "Hello World!"
        ]
    )
    print(len(embeddings), len(embeddings[0]))

    # 嵌入查询
    embedded_query = model.embed_query("What was the name mentioned in the conversation?")
    print(embedded_query[:5])

if __name__ == '__main__':
    main()





