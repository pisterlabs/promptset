from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone  # 向量数据库
import os
from keys import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENV
from langchain.vectorstores import Pinecone

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

directory_path = '.\\my_data'  # 文本数据文件所在的文件夹
pinecone_index_name = "tourism"  # ! put in the name of your pinecone index here

flag_clear_database = True  # 是否需要清空向量数据库
flag_need_to_build_embed_database = True  # 是否需要重新建立向量数据库


def build_embed_database():
    # step1: 将文件夹中的word文档，上传到自己的向量数据库
    data = []

    # 会列出指定目录中的所有文件和子目录，包括隐藏文件，并以列表方式打印
    print(f"开始读取文件夹{directory_path}中的数据")
    print(f"需要处理的文件有：{os.listdir(directory_path)}")

    for filename in os.listdir(directory_path):
        if filename.endswith(".doc") or filename.endswith(".docx"):
            # langchain自带功能，加载word文档
            loader = UnstructuredWordDocumentLoader(f'{directory_path}/{filename}')
            # print(loader)
            data.append(loader.load())
    print(f"共检索到{len(data)} 个文件待处理")

    # 要进行split，设置一个长度上限，因为不可能能把所有的数据都放进去，所以要进行分割
    # Chunking the data into smaller pieces
    # 再用菜刀把文档分隔开，chunk_size就是我们要切多大，建议设置700及以下，因为openai有字数限制，chunk_overlap就是重复上下文多少个字
    # chunk_size的意思是一段文字被切分后的长度，chunk_overlap是指每一段文字的重复部分。这个的设置比较重要，因为与LLM的输入上限有关
    # split后的的texts是一个list，每个元素是一个list，每个text是一个文本块
    print("\n\n开始分割数据")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts = []
    for i in range(len(data)):
        print(f"分割第{i + 1}/{len(data)}条数据")
        texts.append(text_splitter.split_documents(data[i]))
        # print(text_splitter.split_documents(data[i]))

    # Creating embeddings，用的是langchain的openai的embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # initialize pinecone
    # 把数字放进向量数据库，environment填写你的数据库所在的位置，例如useast
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )

    for i in range(len(texts)):
        Pinecone.from_texts([t.page_content for t in texts[i]], embeddings, index_name=pinecone_index_name)
        print(f"第{i + 1}/{len(texts)}个文档的数据已经存储进入向量数据库")

    print("向量数据库构建完成!!!")


def clear_embed_database(index_name):
    # deleting all indexes
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )
    pinecone.Index(index_name).delete(delete_all=True)
    print(f"\n\nDeleted all s in {index_name}\n\n")


if __name__ == '__main__':
    print("flag_clear_database:", flag_clear_database)
    print("flag_need_to_build_embed_database:", flag_need_to_build_embed_database)

    if flag_clear_database:
        clear_embed_database(pinecone_index_name)
    if flag_need_to_build_embed_database:
        build_embed_database()
