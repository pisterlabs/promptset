# %%
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# %%
# 定义模型
embeddings = OpenAIEmbeddings(
    model_kwargs = {"model_name": "text-embedding-ada-002"},
    openai_api_key='',
    openai_api_base='https://openai.api2d.net/v1'
)
# embeddings = OpenAIEmbeddings()
# 定义工具
text_splitter = CharacterTextSplitter(
    chunk_size=1,
    chunk_overlap=0,
    separator = '\n'
)

# %%
# 加载文件
with open('../text-embedding-ada-002/keys.txt') as f:
    document = f.read()

# %%
keys = text_splitter.create_documents([document])

# %%
db = FAISS.from_documents(keys, embeddings)

# %%
db.save_local("index")

# %%
