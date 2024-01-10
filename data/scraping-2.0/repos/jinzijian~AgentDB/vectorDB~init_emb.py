import langchain
from langchain.embeddings import OpenAIEmbeddings
import configparser

# 创建配置解析器
config = configparser.ConfigParser()

# 读取配置文件
config.read('./config/config.ini')

# 获取OpenAI API密钥
api_key = config['openai']['api_key']
embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)

# Todo: 支持多种Embedding; OpenAI; Bge-En; Bge-ZH
def get_embeddings_model(config):
    api_key = config['openai']['api_key']
    embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)
    return embeddings_model
if __name__=="__main__":
    emb_model = get_embeddings_model(api_key)
    embeddings = emb_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
    )
    print(len(embeddings), len(embeddings[0]))
    embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
    print(embedded_query[:5])