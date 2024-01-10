from langchain.document_loaders import YoutubeLoader, YoutubeAudioLoader
from langchain.chat_models import ChatOpenAI
import configparser
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser

config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')

# # 视频链接
# url = "https://www.youtube.com/shorts/13c99EsNt4M"
# # 视频保存路径
# save_dir = "docs/youtube/"
#
# loader = GenericLoader(
#     YoutubeAudioLoader([url], save_dir),
#     OpenAIWhisperParser()
# )
# docs = loader.load()
# print(docs[0].page_content)

from langchain.document_loaders import WebBaseLoader

url = "https://mp.weixin.qq.com/s/RhzHa1oMd0WHk0JamdfVRA"

# 创建webLoader
loader = WebBaseLoader(url)

# 获取文档
docs = loader.load()
# 查看文档内容
text = docs[0].page_content
text = text.replace("\n", '')
print(text)

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=26,  # 块长度
    chunk_overlap=4,  # 重叠字符串长度
    separators=["\n\n", "\n", " ", ""]
)

trunks = r_splitter.split_text(text)
for trunk in trunks:
    print(trunk)

# token_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=1)
# trunks = token_splitter.split_text(text)
# for trunk in trunks:
#     print(trunk)
