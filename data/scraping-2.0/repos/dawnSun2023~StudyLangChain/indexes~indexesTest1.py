"""
索引
索引组件为LangChain提供了文档处理的能力，包括文档加载、检索等等，这里的文档是广义的文档，不
仅仅是txt、epub、pdf等文本类的内容，还包括email、区块链、telegram、Notion甚至是视频内容。
索引组件主要有以下四种类型：
文档加载器
文本分割器
VectorStores
检索器
"""
import os
os.environ["OPENAI_API_KEY"] = "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"
# 案例一
"""
文档加载器
文档加载器主要基于Unstructured 包，Unstructured 是一个python包，可以把各种类型的文件转换成文本。
文档加载器使用起来很简单，只需要引入相应的loader工具：

文档加载器	介绍
Airbyte JSON	从Airbyte加载JSON，Airbyte是一个数据集成平台
Apify Dataset	Apify Dataset是一个可扩展的纯应用存储，具有顺序访问功能，用于存储结构化的网络抓取结果
Arxiv	arXiv是一个开放性的档案库，收录了物理学、数学、计算机科学、定量生物学、定量金融、统计学、电气工程和系统科学以及经济学等领域的200万篇学术文章。
AWS S3 Directory	AWS S3是Amazon的对象存储服务，加载目录
AWS S3 File	加载文件
AZLyrics	AZLyrics是一个大型的、合法的、每天都在增长的歌词集。
Azure Blob Storage Container	微软的对象存储服务
Azure Blob Storage File	加载文件
Bilibili	加载B站视频
Blackboard	一个虚拟学习环境和学习管理系统
Blockchain	基于alchemy 加载区块链数据
ChatGPT Data	chatGPT消息加载器
College Confidential	提供全世界的大学信息
Confluence	一个专业的企业知识管理与协同平台
CoNLL-U	CoNLL-U是一种文件格式
Copy Paste	普通文本
CSV	CSV文件
Diffbot	一个将网站转化为结构化数据的平台
Discord	
DuckDB	一个分析数据库系统
Email	邮件，支持.eml和.msg格式
EPub	epub电子书
EverNote	EverNote文档
Facebook Chat	Facebook消息
Figma	一个web设计工具
File Directory	加载目录下所有文件
Git	GIt
GitBook	GItBook
Google BigQuery	谷歌云服务
Google Cloud Storage Directory	谷歌云服务
Google Cloud Storage File	谷歌云服务
Google Drive	谷歌云服务
Gutenberg	一个在线电子书平台
Hacker News	一个计算机信息网站
HTML	网页
HuggingFace dataset	HuggingFace数据集
iFixit	一个维修为主题的社区
Images	加载图片
Image captions	根据图片生成图片说明
IMSDb	电影数据库
JSON Files	加载JSON文件
Jupyter Notebook	加载notebook文件
Markdown	加载markdown文件
MediaWikiDump	wiki xml数据
Microsoft OneDrive	加载微软OneDrive文档
Microsoft PowerPoint	加载ppt文件
Microsoft Word	加载word文件
Modern Treasury	一家支付运营软件提供商
Notion DB	加载Notion文件
Obsidian	一个笔记软件
Unstructured ODT Loader	加载OpenOffice文件
Pandas DataFrame	Pandas表格型数据结构
PDF	加载pdf文件
ReadTheDocs Documentation	一个在线文档平台
Reddit	一个社交新闻网站
Roam	一个个人笔记产品
Sitemap	网站地图
Slack	一个聊天群组产品
Spreedly	一个支付平台
Stripe	一个支付平台
Subtitle	一个字幕制作平台
Telegram	聊天软件
TOML	一种配置文件
Twitter	
Unstructured File	Unstructured文件
URL	通过url加载内容
WebBaseLoader	基础的web加载器
WhatsApp Chat	WhatsApp聊天
Wikipedia	加载Wikipedia内容
YouTube transcripts	加载YouTube视频
"""
from langchain.document_loaders import TextLoader

loader = TextLoader("../file/test.txt", encoding='utf-8')
documents = loader.load()
print(loader)
print(documents)
print("------------------------------------------------------------------------------------------------------------")

"""
案例二、文本分割器
"""
from langchain.text_splitter import CharacterTextSplitter

# 初始字符串
state_of_the_union = "夏天是一年中最令人期待的季节之一。\\n\\n它带来了温暖的阳光、明亮的天空和充满活力的气氛。"
text_splitter = CharacterTextSplitter(
    separator="\\n\\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.create_documents([state_of_the_union])
print(texts)
"""
案例三、VectorStores 
VectorStores是一种特殊类型的数据库，
它的作用是存储由嵌入创建的向量，提供相似查询等功能。
我们使用其中一个Chroma 组件作为例子：
"""
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

loader = TextLoader('../file/test.txt', encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
