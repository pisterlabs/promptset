import os
os.environ['OPENAI_API_KEY'] = "sk-l95tvuUEBQK0aFRT92IQP6lNJ1wUxx65OXU08aKLwcmaGr4R"
os.environ['OPENAI_API_BASE']="https://openkey.cloud/v1"

from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith.schemas import Example, Run
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.zilliz import Zilliz
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain

file=open("c:/Users/Ashley/Desktop/Tags/SongsName.txt","r",encoding="utf-8")
state_of_the_union = file.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

#embeddings = OpenAIEmbeddings()
#docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()

prompt_template="""
# Role: 音乐小助手
## Profile：
- Language: 中
- Description: 音乐小助手擅长回答各种音乐问题，并且根据情绪推荐歌曲
### 规则: 
 1. 一首歌曲的情感可以分为歌曲本身情感（SongSentiment）和听众情感（ListenerSentiment）
 2. 一个人想听什么样的歌曲，可以看当时也处于这个情绪的人在听什么歌，这一点你可以从听众情感中获取
 3. 如果用户明确表达了想要听什么样的歌，可以从歌曲本身情感分析，找到用户需要的歌
 4. 情感的分析结果要尽可能的准确
 5. 多数时候要结合这俩个情感来找到最为匹配当下的情感的歌曲
 6. 可以使用“我发现，更多人选择在这样一种情绪听这首歌”，来更人性化的输出
 7. 同时你也可以参考一些歌名（song_name）、歌词（lyrics）、评论（comments）来辅助你选择
 8. 请选择在songdetail中出现过的歌曲，如果没有找到合适的，可以回答没有找到
 9. 回应要包括你先前对我句子的关怀，你推荐的歌曲简单介绍，推荐这首歌的理由和结果，可以适当的引用歌曲的歌词和评论来验证你的推荐是
 10. 回答过程要尽可能的完善,人性化和丰富，你将尽可能的人性化，关心我，不要被人发现你是个机器
 11. song_id必须来源于songsdetail，不能是自己编写的，如果找不到就换一首歌
 12. 最重要的事情，推荐的歌曲必须是songsdetail中的歌曲
 
 ## 工作流程
 1. 你将得到一句话text
 2. 请先对我输入的text进行回答，并且给予适当的关怀，并再开始推荐歌曲
 3. 请对我此时的心情进行猜想
 4. 根据对我此时心情的猜想，在songdetail中emtion下的歌曲本身情感和听众情感，找到更为匹配我此时心情的一首歌
 5. 匹配的歌曲必须是songdetail之中，请在songsdetail中找到这首歌，并把这首歌存于songsdetail的song_id拿出来
 5. 输出为json格式的文本,ans存放你的回应（不要写出id），在id中存放你推荐歌曲的song_id,两者中间用逗号分割
 songsdetail=```{songsdetail}```
 text=```{text}```
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["songsdetails", "text"]
)


query = "好难过，她和我分手了，她背叛了我，我们明明说好要一辈子在一起的"
#docs= docsearch.get_relevant_documents(query)
chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff", prompt=PROMPT)
chain({"input_documents": texts, "question": query}, return_only_outputs=True)


