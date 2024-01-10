import os
os.environ['OPENAI_API_KEY'] = "sk-l95tvuUEBQK0aFRT92IQP6lNJ1wUxx65OXU08aKLwcmaGr4R"
os.environ['OPENAI_API_BASE']="https://openkey.cloud/v1"

from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith.schemas import Example, Run
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import json 
from bson import ObjectId


#引入mongo数据库
import pymongo
myclient = pymongo.MongoClient('localhost', 27017)
db = myclient.SONGLIST
collection = db.songs

#做基于长文本的问答
# load document
from langchain.document_loaders import PyPDFLoader
#file=open("c:/Users/Ashley/Desktop/Tags/Songs1.txt","r",encoding="utf-8")
#documents = file.read()

def Analysis (word,document):


   #结构化Prompt
   prompt_template="""

   # Role: 音乐小助手

   ## Profile：
   - Language: 中
   - Description: 音乐小助手根据情绪选择歌曲

   ### 规则: 
    1. 一首歌曲的情感可以分为歌曲本身情感（SongSentiment）和听众情感（ListenerSentiment）
    2. 一个人想听什么样的歌曲，可以看当时也处于这个情绪的人在听什么歌，这一点你可以从听众情感中获取
    3. 如果用户明确表达了想要听什么样的歌，可以从歌曲本身情感分析，找到用户需要的歌
    4. 情感的分析结果要尽可能的准确
    5. 多数时候要结合这俩个情感来找到最为匹配当下的情感的歌曲
    6. 请选择在songdetail中出现过的歌曲，如果没有找到合适的，可以回答没有找到
    7. id必须来源于songsdetail，不能是自己编写的，如果找不到就换一首歌
    8. 最重要的事情，推荐的歌曲必须是songsdetail中的歌曲
    
    ## 工作流程
    1. 你将得到一句话text
    2. 请对我此时的心情进行猜想
    3. 根据对我此时心情的猜想，在songdetail中emtion下的歌曲本身情感和听众情感，找到最为匹配我此时心情的一首歌
    5. 匹配的歌曲必须是songdetail之中，请在songsdetail中找到这首歌，并把这首歌存于songsdetail的id拿出来
    6. 有且仅需要返回id值，以json格式返回id，数据存放在id这一键值对中
    7. 所有的分析结果不必给出

    songsdetail=```{songsdetail}```
    text=```{text}```

   """

   chat = ChatOpenAI(temperature=0,model="gpt-3.5-turbo-1106")
   prompt = ChatPromptTemplate.from_template(prompt_template)

   messages = prompt.format_messages(songsdetail=document,text=word)
   respone = chat(messages).content
   return respone

"""
response_data = Analysis("好难过，她和我分手了，她背叛了我，我们明明说好要一辈子在一起的")
response = json.loads(response_data)
id = ObjectId(response.get("id"))
ans = response.get("ans")
record = collection.find_one({"_id":id})
song_id = record.get("song_id")
print(song_id)
"""

def LastAnalysis(text):
   
   music = []
   details=""
   #D:\武大\学科\大三上\JavaEE\Final\Tags\Songs1.txt
   file=open("D:\武大\学科\大三上\JavaEE\Final\Tags\Songs1.txt","r",encoding="utf-8")
   details = file.read()
   """
   for i in range(18):
      file=open("c:/Users/Ashley/Desktop/Tags/Songs"+str(i+1)+".txt","r",encoding="utf-8")
      doc = file.read()
      response_data = Analysis(word=text,document=doc)
      modified_string = response_data.replace('\n', '')
      data=modified_string.replace(' ','')
      data_str=data.replace('json','')
      data_str_1=data_str.replace('`','')
      response = json.loads(data_str_1)
      id=response.get("id")
      record = collection.find_one({"song_id":id})
      id_str=str(id)
      detail="{id="+id_str+",song_name="+record.get("song_name")+",lyrics="+str(record.get("lyrics"))
      detail+=",emotion="+str(record.get("emotion"))+"}\n"
      details+=detail
   """
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
    9. 回答要包括你先前对我句子的关怀，你推荐的歌曲简单介绍，推荐这首歌的理由和结果\
      可以适当的引用歌曲的歌词和评论来验证你的推荐是最匹配的
    10. 回答过程要尽可能的完善,人性化和丰富，你将尽可能的人性化，关心我，不要被人发现你是个机器
    11. song_id必须来源于songsdetail，不能是自己编写的，如果找不到就换一首歌
    12. 最重要的事情，推荐的歌曲必须是songsdetail中的歌曲
    13. ans中一定要有推荐的理由

    ## 工作流程
    1. 你将得到一句话text
    2. 请先对我输入的text进行回答，并且给予适当的关怀，并再开始推荐歌曲
    3. 请对我此时的心情进行猜想
    4. 根据对我此时心情的猜想，在songdetail中emtion下的歌曲本身情感和听众情感，找到更为匹配我此时心情的一首歌
    5. 匹配的歌曲必须是songdetail之中，请在songsdetail中找到这首歌，并把这首歌存于songsdetail的song_id拿出来
    5. 输出必须为json格式的文本,ans存放你的回答，不要在ans中写出id，在id中存放你推荐歌曲的song_id,两者中间用逗号分割
    6. 请只选择最为匹配的一首歌
    7. 请尽可能的丰富自己的回答,详细叙述自己为什么要选择这首歌，结合emotion来叙述
    8. 不要把歌曲id放在ans中
    songsdetail=```{songsdetail}```
    text=```{text}```
   """

   chat = ChatOpenAI(temperature=0,model="gpt-3.5-turbo-1106")
   prompt = ChatPromptTemplate.from_template(prompt_template)
   messages = prompt.format_messages(songsdetail=details,text=text)
   respone = chat(messages).content
   return respone
   
#print(LastAnalysis("我好难过，因为今天和她分手了，她背叛了我，我们说好要一直在一起的"))


from flask import Flask, request, jsonify
app = Flask(__name__)

data1=""

#api封装
@app.route('/chat', methods=['GET'])
def api_endpoint():
    # 调用你的函数
    get_data = request.args.to_dict()
    text = get_data.get('input')
    result = LastAnalysis(text)

    # 返回结果
    modified_string = result.replace('\n', '')
    data=modified_string.replace('`','')
    data1=data.replace('json','')
    print(data1)

    return json.dumps(data1)

if __name__ == '__main__':
    app.run(debug=True)


