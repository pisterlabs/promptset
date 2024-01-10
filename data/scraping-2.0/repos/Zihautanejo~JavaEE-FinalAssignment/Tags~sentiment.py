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

#引入mongo数据库
import pymongo
myclient = pymongo.MongoClient('localhost', 27017)
db = myclient.SONGLIST
collection = db.songs

"""
#情感词典
#快乐,安心,尊敬,赞扬,相信,喜爱,祝愿,愤怒,悲伤,失望,疚,思,慌,恐惧,羞,烦闷,憎恶,贬责,妒忌,怀疑,惊奇
#钦佩、崇拜、欣赏、娱乐、焦虑、敬畏、尴尬、厌倦、冷静、困惑、渴望、厌恶、痛苦、着迷、嫉妒、兴奋、恐惧、痛恨、有趣、快乐、怀旧、浪漫、悲伤、满意、性欲、同情和满足
#"快乐","幸福","兴奋","满足","喜爱","祝愿","思念","悲伤","痛苦","沮丧","烦闷""失望","愧疚","愤怒","恼怒","恐惧","担心","紧张","慌乱","忧虑","惊讶","怀疑","赞扬","敬畏","厌恶","轻蔑","贬责","嫉妒","信任","依赖","焦虑"
#GoEmotion"钦佩","娱乐","支持","关心","渴望","兴奋","感激","快乐","爱","乐观","自豪","轻松","生气","烦恼","失望","厌恶","反对","尴尬","害怕","悲伤","紧张","悔恨","烦恼","困惑","好奇","领悟","惊讶"
"""
Emotion = ["快乐","幸福","兴奋","满足","喜爱",
           "祝愿","思念","悲伤","痛苦","沮丧",
           "烦闷","失望","愧疚","愤怒","恼怒",
           "恐惧","担心","紧张","慌乱","忧虑",
           "惊讶","怀疑","赞扬","敬畏","厌恶",
           "轻蔑","贬责","嫉妒","信任","依赖",
           "焦虑","困惑","尴尬","冷静","渴望"]
GoEmotion =["钦佩","娱乐","支持","关心","渴望",
            "兴奋","感激","快乐","爱恋","乐观",
            "自豪","轻松","生气","烦恼","失望",
            "厌恶","反对","尴尬","害怕","悲伤",
            "紧张","悔恨","悲痛","困惑","好奇",
            "领悟","惊讶"]      

def EmotionAnalysis (song_name,lyrics,comments):
   
   prompt_template="""
   # Role: 心理咨询师
   ## Profile：
   - Language: 中
   - Description: 心理咨询师很擅长从文字中提取情感，并且对情绪的强弱感知很敏锐
   ### 规则: 分析歌曲情感
      1. 擅长从文本感知情感，对多种情感的种类和强度敏感
      2. 对于一首歌，认为有俩种情感，歌曲本身情感（SongSentiment）和听众情感（ListenerSentiment）
      3. 歌曲本身情感中，歌名（song_name）最能体现情感，其次是歌词（lyrics）
      4. 听众情感可以从评论（comments）中获取
      3. 反复出现的歌词比只出现一次的歌词表达的感情更强烈
      4. 评论列表（comments）中，前面的评论比后面的评论情感价值更高
      5. 一首歌中多次或显著的情感更强烈
      6. 情感强烈和情感不强烈的分数差值要显著
      7. 情感的分析结果要尽可能的准确
      8. 总结要尽可能的反映观众的情感
    ## 工作流程
      1. 对一首歌曲进行情感分析，包括歌曲本身情感和听众情感俩个部分
      2. 在输入中，你可以获得到歌名（song_name）、歌词（lyrics）、评论（comments）信息
      3. 最后请只输出你对这首歌曲的详细情感分析结果的总结,字数控制在150字以内，以json的格式输出
    song_name=```{song_name}```
    lyrics=```{lyrics}```
    comments=```{comments}```
   """

   chat = ChatOpenAI(temperature=0)
   prompt = ChatPromptTemplate.from_template(prompt_template)

   """
   messages = [{"role": "user", "content": prompt}]
   response = ChatOpenAI.ChatCompletion.create(
        model="gdt-3.5-model",
        messages=messages,
        temperature=0, 
    )
   return response.choices[0].message["content"]
   """

   messages = prompt.format_messages(
                                    emotion=GoEmotion,
                                    song_name=song_name,
                                    lyrics=lyrics,
                                    comments=comments)
   respone = chat(messages).content
   return respone

"""
for coll in collection.find():
    song_name = coll.get("song_name")
    lyrics = coll.get("lyrics")
    comments = coll.get("comments")
    word = EmotionAnalysis(song_name,lyrics,comments)
    update_query = {"$set": {"emotion": word}}
    query={"_id": coll.get("_id")}
    collection.update_one(query, update_query)


print("game over")
"""
query = ({"song_name":"Don't Tell Me"})
coll = collection.find_one(query)
song_name = coll.get("song_name")
id = coll.get("song_id")
lyrics = coll.get("lyrics")
comments = coll.get("comments")
#word = EmotionAnalysis(song_name,lyrics,comments)
print(id)
