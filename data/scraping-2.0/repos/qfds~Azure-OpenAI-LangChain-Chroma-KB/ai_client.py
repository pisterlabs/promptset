from chromadb.utils import embedding_functions
from chromadb.config import Settings
from datetime import datetime
from tkinter import filedialog
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import *
from PIL import ImageTk, Image
import tkinter as tk
import azure.cognitiveservices.speech as speechsdk
import chromadb
import openai
import configparser
import zipfile
import logging
import shutil
import os

db_dir = 'vs'
kb_dir = 'history'
config_file = 'config.ini'
def_color = 'black'
query_color = '#000C7B'
info_color = '#C00000'
insert_info = "请输入内容后再操作！"
success = "词条学习成功,可以开始问答啦！"
fail = "词条没有插入，请再试一次！"
goodbye = "问答结束，齐风再见！"

font_style = ("微软雅黑", 9)
tile_style = ("微软雅黑", 9,"bold")

config = configparser.ConfigParser()
config.read(config_file)

openai.api_type = "azure"
openai.api_version = config.get('Azure OpenAI','OPENAI_API_VER')
openai.api_base = config.get('Azure OpenAI','OPENAI_API_BASE')
openai.api_key = config.get('Azure OpenAI','OPENAI_API_KEY')

GPT_NAME = config.get('Azure OpenAI','GPT_NAME')
DEPLOYMENT_NAME = config.get('Azure OpenAI','DEPLOYMENT_NAME')
TEXT_EMBED = config.get('Azure OpenAI','TEXT_EMBED')
TEXT_EMBED_MOD = config.get('Azure OpenAI','TEXT_EMBED_MOD')

SPEECH_KEY = config.get('Cognitive Services','SPEECH_KEY')
SPEECH_REGION = config.get('Cognitive Services','SPEECH_REGION')

#对语音服务进行配置
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)

#文字转化为语音，或者将语音转化为文字.
speech_config.speech_synthesis_voice_name = 'zh-CN-XiaoyouNeural'
speech_config.speech_recognition_language = "zh-CN"

#文字转语音服务
audio_config_txt = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config_txt)

#语音识别并转换成文字
audio_config_voice = speechsdk.audio.AudioConfig(use_default_microphone=True)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config_voice)

#对学习的文本进行切割及embedding
open_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key,
                                                      api_base=openai.api_base,
                                                      api_type=openai.api_type,
                                                      model_name=TEXT_EMBED_MOD)

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=db_dir, anonymized_telemetry=False))

collection = client.get_or_create_collection(name="collection", 
                                             embedding_function=open_ef, 
                                             metadata={"hnsw:space": "cosine"})

logging.basicConfig(filename='error.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def system_info(content, txt_color):
    info_text.insert(tk.END, f"[{datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}] " + content + '\n',"tag%s" % txt_color)
    info_text.tag_config("tag%s" % txt_color, foreground=txt_color)

def get_result(query):
    tempalte = "You should answer the question based on the given context, if no context is found, answer I don't know. The answer should be in Chinese"
    search_res = collection.query(query_texts=[query],n_results=3)
    prompt = "based on the given context, context:"+str(search_res['documents'])+"the answer of"+query
    conversation = [{"role": "system", "content": tempalte}]
    conversation.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(engine=GPT_NAME, messages=conversation, temperature=0.1)
    answer = response['choices'][0]['message']['content']
    return answer

def upload_file():
    try:
        filetypes = [('Zip files', '*.zip')]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        dbname = os.path.basename(filename) 
        if not os.path.exists(kb_dir):  # 判断文件夹是否存在
            os.makedirs(kb_dir)
            system_info("本地目录创建成功", def_color)
        if os.path.exists(db_dir):  # 判断文件夹是否存在
            shutil.rmtree(db_dir)
            system_info("本地数据库已删除",def_color)
        shutil.copy(filename,kb_dir)
        with zipfile.ZipFile(kb_dir+'/'+dbname, 'r') as zip_ref:  
            # 解压所有文件  
            zip_ref.extractall(db_dir) 
        system_info(success,def_color)
        os.remove(kb_dir+'/'+dbname) 
    except Exception as e:
        # 处理异常的代码
        system_info(fail,info_color)
        logging.info(e)
    
def ai_qa():
    try:        
        speech_synthesizer.speak_text_async("请说出您的问题").get()
        speech_recognition_result = speech_recognizer.recognize_once_async().get()
        query = speech_recognition_result.text
        system_info("语音问题：" + query, query_color)
        if query == "结束。":
            system_info(goodbye, def_color)
            speech_synthesizer.speak_text_async(goodbye).get()
            exit(0)
        if query == "":
            blank_info = "没有收到问题，请尝试再问一次"
            system_info(blank_info,def_color)
            speech_synthesizer.speak_text_async(blank_info).get()
        else:
            final = get_result(query)
            system_info("AI的回答是："+final,def_color)
            speech_synthesizer.speak_text_async("AI的回答是："+final).get()
    except Exception as e:
        # 处理异常的代码
        system_info("系统异常，再试一次吧", info_color)
        speech_synthesizer.speak_text_async("系统异常，再试一次吧").get()
        logging.info(e)

def text_qa():
    query = text_entry.get()
    if not query == "": 
        try:
            system_info("文本问题："+query,query_color)
            if query == "结束":
                system_info(goodbye, def_color)
                exit(0)
            else:
                final = get_result(query)
                text_entry.delete(0, tk.END)
                system_info("AI的回答是："+final, def_color)
        except Exception as e:
            system_info("系统异常，再试一次吧", info_color)
            logging.info(e)
    else:
        system_info(insert_info, info_color)

#用户界面布局
root = tk.Tk()

root.title("Open AI本地知识库语音问答客户端 v0.2")
root.geometry("600x405+100+100")
root.iconbitmap("app.ico")
root.resizable(False,False)

# 窗体左侧布局
frame_left = tk.Frame(root)
frame_left.pack(side=tk.LEFT)

# 窗体左侧上部布局
frame_kb = tk.LabelFrame(frame_left, text= "本地数据库导入", font=tile_style)
frame_kb.pack(side=tk.TOP, anchor=tk.N, fill=tk.BOTH, padx=10, pady=10)

img = Image.open("logo.png") 
# 创建PhotoImage对象  
photo = ImageTk.PhotoImage(img) 
  
# 创建Canvas对象，将图片放置在(100, 100)位置 
canvas = Canvas(frame_kb, width=180, height=60)  
canvas.pack() 
canvas.create_image(0, 0, image=photo, anchor=NW) 

separator = ttk.Separator(frame_kb, orient="horizontal")
separator.pack(fill="x", padx=10, pady=10)

entry_label = tk.Label(frame_kb, text="[注意] 上传后数据库文件将被覆盖", fg=info_color)
entry_label.pack(side="top", anchor=tk.NW, padx=3, pady=3)

upload_info = tk.Label(frame_kb, text="上传Chroma向量数据库：").pack(side="left",  padx=3, pady=5)
upload_button = tk.Button(frame_kb, text="上传文件", command=upload_file).pack(side="right", padx=3, pady=5)

# 窗体左侧下部布局 
frame_ai = tk.LabelFrame(frame_left, text= "知识库AI问答", font=tile_style)
frame_ai.pack(side=tk.TOP, anchor=tk.N, fill=tk.BOTH, padx=10, pady=10)

qa_label = tk.Label(frame_ai, text="点击按钮开始语音问答：")
qa_label.pack(side="top", anchor=tk.W, padx=3, pady=5)

aiqa = tk.Button(frame_ai, text="语音问答", bg=info_color, fg="white", command=ai_qa)
aiqa.pack(side="top", anchor=tk.E, padx=3, pady=3)

text_label = tk.Label(frame_ai, text="输入您的问题：")
text_label.pack(anchor=tk.NW, padx=3, pady=3)

text_entry = tk.Entry(frame_ai, width=30)
text_entry.pack(side="top", anchor=tk.NW, padx=3, pady=3)

button = tk.Button(frame_ai, text="获取答案", command=text_qa)
button.pack(side="top", anchor=tk.NE, padx=3, pady=3)

# 窗体右侧布局 
frame_right = tk.Frame(root)
frame_right.pack(side=tk.TOP, padx=5)

info_label = tk.Label(frame_right, text="系统信息：", font=tile_style).pack(anchor=tk.W)
user = tk.Label(frame_right, text="Created by Eric Qi @ 2023"+"\n"+"Powered by Open AI GPT3.5", fg=query_color, font=("微软雅黑", 8)).pack(side="bottom", pady=3, anchor=tk.E)

info_text = scrolledtext.ScrolledText(frame_right, font=font_style, bg='#F0F0F0')
info_text.pack(side=tk.LEFT, fill=tk.BOTH)

root.mainloop()