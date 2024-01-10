from flask import Flask, jsonify, make_response, abort, request, send_file
import requests
import json
from langchain.vectorstores import LanceDB
import lancedb
import langchain
from YaGPT import YaGPTEmbeddings, YandexLLM
import uuid 
from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType

cert = "/home/vmuser/cert/fullchain.pem"
cert_key = "/home/vmuser/cert/privkey.pem"
temp_dir = "/home/vmuser/temp"
db_dir = "/home/vmuser/store"
send_audio = True

app = Flask(__name__)

def synth(txt):
    model = model_repository.synthesis_model()
    model.voice = 'jane'
    model.role = 'good'
    result = model.synthesize(txt,raw_format=False)
    fn = f"/home/vmuser/temp/{uuid.uuid4().urn}.mp3"
    result.export(fn, 'mp3')
    return fn

def reco(bin):
    model = model_repository.recognition_model()
    model.model = 'general'
    model.language = 'ru-RU'
    model.audio_processing_type = AudioProcessingType.Full
    result = model.transcribe_file(bin)
    return ' '.join(x.normalized_text for x in result)

def tg_send(chat_id, text):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    data = {"chat_id": chat_id, "text": text}
    requests.post(url, data=data)

def tg_send_audio(chat_id, text, file):
    url = f"https://api.telegram.org/bot{telegram_token}/sendAudio"
    data = { "chat_id": chat_id, "caption": text }
    files = { "audio" : open(file,'rb') }
    requests.post(url, data=data, files=files)

def do_search(chat_id,txt):
    print(f"Doing search on {txt}")
    res = retriever.get_relevant_documents(txt)
    res = chain.run(input_documents=res,query=txt)
    if send_audio:
        fn = synth(res)
        tg_send_audio(chat_id,res,fn)
    else:
        tg_send(chat_id,res)

def process(post):
    print(post)
    msg = post['message']
    chat_id = msg['chat']['id']
    txt = None
    if 'text' in msg:
        do_search(chat_id,msg['text'])
    if 'voice' in msg:
        url = f"https://api.telegram.org/bot{telegram_token}/getFile"
        data = { "file_id": msg['voice']['file_id'] }
        resp = requests.post(url, data=data).json()
        url = f"https://api.telegram.org/file/bot{telegram_token}/{resp['result']['file_path']}"
        fn = f"/home/vmuser/temp/{uuid.uuid4().urn}.mp3"
        bin = requests.get(url).content
        with open(fn,'wb') as f:
            f.write(bin)
        res = reco(fn)
        tg_send(chat_id,f'Вы спросили: {res}')
        do_search(chat_id,res)


@app.route('/',methods=['GET'])
def home():
    return "<h1>Hello</h1>"

@app.route('/tghook',methods=['GET','POST'])
def telegram_hook():
    if request.method=='POST':
        post = request.json
        process(post)
    return { "ok" : True }

print(" + Reading config")
with open('config.json') as f:
    config = json.load(f)
self_url = config['self_url']
api_key = config['api_key']
telegram_token = config['telegram_token']
folder_id = config['folder_id']

print(" + Initializing LanceDB Vector Store")
embedding = YaGPTEmbeddings(folder_id,api_key)
lance_db = lancedb.connect(db_dir)
table = lance_db.open_table("vector_index")
vec_store = LanceDB(table, embedding)
retriever = vec_store.as_retriever(
    search_kwargs={"k": 5}
)

print(" + Initializing LLM Chains")
instructions = """
Представь себе, что ты сотрудник Yandex Cloud. Твоя задача - вежливо и по мере своих сил отвечать на все вопросы собеседника.
"""
llm = YandexLLM(api_key=api_key, folder_id=folder_id,
                instruction_text = instructions)
document_prompt = langchain.prompts.PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)
# Промпт для языковой модели
document_variable_name = "context"
stuff_prompt_override = """
Пожалуйста, посмотри на текст ниже и ответь на вопрос, используя информацию из этого текста.
Текст:
-----
{context}
-----
Вопрос:
{query}"""
prompt = langchain.prompts.PromptTemplate(
    template=stuff_prompt_override, input_variables=["context", "query"]
)
# Создаём цепочку
llm_chain = langchain.chains.LLMChain(llm=llm, prompt=prompt)
chain = langchain.chains.StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
)

print(" + Configuring speech")
configure_credentials(yandex_credentials=creds.YandexCredentials(api_key=api_key))

#print(" + Registering telegram hook")
#res = requests.post(f"https://api.telegram.org/bot{telegram_token}/setWebhook",json={ "url" : f"{self_url}/tghook" })
#print(res.json())

app.run(host="0.0.0.0",port=8443,ssl_context=(cert,cert_key))