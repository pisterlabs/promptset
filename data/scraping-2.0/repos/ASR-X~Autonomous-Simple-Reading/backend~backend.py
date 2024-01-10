# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install stream-python stream-chat openai bert-extractive-summarizer fastapi nest-asyncio pyngrok uvicorn google-cloud-vision sentencepiece

#@title detect document function
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/gcOCRkey.json"
def detect_document(content):
    """Detects document features in an image."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    """with io.open(path, 'rb') as image_file:
    content = image_file.read()"""

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        bs=[]
        for block in page.blocks:
            print('\nBlock confidence: {}\n'.format(block.confidence))
            q=[]
            for paragraph in block.paragraphs:
                #print('Paragraph confidence: {}'.format(
                #    paragraph.confidence))
                
                for word in paragraph.words:
                    q.append( ''.join([
                        symbol.text for symbol in word.symbols
                    ]))
                    tr="""print('Word text: {} (confidence: {})'.format(
                        word_text, word.confidence))

                    for symbol in word.symbols:
                        print('\tSymbol: {} (confidence: {})'.format(
                            symbol.text, symbol.confidence))"""
            bs.append(' '.join(q))
        return '\n'.join(bs)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

#@title string constants

gptprompt="""
The following is a transcript of a friendly AI simplifying text down for a child. The format is a complex text followed by an example with a simplified text with an analogy that is easy for 5 year olds to comprehend. The 5 year old then asks a few questions, and the AI responds to the 5 year old.

Complex Text: Database, also called electronic database, any collection of data, or information, that is specially organized for rapid search and retrieval by a computer. Databases are structured to facilitate the storage, retrieval, modification, and deletion of data in conjunction with various data-processing operations. A database management system (DBMS) extracts information from the database in response to queries.
Simplified Text: The most common databases would store your toys in pieces. It might make sense, for example, to create a second box inside your toy box (database) to store all the heads of your dolls and another box to store arms. That type of database is called a relational database because it keeps track of which arms and heads go with (relate to) which dolls. But there’s a newer type of database which lets you keep your dolls in one piece. That kind of database is called a non-relational database.
5 year old: I don't think I understand why someone needs a relational database.
AI Response: A relational database is good if you want to keep track of how many dolls you have, or which dolls are missing their arms.
5 year old: Why would someone need to keep track of that?
AI Response: If you were selling your dolls, for example, you might want to know which dolls were missing their arms before you sold them.
5 year old: I don't think I would ever sell my dolls.
AI Response: Okay, but if you did, you might want to know which ones were missing an arm so you could tell who was buying them

Complex Text: Machine learning is the study of computer algorithms that improve automatically through experience and by the use of data. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.
Simplified Text: Machine learning is a type of artificial intelligence that lets computers learn from experience. It is like when you teach your dog to sit by giving it treats. But instead of using treats, machine learning uses data. You can't just tell a dog to do something, but if you give it treats when it does something, it will learn to do that. Machine learning algorithms are used in many applications, such as email filtering and computer vision.
5 year old: Why would someone need to teach a computer to do something a human can do?
AI Response: Computers can process information faster than humans. For example, if you had a thousand books and you wanted to find one that talked about dogs, it would be really hard for a human to find the right book. But if you gave the job to the right computer, it could search through all of the books in seconds.
5 year old: How is machine learning used in email filtering?
AI Response: Machine learning algorithms can help filter out spam emails.
5 year old: Why does a machine need to filter out spam emails?
AI Response: Spam emails are annoying and they can be dangerous. Sometimes, people put viruses in spam emails that make your computer do things it shouldn't do. That's why we have machines that can help us find the good emails from the bad ones.

Complex Text: The feature of a quantum system whereby it exists in several separate quantum states at the same time. For example, electrons possess a quantum feature called spin, a type of intrinsic angular momentum. In the presence of a magnetic field, the electron may exist in two possible spin states, usually referred to as spin up and spin down. Each electron, until it is measured, will have a finite chance of being in either state. Only when measured is it observed to be in a specific spin state. In quantum experience the situation is more unsettling: material properties of things do not exist until they are measured. Until you look (measure the particular property), as it were, it has no fixed face up.
Simplified Text: Quantum systems can exist in many states at the same time. Quantum experience is even more weird- it is like an apple that doesn’t have a color until you look at it. That’s because the apple has a lot of colors and it doesn't decide which one it likes until you look at it. The same thing is true for quantum objects- an electron can be in more than one place at the same time until you look at it. That is called quantum superposition.
5 year old: Why doesn't it choose which state it is in?
AI Response: It only decides what is it like when you look at it. Until then, it is in the quantum superposition.
5 year old: What is a quantum superposition?
AI Response: A state in which something exists in more than one place or more than one state at the same time. For example, an electron can spin both up and down until it is observed.

"""

def shorten(prompt,len):
  response = openai.Completion.create(engine="davinci-instruct-beta", prompt="Shorten the following text:\n"+prompt, max_tokens=len)
  return response['choices'][0]['text']
#print(shorten(body,500))

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def getSimpleText(text,len):
  global gptprompt
  response = openai.Completion.create(engine="davinci-instruct-beta", prompt=gptprompt+"Complex Text: "+text+"""
Simplified Text:""", temperature=0.4,
    max_tokens=200,
    top_p=1,
    frequency_penalty=0.13,
    presence_penalty=0.08,
    stop=["5 year old:"]
  )
  res=response['choices'][0]['text'][1:]
  return [gptprompt+text+response['choices'][0]['text'],res]

def answerQuestion(text,len):
  response = openai.Completion.create(engine="curie-instruct-beta", prompt=text+"\nAI Response:", temperature=0.4,
    max_tokens=200,
    top_p=1,
    frequency_penalty=0.13,
    presence_penalty=0.08,
    stop=["5 year old:", "Complex Text:"]
  )
  res=response['choices'][0]['text']
  return res

def suggestQuestions(text,len):
  responses=[]
  for i in range(3):
    response = openai.Completion.create(engine="curie-instruct-beta", prompt=text+"\n5 year old:", temperature=0.65,
      max_tokens=200,
      top_p=1,
      frequency_penalty=0.13,
      presence_penalty=0.08,
      stop=["AI Response:", "Complex Text:", "AI response"]
    )
    responses.append(response['choices'][0]['text'][1:])

  return responses

openai.api_key = "APIKEY"
#print(getSimpleText("""Water is an inorganic, transparent, tasteless, odorless, and nearly colorless chemical substance, which is the main constituent of Earth's hydrosphere and the fluids of all known living organisms (in which it acts as a solvent[1]). It is vital for all known forms of life, even though it provides no calories or organic nutrients. Its chemical formula is H2O, meaning that each of its molecules contains one oxygen and two hydrogen atoms, connected by covalent bonds. Two hydrogen atoms are attached to one oxygen atom at an angle of 104.45°.[2]
#
#"Water" is the name of the liquid state of H2O at standard conditions for temperature and pressure. It forms precipitation in the form of rain and aerosols in the form of fog. Clouds are formed from suspended droplets of water and ice, its solid state. When finely divided, crystalline ice may precipitate in the form of snow. The gaseous state of water is steam or water vapor.""",200))

users={}

from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from stream_chat import StreamChatAsync
from stream_chat import StreamChat
import asyncio
from time import sleep
import re

class UserIn(BaseModel):
    user : str

class UserOut(BaseModel):
    user: str

class ImageIn(BaseModel):
    base64: str

class ImageOut(BaseModel):
    html: str

class TextIn(BaseModel):
    text: str

class TextOut(BaseModel):
    text: str


class SugOut(BaseModel):
    o1: str
    o2: str
    o3: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins='*',
    allow_credentials=True,
    allow_methods='*',
    allow_headers='*',
)

synclient = StreamChat(api_key="6rtepqre25n5", api_secret="vm2cz34wd9ya9p33trcawn4tt59cm55pyavq3eke8ckeu62pdmswaysrnebrt4dz")
synclient.update_users([{"id": "gpt-3-user-id", "role": "admin", "name": "Bot", "image": "https://getstream.io/random_svg/?name=Bot"}])
client = StreamChatAsync(api_key="6rtepqre25n5", api_secret="vm2cz34wd9ya9p33trcawn4tt59cm55pyavq3eke8ckeu62pdmswaysrnebrt4dz")

async def simplify(req):
  loop = asyncio.get_running_loop()
  print(req)
  channelid = req['channel_id']
  process = True;
  if req['type'] != "message.new" or req['message']['user']['id'] == 'gpt-3-user-id':
    process = False;
  
  if process:
    channel = client.channel("messaging", channelid )
    await channel.send_event({'type': 'message.read'}, "gpt-3-user-id")
    #print('Text - ' + req['message']['text'])
    #print(users)

    await channel.send_event({'type': 'typing.start'}, "gpt-3-user-id")
    if not users[channelid]:
      
      ans = await loop.run_in_executor(None, getSimpleText, req['message']['text'], 200)
      users[channelid]=ans[0]
      #print("Simplify - " + ans[1])
      if re.sub('\s','',ans[1]) != "":
        await channel.send_message({ "text": ans[1] }, "gpt-3-user-id")
        await channel.send_message({ "text": "If you have any more questions, ask me!" }, "gpt-3-user-id")
    else:
      ans = await loop.run_in_executor(None, answerQuestion, users[channelid]+'\n5 year old:'+req['message']['text'], 100)
      #print("Ask - " + ans)

      if re.sub('\s','',ans) != "": await channel.send_message({"text": ans}, "gpt-3-user-id")

    await channel.send_event({'type': 'typing.stop'}, "gpt-3-user-id")

@app.api_route("/", methods=["GET", "POST"])
async def root():
    return 'alive'

@app.post('/create', response_model=UserOut)
async def createChannel(userin: UserIn):
    user = userin.user
    channel = client.channel("messaging", user, {
        'image': 'https://www.drupal.org/files/project-images/react.png',
        'name': 'Simple Reading',
        'members': [user],
    } )
    users[user]=""
    print(userin.user)
    await channel.create(user)
    message = { "text": "Please tell me some complex text you would like to simplify."} 
    await channel.send_message(message, "gpt-3-user-id")

    return {"user": user}

@app.post('/push')
async def push(request: Request, background_tasks: BackgroundTasks):
  reqjson = await request.json()
  background_tasks.add_task(simplify, reqjson)
  return Response(status_code=200)
  

@app.post('/simplifyText', response_model=TextOut)
async def simplifyText(textin: TextIn,userin: UserIn):
  ans=getSimpleText(textin.text,200)
  users[userin.user]=ans[0]
  return {"text":ans[1]}

@app.post('/ask', response_model=TextOut)
async def ask(textin: TextIn,userin: UserIn):
  return {"text":answerQuestion(users[userin.user]+'\n5 year old:'+textin.text,100)}

@app.post('/suggest', response_model=SugOut)
async def suggest(userin: UserIn):
  loop = asyncio.get_running_loop()
  while not users[userin.user]:
    await loop.run_in_executor(None, sleep, 0.1)
  sug= await loop.run_in_executor(None, suggestQuestions, users[userin.user],100)
  return {"o1":sug[0],"o2":sug[1],"o3":sug[2]}


@app.post("/ocr", response_model=ImageOut)
async def process_image(imagein: ImageIn):
    ret=detect_document(base64.b64decode(imagein.base64))
    return {"html":ret}

import nest_asyncio
from pyngrok import ngrok
import uvicorn

ngrok_tunnel = ngrok.connect(80, bind_tls=True)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=80)
