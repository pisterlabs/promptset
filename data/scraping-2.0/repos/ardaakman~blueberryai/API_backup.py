import asyncio
from fastapi import FastAPI, APIRouter, Request, Form, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.websockets import WebSocket

from pathlib import Path
from utils import *
import json
import requests
import httpx
import openai
import os
from configs import *
from twilio.rest import Client

from chat import ContextManager

app = FastAPI()

api_router = APIRouter()
BASE_PATH = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory="static"), name="static")
TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))
USER_ID = 1

openai.api_key = openai_api_key

class Chat:
    def __init__(self, context):
        self.agent_description = self.generate_agent_description(context)
        self.history = [{'role': 'system', 'content': self.agent_description}]

    def generate_agent_description(self, context):
        prompt = f"""
            You're imitating a human that is trying to {context}. 
            You're on a call with customer service.  
            Sound like a human and use your context to return the appropriate response. Keep responses short, simple, and informal.
            You could use filler words like 'um' and 'uh' to sound more human. To end the call, just return 'bye'. For information you are unsure about, return "/user <question>".
            Here is some information about you:
        """
        with open('info.txt', 'r') as f:
            info = f.read()
        prompt += info
        print(prompt)
        return prompt

    def add(self, message, role='user'):
        self.history.append({'role': role, 'content': message})

    def generate(self):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.history,
            )
            value = completion.choices[0].message['content']
            print(value)
            return value
        except:
            return "Sorry, I don't understand. Can you repeat that?"

    def stream(self, socket):
        return True
    
class Call:
    def __init__(self, to, context, call_id):
        self.recipient = to
        self.context = context
        if os.stat("info.txt").st_size == 0:
            self.questions = self.generate_questions()
        self.chat = Chat(context)
        self.call_id = call_id

    def generate_questions(self):
        try:
            prompt = f"""Given the context of {self.context}, what are some possible personal questions, 
                        such as date of birth, account number, etc. that the customer service agent might ask the user?
                        Phrase questions as key words, such as "Date of Birth". Give multiple questions seperated by a new line."""
            prompt = [{'role': 'user', 'content': prompt}]
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=prompt,
            )
            value = completion.choices[0].message['content']
            questions = value.split('\n')
            
            for question in questions:
                # ask question in input terminal and save question: answer as a new line to info.txt
                answer = input(question + '\n')
                with open('info.txt', 'a') as f:
                    f.write(question + ': ' + answer + '\n')
        except:
            print('error')
            return False

    def call(self):
        client = Client(account_sid, auth_token)
        to = self.recipient
        to = "9495016532"
        call = client.calls.create(
            # url='https://handler.twilio.com/twiml/EH9c51bf5611bc2091f8d417b4ff44d104',
            url=f'''https://fe8f-2607-f140-400-a011-20c1-f322-4b13-4bc9.ngrok-free.app/convo/{self.call_id}''',
            # url=f'''https://fe8f-2607-f140-400-a011-20c1-f322-4b13-4bc9.ngrok-free.app/convo''',
            to="+1" + to,
            from_="+18777192546"
        )
        print(call.sid)


async def make_http_request(url: str, data: dict):
    async with httpx.AsyncClient() as client:
        response = await client.post(url, data=data)
    return response


@app.get("/favicon.ico")
async def get_favicon():
    return FileResponse("static/favicon.ico")


@app.get("/", response_class=HTMLResponse)
async def init(request: Request):
    '''
    Load page to set call settings
    '''
    # get phonebook from db
    with get_db_connection() as conn:
        # select distict phone numbers and recipient from call_log where user_id is user id
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT phone_number, recipient FROM call_log WHERE user_id = ?", (USER_ID,))
        phonebook = cur.fetchall()

    return TEMPLATES.TemplateResponse(
        "init.html",
        {
            "request": request,
            "page": "init",
            "phonebook": phonebook,
        }
    )


@app.post("/init_call", response_class=HTMLResponse)
async def init(request: Request, number: str = Form(), recipient: str = Form(), context: str = Form()):
    '''
    Load page to set call settings
    '''

    # save call to db [TODO]
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO call_log (user_id, phone_number, recipient, context) VALUES (?, ?, ?, ?)", (USER_ID, number, recipient, context))
        call_id = cur.lastrowid
        conn.commit()


    # redirect to call page
    return RedirectResponse(f"/questions/{call_id}", status_code=303)



@app.get("/questions/{call_id}", response_class=HTMLResponse)
async def questions(request: Request, call_id: str):
    '''
    Page to view call history
    '''

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT context FROM call_log WHERE id = ?", (call_id,))
        context = cur.fetchone()[0]

    questions = ContextManager.generate_questions_from_task(context)
    # generate questions
    return TEMPLATES.TemplateResponse(
        "questions.html",
        {
            "request": request,
            "call_id": call_id,
            "page": "questions",
            "questions": questions
        }
    )


@app.post("/questions", response_class=HTMLResponse)
async def questions(request: Request, call_id: str = Form()):
    '''
    Page to view call history
    '''
    body = await request.form()
    print("printing body")
    question_answer_pairs = []
    for key in body.keys():
        if key != "call_id":
            question_answer_pairs.append(f'''{key}: {body[key]}''')

    return RedirectResponse(f"/call/{call_id}", status_code=303)


@app.get("/call/{call_id}", response_class=HTMLResponse)
async def call(request: Request, call_id: str, background_tasks: BackgroundTasks):
    '''
    Page to view ongoinng call
    ''' 
    # initiate call [TODO]
    # url = 'http://127.0.0.1:5001/start_call'

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT phone_number, context, id FROM call_log WHERE id = ?", (call_id,))
        call = cur.fetchone()

    call = Call(call[0], call[1], call[2])
    call.call()
    # data = {
    #     'call_id': call_id,
    #     'to': call[0],
    #     'context': call[1]
    # }

    # background_tasks.add_task(make_http_request, url, data)
    # print("added task")
    
    # # Add the request function to the background tasks
    # async with httpx.AsyncClient() as client:
    #     response = await client.get(url, params=data)

    return TEMPLATES.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "page": "call",
            'call_id': call_id
        }
    )

@app.post("/save_message")
def save_message(request: Request):
    '''
    Send data to client
    '''
    # save to db

    # send to client
    response_val = "Hi how are you doing today?"

    # generate respoinse
    data = {'message': message, 'call_id': call_id, 'sender': sender}
    send_data_to_clients(json.dumps(data))
    return response_val


# SOCKETS
sockets = []
async def send_data_to_clients(data):
    # Iterate over each connected websocket
    for websocket in sockets:
        try:
            data = json.dumps(data)
            await websocket.send_text(data)
        except Exception:
            # Handle any errors that occur while sending data
            pass


@app.websocket("/websocket")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Append the connected websocket to the 'sockets' list
    sockets.append(websocket)

    try:
        while True:
            data = await websocket.receive_text()

            # Send the received data to all connected clients
            await send_data_to_clients(data)
    except WebSocketDisconnect:
        print("client disconnected")
        sockets.remove(websocket)


@app.get("/account", response_class=HTMLResponse)
async def history(request: Request):
    '''
    Page to view call history
    '''
    return TEMPLATES.TemplateResponse(
        "account.html",
        {
            "request": request,
        }
    )


# end call
@app.post("/end_call")
async def end_call(request: Request):
    '''
    End call
    '''
    body = await request.json()
    call_id = body['call_id']

    print('ending call' + call_id)
    
    # add message
    return {'status': 'success'}


# add message
# @app.post("/send_message")
# async def add_message(request: Request):
#     '''
#     End call
#     '''
#     body = await request.json()
#     call_id = body['call_id']
#     message = body['message']
    
#     print('sending message' + message)
#     # add message

#     return {'status': 'success'}

