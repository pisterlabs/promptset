# server for customized chatbot
from hashlib import md5
import json
import uuid
import sqlite3
import time

from openai import OpenAI
import tornado.ioloop
import tornado.web

OPENAI_CLIENT = OpenAI()
MESSAGE_STORAGE = {}

MODELS = None
MODELS_FETCH_TIME = 0

PROMPTS = None
PROMPTS_FETCH_TIME = 0

# sqlite3 connection
# dev: test.db
# personal prompt engineering: prompt.db
# yipit: yipit.db
DB_CONN = None
def create_table():
    cursor = DB_CONN.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid VARCHAR(255) UNIQUE,
            model VARCHAR(255),
            system_message TEXT,
            system_message_hash VARCHAR(32),
            full_context TEXT,
            timestamp INTEGER
        );
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_uuid ON chat_history (uuid);
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_system_message_hash ON chat_history (system_message_hash);
        '''
    )
    DB_CONN.commit()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS saved_prompt (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(255),
            version VARCHAR(255),
            system_message TEXT,
            system_message_hash VARCHAR(32) UNIQUE,
            timestamp INTEGER,

            UNIQUE(name, version)
        );
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_name ON saved_prompt (name);
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_system_message_hash ON saved_prompt (system_message_hash);
        '''
    )
    DB_CONN.commit()

def insert_chat_history(uuid: str, model: str, full_context: list):
    cursor = DB_CONN.cursor()
    system_message = full_context[0]['content']
    system_message_hash = md5(system_message.encode('utf-8')).hexdigest()

    # insert chat history
    cursor.execute(
        '''
        INSERT INTO chat_history (uuid, model, system_message, system_message_hash, full_context, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        ''',
        (uuid, model, system_message, system_message_hash, json.dumps(full_context), int(time.time()))
    )
    DB_CONN.commit()

def save_prompt(name: str, version: str, system_message: str):
    cursor = DB_CONN.cursor()
    system_message_hash = md5(system_message.encode('utf-8')).hexdigest()

    cursor.execute(
        '''
        INSERT INTO saved_prompt (name, version, system_message, system_message_hash, timestamp)
        VALUES (?, ?, ?, ?, ?)
        ''',
        (name, version, system_message, system_message_hash, int(time.time()))
    )
    DB_CONN.commit()

def delete_prompt(name: str, version: str):
    pass

def get_all_prompts():
    global PROMPTS, PROMPTS_FETCH_TIME
    if PROMPTS is None or time.time() - PROMPTS_FETCH_TIME > 60:
        cursor = DB_CONN.cursor()
        cursor.execute(
            '''
            SELECT name, version, system_message FROM saved_prompt
            '''
        )
        rows = cursor.fetchall()
        PROMPTS = {}
        for row in rows:
            name, version, system_message = row
            PROMPTS[f"{name}:{version}"] = {
                "name": name,
                "version": version,
                "systemMessage": system_message
            }
        PROMPTS_FETCH_TIME = time.time()
    return PROMPTS


class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        # Allow all origins
        self.set_header("Access-Control-Allow-Origin", "*")
        
        # Allow specific headers
        self.set_header("Access-Control-Allow-Headers", "*")
        
        # Allow specific methods
        self.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    
    def options(self):
        # Handle preflight OPTIONS requests
        self.set_status(204)
        self.finish()

class ChatHandler(BaseHandler):
    def post(self):
        # get message from request body
        body = json.loads(self.request.body)
        system_message = body.get("systemMessage", "You are a helpful assistant")
        user_message = body.get("userMessage", "Repeat after me: I'm a helpful assistant")
        request_uuid = str(uuid.uuid4())
        MESSAGE_STORAGE[request_uuid] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        print(f"Receive post request, uuid: {request_uuid}")
        self.set_status(200)
        self.write({"uuid": request_uuid})
        self.finish()
    
    def build_sse_message(self, data):
        body = json.dumps({"content": data})
        return f"data: {body}\n\n"

    def get(self):
        global MESSAGE_STORAGE
        # set headers for SSE to work
        self.set_header('Content-Type', 'text/event-stream')
        self.set_header('Cache-Control', 'no-cache')
        self.set_header('Connection', 'keep-alive')

        # get uuid from url
        request_uuid = self.get_argument('uuid')
        model = self.get_argument('model')
        full_context = MESSAGE_STORAGE[request_uuid]
        print(f"Receive get request, uuid: {request_uuid}, model: {model}")

        # create openai stream
        stream = OPENAI_CLIENT.chat.completions.create(
            model=model,
            messages=full_context,
            stream=True
        )

        assistant_message = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                assistant_message += content
                self.write(self.build_sse_message(content))
                self.flush()

        # save chat history
        full_context.append(
            {"role": "assistant", "content": assistant_message}
        )
        insert_chat_history(request_uuid, model, full_context)
        del MESSAGE_STORAGE[request_uuid]
        self.finish()

class ListModelsHandler(BaseHandler):
    def get(self):
        global MODELS, MODELS_FETCH_TIME
        if MODELS is None or time.time() - MODELS_FETCH_TIME > 3600:
            MODELS = []
            for model in OPENAI_CLIENT.models.list():
                model_id = str(model.id)
                if model_id.startswith(("gpt-3.5", "gpt-4")):
                    MODELS.append(model_id)
            MODELS = sorted(MODELS)
            MODELS_FETCH_TIME = time.time()

        self.set_status(200)
        self.write({"models": MODELS})
        self.finish()

class PromptHandler(BaseHandler):
    def get(self):
        prompts = get_all_prompts()
        self.write(json.dumps({"prompts": prompts}))
        self.set_status(200)
        self.finish()
    
    def post(self):
        body = json.loads(self.request.body)
        name = body.get("name", None)
        version = body.get("version", None)
        system_message = body.get("systemMessage", None)
        if name is None or system_message is None:
            self.set_status(400)
            self.finish()
        else:
            print(f"Save prompt, name: {name}, version: {version}")
            save_prompt(name, version, system_message)
            self.set_status(200)
            self.finish()

def make_app():
    return tornado.web.Application([
        (r'/chat', ChatHandler),
        (r'/list_models', ListModelsHandler),
        (r'/prompt', PromptHandler)
    ])

if __name__ == '__main__':
    import sys
    db_name = str(sys.argv[1]) if len(sys.argv) > 1 else 'test.db'

    DB_CONN = sqlite3.connect(db_name)
    print("Connected to database: ", db_name)
    
    create_table()
    app = make_app()
    app.listen(5600)
    print('Starting Tornado server on http://localhost:5600')
    tornado.ioloop.IOLoop.current().start()
