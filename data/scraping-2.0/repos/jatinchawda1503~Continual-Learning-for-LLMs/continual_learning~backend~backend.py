import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config as CFG


from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain.chat_models import ChatOpenAI

from uuid import uuid4
import psycopg

from continual_learning.backend.memory_management.postgres_memory_manager import PostGresMemoryHistory
from continual_learning.backend.agents.get_agent import MemoryConversationalAgent
from continual_learning.backend.prompt_manager.conversation_prompt import CONVERSATIONAL_PROMPT_TEMPLATE
from continual_learning.backend.prompt_manager.prompt import MemoryPromptHandler

OPENAI_API_KEY = CFG.get_openai_api_key()
DB_CONFIG = CFG.get_db_config()

print(DB_CONFIG)

app = Flask(__name__)
CORS(app)

llms = ChatOpenAI(model="gpt-4", temperature=0.5, api_key=OPENAI_API_KEY)
    

@app.route('/history', methods=['GET'])
def history():
    try:
        conn =  psycopg.connect(dbname=DB_CONFIG['dbname'], user=DB_CONFIG['user'], password=DB_CONFIG['password'], host=DB_CONFIG['host'], port=DB_CONFIG['port'])
        cur = conn.cursor()
        query = f"SELECT chat_id, MAX(time_stamp) AS latest_time FROM {DB_CONFIG['conversation_table_name']} GROUP BY chat_id ORDER BY latest_time DESC;"
        cur.execute(query)
        current_chat_ids = [record[0] for record in cur.fetchall()]
        conn.close()
        cur.close()
    except psycopg.Error as e:
        current_chat_ids = []
    return jsonify({'chat_ids': current_chat_ids})
    
@app.route('/get_chat_conversations/<chat_id>', methods=['POST'])
def get_chat_conversations(chat_id):
    conn =  psycopg.connect(dbname=DB_CONFIG['dbname'], user=DB_CONFIG['user'], password=DB_CONFIG['password'], host=DB_CONFIG['host'], port=DB_CONFIG['port'])
    cur = conn.cursor()
    query = f"SELECT message FROM {DB_CONFIG['conversation_table_name']} WHERE chat_id = %s ORDER BY time_stamp ASC;"
    cur.execute(query, (chat_id,))
    rows = cur.fetchall()
    return jsonify({'messages': rows})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    chat_id = data['data']['chat_id']
    message = data['data']['input']
    postgres_history = PostGresMemoryHistory(db_host=DB_CONFIG['host'], db_name=DB_CONFIG['dbname'], db_user=DB_CONFIG['user'], db_password=DB_CONFIG['password'], db_port=DB_CONFIG['port'], table_name=DB_CONFIG['conversation_table_name'], chat_id=chat_id)
    agent = MemoryConversationalAgent(llms=llms, history=postgres_history, memory_prompt_handler=MemoryPromptHandler(), template=CONVERSATIONAL_PROMPT_TEMPLATE).get_agent()
    response = agent.run(message)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
