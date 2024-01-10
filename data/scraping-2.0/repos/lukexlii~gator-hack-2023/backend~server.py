from flask import Flask, jsonify, request, session, Response, stream_with_context
from flask_cors import CORS
from flask_session import Session
from flask_socketio import SocketIO
from dotenv import load_dotenv
import os

load_dotenv()

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os

faiss_index = FAISS.load_local("data/vecIndex", OpenAIEmbeddings())

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains import RetrievalQA
from langchain.agents import Tool

from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool, StructuredTool

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent
from langchain.load.dump import dumps

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}},  supports_credentials=True)

app.secret_key = os.getenv('SECRET_KEY')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'

Session(app)
socketio = SocketIO(app)

@app.route('/api/leftMenu', methods=['GET'])
def get_left_menu():
  return {"leftMenuItems":'[{"heading":"Verizon","items":[{"label":"Start New Chat","route":"/new"}]}]'}, 200

def parseOutput(string):
    start = string.find('output') + 10
    return string[start: len(string) - 2]

@app.route('/api/assistant', methods=['POST'])
def get_assistant():
    assistant = request.json.get('assistant')
    #if 'agent' not in session:
    llm=ChatOpenAI(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        temperature=.6,
        model_name='gpt-4',
        streaming=False
    )


    retriever = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=faiss_index.as_retriever(search_kwargs={'k': 3})
    )

    # Define Verizon DB tool
    tool_desc = """Use this tool to answer user questions Verizon phone plans.
    If the user asks any questions about different phone plans or perks, use this tool.
    This tool can also be used for follow up questions from
    the user.  Use this tool for most responses, except when the user asks to check out."""
    verizonTool = Tool(
        func=retriever.run,
        description=tool_desc,
        name='verizonDB'
    )

    memory = ConversationBufferWindowMemory(
    memory_key="chat_history",  # important to align with agent prompt (below)
    k=5,
    return_messages=True
    )

    tools = [verizonTool]
    conversational_agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=False,
        max_iterations=2,
        early_stopping_method="generate",
        memory=memory,
        handle_parsing_errors=True
    )
    memory.save_context({"input": "Do you get it?"},
    {"output": """Yeah I get it. I'm {assistant}. Yes, I can help answer questions about Verizon phone plans.
    I will respond with the personality you specify, in JSON format.
    No repeating info or mentioning tools I use.
    I'll be quick and to the point, while being clever and out of the box."""})

    if assistant == "Deadpool":
        catch_phrases = [
        "Maximum effort!",
        "Chimichangas!",
        "I'm touching myself tonight.",
        "With great power comes great irresponsibility.",
        "Did I leave the stove on?",
        "Time to make the chimi-freakin'-changas!",
        "I'm the merc with a mouth.",
        "I know, right?",
        "You can't buy love, but you can rent it for three minutes!",
        "Fourth wall break inside a fourth wall break? That's like... sixteen walls.",
        "Hi, I'm Wade Wilson, and I'm here to talk to you about testicular cancer.",
        "Smells like old lady pants in here.",
        "Say the magic words, fat Gandalf.",
        "Superhero landing!",
        "So dark! Are you sure you're not from the DC Universe?"
    ]

    elif assistant == "Shrek":
        catch_phrases = [
        "Ogres are like onions.",
        "Better out than in, I always say.",
        "I'm not a puppet. I'm a real boy!",
        "What are you doing in my swamp?",
        "This is the part where you run away.",
        "Do you know what that thing could do? It'll grind your bones for its bread.",
        "I got this.",
        "I like that boulder. That is a nice boulder.",
        "You're going the right way for a smacked bottom.",
        "I'm making waffles!"
    ]
    elif assistant == "Ron Burgundy":
        catch_phrases = [
        "Stay classy, San Diego.",
        "I'm kind of a big deal.",
        "I'm Ron Burgundy?",
        "You stay classy.",
        "I love scotch. Scotchy, scotch, scotch.",
        "Don't act like you're not impressed.",
        "It's so damn hot. Milk was a bad choice.",
        "I immediately regret this decision.",
        "60% of the time, it works every time.",
        "Well, that escalated quickly.",
        "I'm in a glass case of emotion!",
        "What is this? A center for ants?",
        "I'm not a baby, I am a man! I am an anchorman!",
        "I look good. I mean, really good. Hey everyone! Come and see how good I look!",
        "You're so wise. You're like a miniature Buddha, covered in hair."
    ]
    elif assistant == "Zoolander":
        catch_phrases = [
            "What is this? A center for ants?",
            "I'm pretty sure there's a lot more to life than being really, really, ridiculously good looking.",
            "I feel like I'm taking crazy pills!",
            "Blue Steel!",
            "Magnum!",
            "Moisture is the essence of wetness, and wetness is the essence of beauty.",
            "So hot right now!",
            "He's absolutely right.",
            "I'm not an ambi-turner.",
            "It's a walk-off!",
            "Obey my dog!",
            "I invented the piano key necktie!",
            "Who you tryin' to get crazy with, ese? Don't you know I'm loco?",
            "Listen to your friend Billy Zane. He's a cool dude."
        ]
    else:
        catch_phrases = ["."]
    
    sys_msg = f"""You are a helpful chatbot that answers the user's questions about Verizon phone plans.
    You will respond with the personality of {assistant}.
    Remember to always return your result in the JSON format, or else I won't be able to understand you!.
    Do not repeat information.
    Do not use the name of your tools, I dont want to hear them.
    Make your reponses as short as possible, being effective at getting to the point while providing the necessary info.
    Use catch phrases: {', '.join(catch_phrases)} but don't overuse them, be creative and clever!
    If the user is ready to checkout, or if they indicate they are satisified with their plan, ask them for their name and email address.  
    Makes sure to include the keywords Purchase Confirmed when the user is done checking out so that we know the process is done. Remember to use the JSON format for this. There is no tool for this.
    """

    prompt = conversational_agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
    )
    conversational_agent.agent.llm_chain.prompt = prompt
    session['agent'] = conversational_agent
    
    #chat = ""
    response = session['agent']("Introduce yourself and ask me if I am a returning Verizon customer or not and let the user know that they can ask any questions to clarify things.")
    '''for message in response['action_input']:
        chat += message
        socketio.emit('data', chat)
    '''
    '''for item in session['agent']:
        if hasattr(item.choices[0].delta, "action_input"):
            answer_text = item.choices[0].delta.content
            self.chats[self.current_chat][-1].answer += answer_text
            self.chats = self.chats
            yield
            '''
    return {'role': assistant, 'content': parseOutput(str(response))}, 200

import time
@app.route('/api/stream')
def generate_stream():
    data = '5G, or "Fifth Generation," represents the latest advancement in wireless communication technology. It is the successor to 4G (LTE) and is designed to provide significantly faster data speeds, lower latency, increased network capacity, and improved reliability compared to previous generations of wireless networks.'
    def generate():
        for char in data:
            yield char
            time.sleep(0.5)
    return Response(generate(), {"Content-Type": "text/plain"})
    
# Chatbox API CALL
@app.route('/api/chat', methods=['POST'])
def chatbox():
    #get query string
    prompt = request.json.get('prompt')
    assistant = request.json.get('assistant')

    if prompt is None or not isinstance(prompt, str):
        return jsonify(error="Missing or invalid prompt"), 400

    response = session['agent'](prompt)
    confirmed = False
    if str(response).find('Purchase Confirmed') != -1:
        confirmed = True
        print(confirmed)


    return {"role": assistant, "content": parseOutput(str(response)), 'confirmed': confirmed}, 200



if __name__ == "__main__":
    app.run(debug=True)