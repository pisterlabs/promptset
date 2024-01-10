import os
from dotenv import load_dotenv, find_dotenv

from threading import Thread
from typing import Any
from queue import Queue

from flask import Flask, render_template, request, Response
from langchain.schema import AgentFinish
from waitress import serve

from langchain.callbacks.streaming_stdout_final_only import StreamingStdOutCallbackHandler
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents.agent_toolkits import GmailToolkit
from langchain.schema.messages import SystemMessage


load_dotenv(find_dotenv())



# use a threaded generator to return response in a stream
class ThreadedGenerator:
    def __init__(self):
        self.queue = Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)


# a callback handler to send the tokens to the generator
class callback(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen
        self.response = ""
        self.sent = False

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:        
        if self.response.endswith("\"action\": \"Final Answer\",\n  \"action_input\": \""):
            self.sent = True
            self.gen.send(token)      
        else:            
            self.response += token

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        if not self.sent:
            self.gen.send(finish.return_values["output"])
        

# tools for the agent
tools = [ DuckDuckGoSearchRun()] + GmailToolkit().get_tools()
# memory for the agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatOpenAI(streaming=True, 
                    model="gpt-4-32k",
                    max_tokens=4096,
                    temperature=0.2,
                    verbose=True)

agent = initialize_agent(tools, llm, 
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors="Check your output and make sure it conforms!",
    memory=memory)
agent.verbose=True

print(agent)

# setup the thread to run the LLM query
def llm_thread(g, query):
    cb = callback(g)
    try:
        resp = agent(inputs=query, callbacks=[cb])
    finally:
        g.close()

# run the query in a thread
def llm_run(query):
    g = ThreadedGenerator()
    Thread(target=llm_thread, args=(g, query)).start()
    return g

# get path for static files
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_dir):
    static_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'static')

# start server
print("\033[96mStarting Storm at http://127.0.0.1:1339\033[0m")
storm = Flask(__name__, static_folder=static_dir, template_folder=static_dir)

# server landing page
@storm.route('/')
def landing():
    memory.clear()
    return render_template('index.html')

# run
@storm.route('/run', methods=['POST'])
def run():
    data = request.json
    return Response(llm_run(data['input']), mimetype='text/event-stream')    

if __name__ == '__main__':
    print("\033[93mStorm started. Press CTRL+C to quit.\033[0m")
    serve(storm, port=1339, threads=16)
