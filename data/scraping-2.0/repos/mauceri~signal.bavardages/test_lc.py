from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from collections import deque
import logging

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
logging.basicConfig(level = logging.INFO)

class ChatHistory:
    def __init__(self, msg_limit):
        self.stack = deque(maxlen=msg_limit)

    def append(self, msg):
        return self.stack.append(msg)

    def get_as_list(self):
        return list(self.stack)

    def get_as_string(self):
        res = ""
        for e in self.get_as_list():
            res += res + e['role'] + ": " + e['content'] + "\n"
        return res

prompt = ChatPromptTemplate.from_template(
    """
    <|system|>: Vos réponses sont concises
    <|user|>: {q}
    """
)
# Make sure the model path is correct for your system!
llm = LlamaCpp(
#    model_path="/Users/mauceric/PRG/llama.cpp/models/7B/ggml-model-q4_0.bin",
#    model_path="./models/ggml-model-f16.gguf",
    model_path="/home/christian/PRG/llama.cpp/models/vigogne_2_7b_chat/ggml-model-f16.gguf.bin",
    temperature=1,
    max_tokens=200,
    top_p=1,
    n_ctx=4096,
    callback_manager=callback_manager, 
    verbose=True, # Verbose is required to pass to the callback manager
)

def question(history,txt):
    # Make sure the model path is correct for your system!
#    llm = LlamaCpp(
#    model_path="/Users/mauceric/PRG/llama.cpp/models/7B/ggml-model-q4_0.bin",
#    model_path="./models/ggml-model-f16.gguf",
 #   model_path="/home/christian/PRG/llama.cpp/models/vigogne_2_7b_chat/ggml-model-f16.gguf",
 #   temperature=1,
 #   max_tokens=200,
 #   top_p=1,
 #   n_ctx=4096,
 #   callback_manager=callback_manager, 
 #   verbose=True, # Verbose is required to pass to the callback manager
 #   h = history.get_as_string()
 #   um = {"role": "<|user|>", "content": txt}
 #   history.append(um)

    p = prompt.format(q=txt)
    r = llm(p)
    r = r.strip()

 #   am = {"role": "<|assistant|>", "content": r[0:4]}
 #   history.append(am)

    return r


hist = ChatHistory(4)
#m = "Je m'appelle Christian, et vous ?"
#r = question(hist,m)
#print(m+"\n"+r)
m = "Que pouvez-vous me dire du roi Henri IV de France ?"
r = question(hist,m)
print(m+"\n"+r)
#m = "Quel est mon nom ?"
#r = question(hist,m)
#print(m+"\n"+r)
m = "Que pouvez-vous me dire du du renard et du corbeau ?"
r = question(hist,m)
print(m+"\n"+r)
m = "Quelle est la fable de la Fontaine la plus célèbre ?"
r = question(hist,m)
print(m+"\n"+r)
m = "Qui était Louis-Ferdinand Céline ?"
r = question(hist,m)
print(m+"\n"+r)


