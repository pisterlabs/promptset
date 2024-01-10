import sys
import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

sys.path.append(f'{os.path.dirname(__file__)}/../')
from botcore.chains.ask_feature import build_ask_feature_chain
from botcore.setup import trace_ai21

MODEL = trace_ai21()


ask_feature = build_ask_feature_chain(MODEL)

product = "washing machine"
features = ask_feature({"product": product, "n_top": 5})
print(features)