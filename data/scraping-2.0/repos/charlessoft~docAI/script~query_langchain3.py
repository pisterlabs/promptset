# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import numpy as np
import os
import json
# from utils import calc_cos_similarity
import codecs
import json
import codecs
import joblib



# from langchain import OpenAI
import openai
from langchain.schema import HumanMessage

openai.api_type = "azure"
openai.api_base = "https://adt-openai.openai.azure.com/"
openai.api_version = "2022-12-01"
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY", '938ce9d50df942d08399ad736863d063')

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://adt-openai.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "938ce9d50df942d08399ad736863d063"

OPENAI_API_KEY="938ce9d50df942d08399ad736863d063"
from langchain.llms import AzureOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.callbacks import get_openai_callback
import os


QUERIES = [
    "如何使用foxit sdk 打开文档?",
]

def track_tokens_usage(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Total tokens: {cb.total_tokens}')
        print(f'Requests: {cb.successful_requests}')
        # tokens.append(cb.total_tokens)
        # requests.append(cb.successful_requests)

    return result


from typing import List
openai.api_type = "azure"
openai.api_base = "https://adt-openai.openai.azure.com/"
openai.api_version = "2022-12-01"
# openai.api_version = "2023-03-15-preview"
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY", '938ce9d50df942d08399ad736863d063')

from langchain.llms import AzureOpenAI


class NewAzureOpenAI(AzureOpenAI):
    stop: List[str] = '<|im_end|>'
    @property
    def _invocation_params(self):
        params = super()._invocation_params
        # fix InvalidRequestError: logprobs, best_of and echo parameters are not available on gpt-35-turbo model.
        params.pop('logprobs', None)
        params.pop('best_of', None)
        params.pop('echo', None)
        params['stop'] = self.stop
        return params


llm = NewAzureOpenAI(
    temperature=0,
    openai_api_key=os.environ["OPENAI_API_KEY"],
    engine="ChatGPT-0301"
)
from langchain import PromptTemplate, LLMChain

first_template = """
You are Foxiter, working on Foxit as a senior C++ developer, when given context, can answer questions using only that information and generate relevant code.
content: {CONTEXT}
{QUERY} '<|im_end|>'
AI: 
"""


prompt_template=PromptTemplate(
    input_variables=["CONTEXT","QUERY"],
    template=first_template
)

mem = ConversationBufferMemory()
# # conversation = ConversationChain(llm=llm, memory = mem,prompt=prompt_template)
#
chain = LLMChain(llm=llm, prompt=prompt_template)
#
s=chain.run(CONTEXT='How to load an existing PDF document from file path "Sample.pdf" with password "123" ?\n #include "include/pdf/fs_pdfdoc.h" \n using namespace std;\n using namespace foxit; \n using namespace common; \n using namespace pdf; \n \n const char* sn = "";\n const char* key = "";\n Library::Initialize(sn,  key);\n \n {\n PDFDoc doc("Sample.pdf"); \n ErrorCode error_code = doc.Load("123"); \n if (error_code != foxit::e_ErrSuccess) {\n cout << "Load document error!" << endl;\n }\n }\n \n Library::Release();\n',
                QUERY="如何打开pdf?")
print(s)




# second_template = """You are Foxiter, working on Foxit as a senior C++ developer. If the AI does not know the answer to a question, it truthfully says it does not know.
# second_template = """You are Foxiter, working on Foxit as a senior C++ developer. The AI is talkative and provides lots of specific details from its context.
second_template = """You are Foxiter, working on Foxit as a senior C++ developer, when given context, can answer questions using only that information and generate relevant code.

Current conversation:
{history}
Human: {input}
AI:"""

prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template=second_template
)

mem.chat_memory.add_user_message("如何打开pdf?")
mem.chat_memory.add_ai_message(s)

chain = ConversationChain(llm=llm, memory=mem,prompt=prompt_template)


print(chain.run("Library::Initialize需要传什么参数?<|im_end|>"))
# chain.run(CONTEXT='How to load an existing PDF document from file path "Sample.pdf" with password "123" ? #include "include/pdf/fs_pdfdoc.h"using namespace std; using namespace foxit; using namespace common; using namespace pdf; const char* sn = ""; const char* key = ""; Library::Initialize(sn,  key); {PDFDoc doc("Sample.pdf"); ErrorCode error_code = doc.Load("123"); if (error_code != foxit::e_ErrSuccess) {cout << "Load document error!" << endl; } }',
#                           QUERY="how to open pdf?<|im_end|>")
print(mem.buffer)
HumanMessage
print(chain.run("what did i just ask you?<|im_end|>"))
# mem.buffer=""

print(chain.run("what did i just ask you?<|im_end|>"))

