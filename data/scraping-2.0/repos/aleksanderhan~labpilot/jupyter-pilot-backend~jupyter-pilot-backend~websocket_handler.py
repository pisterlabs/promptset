import json
import tornado.web
import tornado.websocket
import tornado.ioloop
from langchain.chains.sequential import SequentialChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

from .prompt import debug_template, debug_explain_template, explain_template, refactor_template
from .callback import DefaultCallbackHandler

class RefactorWebSocketHandler(tornado.websocket.WebSocketHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cells = {}

    def check_origin(self, origin):
        # Override to enable support for allowing all cross-origin traffic
        return True

    def on_message(self, message):
        data = json.loads(message)
        code = data.get('code', 'No code provided')
        model = data.get('model', 'gpt-3.5-turbo')
        temp = data.get('temp', 1)
        cell_id = data.get("cellId", None)
        openai_api_key = data.get("openai_api_key", None)

        memory = self.cells.get(cell_id)
        if not memory:
            memory = ConversationBufferWindowMemory(k=3, memory_key="memory", return_messages=True)
            self.cells[cell_id] = memory

        llm = ChatOpenAI(openai_api_key=openai_api_key, model=model, temperature=temp, streaming=True, callbacks=[DefaultCallbackHandler(self.write_message)])
        
        prompt_template = PromptTemplate(input_variables=["memory", "code"], template=refactor_template)
        chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            verbose=True,
            memory=memory
        )
        chain({"code": code})


class DebugWebSocketHandler(tornado.websocket.WebSocketHandler):

    def check_origin(self, origin):
        # Override to enable support for allowing all cross-origin traffic
        return True

    def on_message(self, message):
        data = json.loads(message)
        code = data.get('code', 'No code provided')
        output = data.get('output', 'No output provided')
        error = data.get('error', 'No error provided')
        model = data.get('model', 'gpt-3.5-turbo')
        temp = data.get('temp', 1)
        openai_api_key = data.get("openai_api_key", None)

        llm = ChatOpenAI(openai_api_key=openai_api_key, model=model, temperature=temp, streaming=True, callbacks=[DefaultCallbackHandler(self.write_message)])
        
        debug_prompt_template = PromptTemplate(input_variables=["code", "output", "error"], template=debug_template)
        debug_chain = LLMChain(
            llm=llm,
            prompt=debug_prompt_template,
            verbose=True,
            output_key="refactored"
        )

        debug_explain_prompt_template = PromptTemplate(input_variables=["code", "output", "error", "refactored"], template=debug_explain_template)
        debug_explain_chain = LLMChain(
            llm=llm,
            prompt=debug_explain_prompt_template,
            verbose=True,
            output_key="explanation"
        )

        overall_chain = SequentialChain(
            chains=[debug_chain, debug_explain_chain],
            input_variables=["code", "output", "error"],
            # Here we return multiple variables
            output_variables=["refactored", "explanation"],
            verbose=True
        )

        overall_chain({"code": code, "output": output, "error": error})


class ExplainWebSocketHandler(tornado.websocket.WebSocketHandler):

    def check_origin(self, origin):
        # Override to enable support for allowing all cross-origin traffic
        return True

    def on_message(self, message):
        data = json.loads(message)
        code = data.get('code', 'No code provided')
        model = data.get('model', 'gpt-3.5-turbo')
        temp = data.get('temp', 1)
        openai_api_key = data.get("openai_api_key", None)

        llm = ChatOpenAI(openai_api_key=openai_api_key, model=model, temperature=temp, streaming=True, callbacks=[DefaultCallbackHandler(self.write_message)])
        
        prompt_template = PromptTemplate(input_variables=["code"], template=explain_template)
        chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            verbose=True
        )
        chain({"code": code})