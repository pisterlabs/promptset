# This chat variant determines if the user's query is related to a widget or a search
import re
import time
import json
import uuid
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, Generator, List, Optional, Union, Literal, TypedDict, Callable

from gpt_index.utils import ErrorToRetry, retry_on_exceptions_with_backoff
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseMessage

import context
import utils
import utils.timing as timing
from utils.common import FUNCTIONS, modelname_to_contextsize, get_user_info
from utils.constants import WIDGET_INFO_TOKEN_LIMIT
import registry
import streaming
from .base import (
    BaseChat, Response, ChatHistory
)
from ui_workflows import (
    aave, ens
)
from ui_workflows.multistep_handler import register_ens_domain, exec_aave_operation
from tools.index_widget import *

SYSTEM_MESSAGE_DEFAULT = \
"""You are an agent named Cacti, that is trained to execute functions based on a user request. If you found a suitable function but not all the input parameters are known, ask for them. Otherwise just proceed with calling the function without reconfirming.

Here is the user personal information, which you can use as input parameters of the functions. 
# USER INFO:
{user_info}"""

@registry.register_class
class ChatGPTFunctionCallChat(BaseChat):
    def __init__(self, widget_index: Any, model_name: Optional[str] = "gpt-3.5-turbo-0613", top_k: int = 5, evaluate_widgets: bool = True) -> None:
        super().__init__()
        self.widget_index = widget_index
        self.model_name = model_name
        self.top_k = top_k
        self.evaluate_widgets = evaluate_widgets  # this controls whether we want to execute widgets, set to false to get the raw command back
        self.system_message = SYSTEM_MESSAGE_DEFAULT
        self.token_limit = max(1800, modelname_to_contextsize(model_name) - WIDGET_INFO_TOKEN_LIMIT)

    def receive_input(
            self,
            history: ChatHistory,
            userinput: str,
            send: Callable,
            message_id: Optional[uuid.UUID] = None,
            before_message_id: Optional[uuid.UUID] = None,
    ) -> None:
        with context.with_request_context(history.wallet_address, message_id):
            self.system_message = self.system_message.replace("{user_info}", get_user_info(not self.evaluate_widgets))

        userinput = userinput.strip()
        history.add_user_message(userinput, message_id=message_id, before_message_id=before_message_id)

        history_messages = history.to_openai_messages(system_message=self.system_message, system_prefix=None, token_limit=self.token_limit, before_message_id=before_message_id)  # omit system messages

        timing.init()

        bot_chat_message_id = None
        bot_response = ''
        has_sent_bot_response = False

        def bot_flush(response):
            nonlocal bot_chat_message_id
            response = response.strip()
            send(Response(
                response=response,
                still_thinking=False,
                actor='bot',
                operation='replace',
            ), last_chat_message_id=bot_chat_message_id, before_message_id=before_message_id)
            history.add_bot_message(response, message_id=bot_chat_message_id, before_message_id=before_message_id)

        def bot_new_token_handler(token):
            nonlocal bot_chat_message_id, bot_response, has_sent_bot_response

            bot_response += token
            if not bot_response.strip():
                # don't start returning something until we have the first non-whitespace char
                return

            timing.log('first_visible_bot_token')
            bot_chat_message_id = send(Response(
                response=token,
                still_thinking=False,
                actor='bot',
                operation='append' if bot_chat_message_id is not None else 'create',
            ), last_chat_message_id=bot_chat_message_id, before_message_id=before_message_id)
            has_sent_bot_response = True

        new_token_handler = bot_new_token_handler
        response_buffer = ""

        def injection_handler(token):
            nonlocal new_token_handler, response_buffer

            timing.log('first_token')
            timing.log('first_widget_token')  # for comparison with basic agent

            response_buffer += token

            # although this is for gpt functions, we also handle the case where the bot message
            # might come back as a widget command in string form.
            while WIDGET_START in response_buffer and self.evaluate_widgets:
                if WIDGET_END in response_buffer:
                    # parse fetch command
                    response_buffer = iterative_evaluate(response_buffer)
                    if isinstance(response_buffer, Callable):  # handle delegated streaming
                        def handler(token):
                            nonlocal new_token_handler
                            timing.log('first_visible_widget_response_token')
                            return new_token_handler(token)
                        response_buffer(handler)
                        response_buffer = ""
                        return
                    elif isinstance(response_buffer, Generator):  # handle stream of widgets
                        for item in response_buffer:
                            timing.log('first_visible_widget_response_token')
                            new_token_handler(str(item) + "\n")
                        response_buffer = ""
                        return
                    elif len(response_buffer.split(WIDGET_START)) == len(response_buffer.split(WIDGET_END)):
                        # matching pairs of open/close, just flush
                        # NB: for better frontend parsing of nested widgets, we need an invariant that
                        # there are no two independent widgets on the same line, otherwise we can't
                        # detect the closing tag properly when there is nesting.
                        response_buffer = response_buffer.replace(WIDGET_END, WIDGET_END + '\n')
                        break
                    else:
                        # keep waiting
                        return
                else:
                    # keep waiting
                    return
            if 0 < len(response_buffer) < len(WIDGET_START) and WIDGET_START.startswith(response_buffer):
                # keep waiting if we could potentially be receiving WIDGET_START
                return
            token = response_buffer
            response_buffer = ""
            if token.strip():
                timing.log('first_visible_widget_response_token')
            new_token_handler(token)

        if self.widget_index is None:
            functions = FUNCTIONS
        else:
            widgets = retry_on_exceptions_with_backoff(
                lambda: self.widget_index.similarity_search(userinput, k=self.top_k),
                [ErrorToRetry(TypeError)],
            )
            function_names = [fn['name'] for fn in FUNCTIONS]
            functions = []
            for w in widgets:
                fn_name = '_'.join(RE_COMMAND.search(w.page_content.replace('{', '').replace('}', '')).group('command').split('-'))
                try:
                    idx = function_names.index(fn_name)
                except ValueError:
                    continue
                functions.append(FUNCTIONS[idx])

        llm = streaming.get_streaming_llm(injection_handler, model_name=self.model_name)

        with context.with_request_context(history.wallet_address, message_id):
            ai_message = llm.predict_messages(history_messages, functions=functions)

            if 'function_call' in ai_message.additional_kwargs:
                # when there is a function call, the callback is not called, so we process it
                # here and call it ourselves with the widget str
                function_call = ai_message.additional_kwargs['function_call']
                if function_call['name'].startswith('functions.'):
                    function_call['name'] = function_call['name'][len('functions.'):]
                function_message_id = send(Response(
                    response=json.dumps(function_call),
                    still_thinking=True,
                    actor='function',
                ), before_message_id=before_message_id)
                bot_chat_message_id = None
                command = '-'.join(function_call['name'].split('_'))
                params = ','.join(json.loads(function_call['arguments']).values())
                widget_str = f"{WIDGET_START}{command}({params}){WIDGET_END}"
                injection_handler(widget_str)

        timing.log('response_done')

        if bot_chat_message_id is not None:
            bot_flush(bot_response)

        response = f'Timings - {timing.report()}'
        system_chat_message_id = send(Response(response=response, actor='system'), before_message_id=before_message_id)
        history.add_system_message(response, message_id=system_chat_message_id, before_message_id=before_message_id)
