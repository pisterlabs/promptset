from loguru import logger
from pydantic import BaseModel, root_validator
from typing import Optional, List
from retry import retry
import openai
import tiktoken
from time import time

encoder = tiktoken.get_encoding('cl100k_base')

class ChatRoleMessage(BaseModel):
    role: str
    text: str
    tokens: Optional[int]

    @root_validator
    def compute_tokens(cls, values) -> int:
        _text = f'{values["role"]}\n{values["text"]}' 
        values["tokens"] = len(encoder.encode(_text))
        values["tokens"] += 2    # XXX validate/model unknown overhead
        return values


class SystemMessage(ChatRoleMessage):
    role = 'system'
    text: str

class ContextMessage(ChatRoleMessage):    
    role = 'system'    
    text: str
    
class AssistantMessage(ChatRoleMessage):
    role = 'assistant'    
    text: str

class UserMessage(ChatRoleMessage):
    role = 'user'
    text: str


class ChatContext:

    def __init__(self,
                 min_response_tokens=200,           # minimum number of tokens to reserve for model completion response;  max input context will be (4096 - min_response_tokens)
                 max_response_tokens=400,           # maximum number of tokens to allow for model completion response
                 max_context_assistant_messages=5,  # number of recent assistant messages to keep in context
                 max_context_user_messages=50,      # number of recent user messages to keep in context
                 model="gpt-3.5-turbo",
                 temperature=0.5,
                 base_system_msg_text="You are Jingbot, a clever bot.  Do not apologize, or make excuses.  Do not mention that you are an AI language model since that is annoying to users."):
        """            
        
        """
        self.model = model
        self.temperature = temperature
        self.base_system_msg = SystemMessage(text=base_system_msg_text)
        self.max_model_context = 4096
        self.min_response_tokens = min_response_tokens
        self.max_response_tokens = max_response_tokens
        self.max_context_assistant_messages = max_context_assistant_messages
        self.max_context_user_messages = max_context_user_messages
        self.messages = []    # reverse chronological order, newest messages first

    def _compose_completion_msg(self) -> List[ChatRoleMessage]:
        # assemble the input messages subject to the following constraints:
        # must leave room for min_response_tokens in the context
        # maximum of max_context_assistant_messages assistant messages
        # maximum of max_context_user_messages user messages
        max_input_context = self.max_model_context - self.min_response_tokens
        messages = []
        current_input_tokens = self.base_system_msg.tokens
        current_user_messages = 0
        current_assistant_messages = 0
        for msg in self.messages:
            tokens = msg.tokens
            if current_input_tokens + tokens > max_input_context:
                break
            if msg.role == 'assistant':
                if current_assistant_messages < self.max_context_assistant_messages:
                    messages.append(msg)
                    current_assistant_messages += 1
                    current_input_tokens += msg.tokens
            elif msg.role == 'user':
                if current_user_messages < self.max_context_user_messages:
                    messages.append(msg)
                    current_user_messages += 1
                    current_input_tokens += msg.tokens
        messages.append(self.base_system_msg)
        messages.reverse()
        return messages


    @retry(tries=10, delay=.05)
    def _completion(self, msgs :ChatRoleMessage) -> str:
        for msg in msgs:
            logger.info(f"completion message: role {msg.role}: '{msg.text}'")

        messages = [{"role": msg.role, "content": msg.text} for msg in msgs]
        
        try:
            t0 = time()
            response =  openai.ChatCompletion.create(model = self.model,
                                                    messages = messages,
                                                    max_tokens = self.max_response_tokens,
                                                    temperature = self.temperature)
            dt = time() - t0
            logger.info(f'completion time: {dt:.3f} s')
        except Exception as e:
            logger.exception(e)
            raise
        
        logger.debug(f'completion response: {response}')        
        response_text = response['choices'][0]['message']['content']
        logger.info(f'completion response: {response_text}')
        self.messages.insert(0, AssistantMessage(text=response_text))
        return response_text

    @retry(tries=10, delay=.05, ExceptionToRaise=openai.error.InvalidRequestError)
    async def _completion_stream(self, msgs :ChatRoleMessage) -> str:
        oai_messages = [{"role": msg.role, "content": msg.text} for msg in msgs]
        try:
            r_gen = await openai.ChatCompletion.acreate(model=self.model,
                                                        messages=oai_messages,
                                                        stream=True,
                                                        max_tokens = self.max_response_tokens)
            response = ""
            async for r_item in r_gen:
                delta = r_item.choices[0].delta.get('content', '')
                if delta:
                    response += delta
                    yield "not_finished", response
            
        except openai.error.RateLimitError as e:   
            logger.warning(f"OpenAI RateLimitError: {e}")
            raise
        except openai.error.InvalidRequestError as e: # too many token
            logger.error(f"OpenAI InvalidRequestError: {e}")
            raise
        except Exception as e:
            logger.exception(e)
            raise
        response = response.strip()
        self.messages.insert(0, AssistantMessage(text=response))
        yield "finished", response


    async def user_message_stream(self, msg_text) -> str:
        msg = UserMessage(text=msg_text)
        # put message at beginning of list
        self.messages.insert(0, msg)
        msgs = self._compose_completion_msg()
        for msg in msgs:
            logger.info(f"completion message: role {msg.role}: '{msg.text}'")
        gen = self._completion_stream(msgs)
        async for gen_item in gen:
            yield gen_item


    
    def user_message(self, msg_text) -> str:
        msg = UserMessage(text=msg_text)
        # put message at beginning of list
        self.messages.insert(0, msg)
        response_text = self._completion(self._compose_completion_msg())
        return response_text
    
        
