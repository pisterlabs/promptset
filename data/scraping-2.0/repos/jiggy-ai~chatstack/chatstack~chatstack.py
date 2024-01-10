from loguru import logger
from pydantic import BaseModel, root_validator
from typing import Optional, List
from .retry import retry
import openai
import tiktoken
from time import time
import sys
from .pricing import price
from typing import List, Optional, Tuple

encoder = tiktoken.get_encoding('cl100k_base')

class ChatRoleMessage(BaseModel):
    role: str
    text: str
    tokens: Optional[int]

    def content(self) -> str:
        return self.text
    
    @root_validator
    def compute_tokens(cls, values) -> int:
        _text = f'{values["role"]}\n{values["text"]}' 
        values["tokens"] = len(encoder.encode(_text))
        values["tokens"] += 4    # per https://platform.openai.com/docs/guides/chat/managing-tokens
        return values


class SystemMessage(ChatRoleMessage):
    role = 'system'
    text: str

class ContextMessage(ChatRoleMessage):    
    """
    A message added to the model input context to provide context for the model.
    Generally contains source material from a search function
    Includes a prefix header to provide additional context to the model as to the nature of the content
    """
    role = 'system'
    prefix: str    
    text: str

    def content(self) -> str:
        return f'{self.prefix}: {self.text}'
    
    @root_validator
    def compute_tokens(cls, values) -> int:
        _text = f'{values["role"]}\n{values["prefix"]}: {values["text"]}'
        values["tokens"] = len(encoder.encode(_text))
        values["tokens"] += 4    # per https://platform.openai.com/docs/guides/chat/managing-tokens
        return values    

    
class AssistantMessage(ChatRoleMessage):
    role = 'assistant'    
    text: str

class UserMessage(ChatRoleMessage):
    role = 'user'
    text: str


class ChatResponse(BaseModel):
    """
    returned in response to a User Message
    contains the complete state of the input context as sent to the model as well as the response from the model
    and other details such as the model used, the number of tokens used, and the estimated cost of the request
    """
    text:               str                             # the completion response from the model
    delta:              Optional[str]                   # the response text delta for streaming case
    model:              str                             # the model used to generate the response
    temperature:        float                           # the temperature used to generate the response
    inputs:             List[ChatRoleMessage]           # the input context as sent to the model    
    input_tokens:       int                             # the number of tokens used in the input context
    response_tokens:    int                             # the number of tokens in the response text.  only valid on last output for streaming
    price:              float                           # the estimated price of the request in dollars. only valid on last output for streaming


GPT3_MODELS = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301']
GPT4_MODELS = ['gpt-4', 'gpt-4-0314', 'gpt-4-1106-preview']

class ChatContext:

    def __init__(self,
                 base_system_msg_text : str = "Please respond to the user request.",  # base system message to use for context
                 min_response_tokens : int = 200,           # minimum number of tokens to reserve for model completion response;  
                 max_response_tokens : int = 400,           # maximum number of tokens to allow for model completion response; set to None to allow unlimited use of the model context
                 chat_context_messages : int = 50,          # number of recent user/assistant messages to keep in context
                 model : str = "gpt-3.5-turbo",             # model to use for completion
                 temperature : float = 0.5):                # temperature to use for model completion
                 
        self.model = model
        self.temperature = temperature
        self.base_system_msg = SystemMessage(text=base_system_msg_text)
        if model in GPT3_MODELS :
            self.max_model_context = 4096
        elif model in GPT4_MODELS:
            self.max_model_context = 8192
        else:
            raise ValueError(f"Model {model} not supported. Supported models are {GPT3_MODELS} and {GPT4_MODELS}")        
        self.min_response_tokens = min_response_tokens
        self.max_response_tokens = max_response_tokens
        self.chat_context_messages = chat_context_messages
        self.messages = []    # reverse chronological order, newest messages first


    def _assemble_completion_msgs(self, dynamic_context : list[ContextMessage]) -> List[ChatRoleMessage]:
        """
        assemble the input messages subject to the following constraints:
        must leave room for min_response_tokens in the context
        maximum of chat_context_messages user/assistant messages
        dynamic_context fills the remaining space
        returns list of finalized ChatRoleMessage to feed to model as well as the raw dynamic completion text for future matching purposes
        """        
        max_input_context = self.max_model_context - self.min_response_tokens
        #logger.info(f'maximum input context: {max_input_context} tokens')
        chat_messages = []
        chat_tokens = 0
        for msg in self.messages[:self.chat_context_messages]:
            chat_messages.append(msg)
            chat_tokens += msg.tokens
        chat_messages.reverse()   # put chat messages back in chronological order
        #logger.info(f'chat messages: {chat_tokens} tokens')
        
        # compute the maximum number of tokens to use for dynamic context
        max_dynamic_context = max_input_context - chat_tokens - self.base_system_msg.tokens
        #logger.info(f'maximum dynamic context: {max_dynamic_context} tokens')
                
        # assemble dynamic context up to the token limit
        dynamic_context_messages = []
        dynamic_context = dynamic_context if dynamic_context else []        
        for msg in dynamic_context:
            if msg.tokens > max_dynamic_context:
                break            
            dynamic_context_messages.append(msg)
            max_dynamic_context -= msg.tokens
            
        # compose final list of messages
        messages = [self.base_system_msg]            
        messages.extend(dynamic_context_messages)
        messages.extend(chat_messages)
        # log estimates messages tokens
        #logger.info(f'estimated messages tokens: {sum([msg.tokens for msg in messages])}')
        return messages


    @retry(tries=10, delay=.05, ExceptionToRaise=openai.BadRequestError)
    def _completion(self, msgs :ChatRoleMessage) -> str:

        messages = [{"role": msg.role, "content": msg.content()} for msg in msgs]
        
        try:
            t0 = time()
            response =  openai.ChatCompletion.create(model = self.model,
                                                     messages = messages,
                                                     max_tokens = self.max_response_tokens,
                                                     temperature = self.temperature)
        except openai.error.RateLimitError as e:
            logger.info(f'OpenAI RateLimitError, retrying...')
            raise
        except Exception as e:
            logger.warning(f'Exception during openai.ChatCompletion.create: {e}, retrying...')
            raise            
        response_text = response['choices'][0]['message']['content']
        dt = time() - t0
        #logger.info(f'completion time: {dt:.3f} s')                                                    
        return response_text

    def add_message(self, msg : ChatRoleMessage):
        """
        Add a message to the context for presentation to the model in subsequent completion requests.
        Does not result in a model completion.
        """
        self.messages.insert(0, msg)
        
    def user_message(self, msg_text : str, dynamic_context : Optional[list[ContextMessage]] = None) -> ChatResponse:
        msg = UserMessage(text=msg_text)
        # put message at beginning of list
        self.messages.insert(0, msg)
        completion_msgs = self._assemble_completion_msgs(dynamic_context)
        response_text = self._completion(completion_msgs)
        response_msg = AssistantMessage(text=response_text)
        self.messages.insert(0, response_msg)
        
        input_tokens = sum([msg.tokens for msg in completion_msgs]) + 2            
        cr = ChatResponse(text = response_text,
                          model = self.model,
                          temperature=self.temperature,
                          inputs=completion_msgs,
                          input_tokens=input_tokens,
                          response_tokens=response_msg.tokens,
                          price=price(self.model, input_tokens, response_msg.tokens))
                                    
        return cr
    
    @retry(tries=10, delay=.05, ExceptionToRaise=openai.BadRequestError)
    def _completion_stream(self, msgs :ChatRoleMessage) -> ChatResponse:
        """
        return tuple of (delta, response, done):  
        where delta is the incremental response text,
        response is the cumulative response text,
        and done is True if the completion is complete
        """
        cr = ChatResponse(text = "",
                          delta = "",
                          model = self.model,
                          temperature=self.temperature,
                          inputs=msgs,
                          input_tokens=sum([msg.tokens for msg in msgs]) + 2 ,
                          response_tokens=0,
                          price=0)
        
        oai_messages = [{"role": msg.role, "content": msg.content()} for msg in msgs]
        try:
            response = openai.chat.completions.create(model=self.model,
                                                      messages=oai_messages,
                                                      stream=True,
                                                      max_tokens = self.max_response_tokens)
            for chunk in response:
                output = chunk.choices[0].delta.content or ''
                if output:
                    cr.delta = output
                    cr.text += output
                    yield cr
            
        except openai.RateLimitError as e:   
            logger.warning(f"OpenAI RateLimitError: {e}")
            raise
        except openai.BadRequestError as e: # too many token
            logger.error(f"OpenAI BadRequestError: {e}")
            raise
        except Exception as e:
            logger.exception(e)
            raise
        resp_msg = AssistantMessage(text=cr.text)
        self.messages.insert(0, resp_msg)  # add response to context
        cr.response_tokens=resp_msg.tokens
        cr.price=price(self.model, cr.input_tokens, resp_msg.tokens)
        yield cr

    def user_message_stream(self, msg_text, dynamic_context : Optional[list[ContextMessage]] = None) -> ChatResponse:
        msg = UserMessage(text=msg_text)
        # put message at beginning of list
        self.messages.insert(0, msg)
        completion_msgs = self._assemble_completion_msgs(dynamic_context)
        for cr in self._completion_stream(completion_msgs):
            yield cr
        
    
        
