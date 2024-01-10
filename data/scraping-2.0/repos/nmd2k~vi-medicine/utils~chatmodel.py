import os
import sys
import openai
import requests
import json

from typing import Dict, Any

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import StreamingStdOutCallbackHandler

from utils.utils import latency_benchmark
from config.OAIConfig import API_BASE, \
                            API_VERSION, \
                            MODEL_DEPLOYMENT, \
                            OPENAI_API_KEY
                

DUMMY_PROMPT = ("You are useful assistant. "
                "Follow instruction and answer user question""")
OPENAI_AVAILABLE = True
try:
    import openai
    openai.api_type = "azure"
    openai.api_key = OPENAI_API_KEY
    openai.api_base = API_BASE
    openai.api_version = API_VERSION
except ModuleNotFoundError:
    OPENAI_AVAILABLE = False


class ChatModel:
    def __init__(self, 
        chatmodel: str = "openai",
        temperature: int = 0,
        stream: bool = False,
        frequency_penalty: float = 0.2,
        max_tokens: int = 2000,
        top_p: float = 1.0,
        presence_penalty: float = 1.0,
        stop_sequences: str = None,
        n_retry: int = 3,
        request_timeout = 60,
        verbose: bool = True) -> None:
        '''Setup chat model
        '''
        self.top_p              = top_p 
        self.stream             = stream
        self.max_tokens         = max_tokens
        self.temperature        = temperature
        self.frequency_penalty  = frequency_penalty
        self.presence_penalty   = presence_penalty
        self.stop_sequences     = stop_sequences
        self.verbose            = verbose
        self.request_timeout    = request_timeout
        self.max_retries        = n_retry
        
        if chatmodel == "openai":
            assert OPENAI_AVAILABLE, "Missing openai module, try `pip install openai`"
            self.generative_model = AzureChatOpenAI(
                openai_api_base     = API_BASE,
                openai_api_version  = API_VERSION,
                openai_api_key      = OPENAI_API_KEY,
                deployment_name     = MODEL_DEPLOYMENT,
                max_retries         = self.max_retries,
                request_timeout     = self.request_timeout,
                streaming           = self.stream,
                max_tokens          = self.max_tokens,
                temperature         = self.temperature,
                model_kwargs        = {
                    "top_p": self.top_p,
                    "stop" : self.stop_sequences,
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty
                }
            )
            self.embedding_model = OpenAIEmbeddings(
                openai_api_type="azure",
                openai_api_base=API_BASE,
                openai_api_key=OPENAI_API_KEY,
                model=MODEL_DEPLOYMENT,
            )
            self.generate_function = self._openai_generate_function
            self.embedding_function = self._openai_embedding_function
        else:
            raise NotImplemented("Not implemented other model caller")
            self.generate_function = self._llama_generate_function
    
    def _openai_generate_function(self, message):
        if self.stream:
            response = self.generative_model(
                messages=message, callbacks=[StreamingStdOutCallbackHandler()]) 
        else:
            response = self.generative_model(messages=message)
        return response
    
    def _openai_embedding_function(self, message):
        query_result = self.embedding_model.embed_query(message)
        return query_result
    
    def _llama_generate_function(self, message):
        #     chat = requests.request(
        #         "POST", os.getenv["LLAMA2_API_SERVICE"], 
        #         headers={'Content-Type': 'application/json'}, 
        #         data=message
        #     )
        #     reply = json.loads(chat.text)['generation']['content']
        raise NotImplemented("Method not implemented")
    
    @latency_benchmark
    def embed(self, message):
        '''Request embedding model to embed given query'''
        embed = self.embedding_function(message=message)
        return embed
    
    @latency_benchmark
    def generate(self, message) -> str:
        '''Request Chatbot model to generate answer
        Chatbot using in demo is gpt-3.5-turbo from Openai'''
        
        try:
            response = self.generate_function(message=message)

            reply = response.content
            # reply = response.content
            # reply = response["choices"][0]["message"]["content"]
            if self.verbose:
                sys.stdout.write(reply)
                sys.stdout.flush()
            
        except openai.error.Timeout as e:
            raise Exception(f"OpenAI API request timed out: {e}. Please retry")
        except openai.error.APIError as e:
            raise Exception(f"OpenAI API returned an API Error: {e}. Please retry")
        except openai.error.APIConnectionError as e:
            raise Exception(f"OpenAI API request failed to connect: {e}. Please retry")
        except openai.error.InvalidRequestError as e:
            raise Exception(f"OpenAI API request was invalid: {e}. Please retry")
        except openai.error.AuthenticationError as e:
            raise Exception(f"OpenAI API request was not authorized: {e}. Please retry")
        except openai.error.PermissionError as e:
            raise Exception(f"OpenAI API request was not permitted: {e}. Please retry")
        except openai.error.RateLimitError as e:
            raise Exception(f"OpenAI API request exceeded rate limit: {e}. Please retry")
        
        return reply
