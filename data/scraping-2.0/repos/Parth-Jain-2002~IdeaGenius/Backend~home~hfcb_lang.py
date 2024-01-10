from hugchat import hugchat
import time 
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from hugchat.login import Login

import os
load_dotenv()


# THIS IS A CUSTOM LLM WRAPPER Based on hugchat library
# Reference :
# - Langchain custom LLM wrapper : https://python.langchain.com/docs/modules/model_io/models/llms/how_to/custom_llm
# - HugChat library : https://github.com/Soulter/hugging-chat-api

class HuggingChat(LLM):

    """HuggingChat LLM wrapper."""

    chatbot : Optional[hugchat.ChatBot] = None
    email: Optional[str] = None
    psw: Optional[str] = None
    cookie_path : Optional[str] = "./cookies_snapshot"
    conversation : Optional[str] = None
    model: Optional[int] = 1 # 0 = OpenAssistant/oasst-sft-6-llama-30b-xor , 1 = meta-llama/Llama-2-70b-chat-hf

    temperature: Optional[float] = 0.9
    top_p: Optional[float] = 0.95
    repetition_penalty: Optional[float] = 1.2
    top_k: Optional[int] = 50
    truncate: Optional[int] = 1024
    watermark: Optional[bool] = False
    max_new_tokens: Optional[int] = 1024
    stop: Optional[list] = ["</s>"]
    return_full_text: Optional[bool] = False
    stream_resp: Optional[bool] = True
    use_cache: Optional[bool] = False
    is_retry: Optional[bool] = False
    retry_count: Optional[int] = 5

    avg_response_time: float = 0.0
    log : Optional[bool] = False
    
    
    @property
    def _llm_type(self) -> str:
        return "Custom LLM Wrapper for LangChain using HuggingChat"
    
    def create_chatbot(self) -> None:
        if not any([self.email, self.psw, self.cookie_path]):
            raise ValueError("email, psw, or cookie_path is required.")
        
        try:
            if not os.path.exists(self.cookie_path):
                # Create a ChatBot using email and psw
                start_time = time.time()
                sign = Login(self.email, self.psw)
                cookies = sign.login()
                end_time = time.time()
                if self.log : print(f"\n[LOG] Login successfull in {round(end_time - start_time)} seconds")

                sign.saveCookiesToDir(self.cookie_path)
            else:
                # Create a ChatBot using cookie_path
                sign_in = Login(self.email, None)
                cookies = sign_in.loadCookiesFromDir(self.cookie_path)
            
            self.chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
            if self.log : print(f"[LOG] LLM WRAPPER created successfully")
            
        except Exception as e:
            raise ValueError("LogIn failed. Please check your credentials or cookie_path. " + str(e))

        # Setup ChatBot info (Commented out because it's not necessary)
        # self.chatbot.switch_llm(self.model)
        # if self.log : print(f"[LOG] LLM WRAPPER switched to model { 'OpenAssistant/oasst-sft-6-llama-30b-xor' if self.model == 0 else 'meta-llama/Llama-2-70b-chat-hf'}")

        self.conversation = self.conversation or self.chatbot.new_conversation()
        self.chatbot.change_conversation(self.conversation)
        if self.log : print(f"[LOG] LLM WRAPPER changed conversation to {self.conversation}\n")
        


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop:
            raise ValueError("stop kwargs are not permitted.")
        
        self.create_chatbot() if not self.chatbot else None
        
        try:
            if self.log : print(f"[LOG] LLM WRAPPER called with prompt: {prompt}")
            start_time = time.time()
            resp = self.chatbot.chat(
                prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                truncate=self.truncate,
                watermark=self.watermark,
                max_new_tokens=self.max_new_tokens,
                stop=self.stop,
                return_full_text=self.return_full_text,
                stream=self.stream_resp,
                use_cache=self.use_cache,
                is_retry=self.is_retry,
                retry_count=self.retry_count,
            )

            end_time = time.time()
            
            self.avg_response_time = (self.avg_response_time + (end_time - start_time)) / 2 if self.avg_response_time else end_time - start_time
            
            if self.log : print(f"[LOG] LLM WRAPPER response time: {round(end_time - start_time)} seconds")
            if self.log : print(f"[LOG] LLM WRAPPER avg response time: {round(self.avg_response_time)} seconds")
            if self.log : print(f"[LOG] LLM WRAPPER response: {resp}\n\n")

            return str(resp)
            
        except Exception as e:
            raise ValueError("ChatBot failed, please check your parameters. " + str(e))

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        parms = { 
            "model": "HuggingChat",
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "top_k": self.top_k,
            "truncate": self.truncate,
            "watermark": self.watermark,
            "max_new_tokens": self.max_new_tokens,
            "stop": self.stop,
            "return_full_text": self.return_full_text,
            "stream": self.stream_resp,
            "use_cache": self.use_cache,
            "is_retry": self.is_retry,
            "retry_count": self.retry_count,
            "avg_response_time": self.avg_response_time,
        }
        return parms
    
    @property
    def _get_avg_response_time(self) -> float:
        """Get the average response time."""
        return self.avg_response_time

if __name__ == "__main__":
    llm = HuggingChat(email=os.getenv("EMAIL"), cookie_path="./cookies_snapshot")
    prompt = PromptTemplate.from_template("Summarize this for me: {text}")

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run("The quick brown fox jumps over the lazy dog.")
    print(output)