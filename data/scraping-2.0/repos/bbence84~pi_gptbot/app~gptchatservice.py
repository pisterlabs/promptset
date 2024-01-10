import os
import openai
from openai import RateLimitError
import logging
import backoff
import time
import re
import tiktoken

from dotenv import load_dotenv
load_dotenv()

from botconfig import BotConfig
bot_config = BotConfig()

class GPTChatService:
    
    def init_logging(self):
        self.log = logging.getLogger("bot_log")
        logging.basicConfig(filename='gpt_service.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    
    def __init__(self, default_language="German"):

        self.init_logging()

        self.default_language = default_language        

        default_lang_prompt = f" Respond in {self.default_language} by default."
        
        self.initial_prompt = [{"role": "user", "content":bot_config.initial_prompt + default_lang_prompt }]

        self.chat_messages = self.initial_prompt
                
        self.log.info(f"Initial ChatGPT prompt:  {self.chat_messages}")

        self.tokenizer_encoding = tiktoken.get_encoding("cl100k_base")

        self.api_type = os.getenv('OPENAI_API_TYPE')
        if self.api_type == 'azure':
            self.client = openai.AzureOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT'),
                api_version = os.getenv('AZURE_OPENAI_VERSION'),                          
            )                    
        else:
            self.client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )           

        # Statistics
        self.total_ai_tokens = 0        
        
    @backoff.on_exception(backoff.expo, RateLimitError, max_time=10, max_tries=2)    
    def ask(self, question):
    
        self.append_text_to_chat_log(question, True)
                   
        start = time.time()

        try:

            if self.api_type == 'azure':
                response = self.client.chat.completions.create( 
                    model=os.getenv('AZURE_OPENAI_DEPLOYMENT'),            
                    messages=self.chat_messages ,
                    max_tokens=bot_config.max_tokens,
                    temperature=bot_config.temperature
                )
            else:
                response = self.client.chat.completions.create(
                    model=bot_config.gpt_model,
                    messages=self.chat_messages ,
                    max_tokens=bot_config.max_tokens,
                    temperature=bot_config.temperature
                )


        except Exception as e:
            print(f"OpenAI API returned an Error", flush=True)
            self.log.error(f"OpenAI API returned an Error")
            if hasattr(e, 'message'):
                print(e.message, flush=True)
                self.log.error(e.message)
            else:
                print(e, flush=True)                         
            return ""    
        

        # print(f'OpenAI API call ended: {time.time() - start} ms')

        response_text = response.choices[0].message.content;
        response_text.isalnum();

        self.update_stats(self.chat_messages, response_text)
        self.check_token_count(self.chat_messages, self.total_ai_tokens)
        
        response_text = self.adjust_response(response_text)
        
        self.append_text_to_chat_log(response_text)
        
        self.log.info(f"ChatGPT response:  {response_text}")

        return response_text    

    def check_token_count(self, chat_messages, total_ai_tokens):
        if total_ai_tokens > bot_config.max_conversation_tokens:
            self.chat_messages = self.initial_prompt
            self.total_ai_tokens = 0
            print(f'Chat buffer cleared, token count reached: {bot_config.max_conversation_tokens}')
            self.log.info(f"Chat buffer cleared, token count reached: {bot_config.max_conversation_tokens}")

    def change_language(self, language):
        self.default_language = language            
    
    def update_stats(self, chat_messages, response):

        start = time.time()
        
        self.total_ai_tokens += self.num_tokens_from_string(response)
        for msg in chat_messages:
            self.total_ai_tokens += self.num_tokens_from_string(msg['content'])
        print(f'Current token count: {self.total_ai_tokens}')

        #print(f'Token count ended: {time.time() - start} ms')

    def get_stats(self):
        return self.total_ai_tokens
 
    def append_text_to_chat_log(self, text, is_user = True):
        role = "assistant"
        if (is_user):
            role = "user"
        self.chat_messages.append({"role": role, "content": text })
        
    def adjust_response(self, response_text):
        adjusted_response_text = response_text
        if response_text.endswith('.'): 
            word_before_last_dot = re.findall(r'\b\d+\b(?=[^\W_]*\.[^\W_]*$)', response_text)
            if word_before_last_dot:
                number_str = word_before_last_dot[0]
                sentence_without_dot = response_text[:-1] if response_text[-2] == ' ' else response_text[:-1] + ' '
                if number_str in sentence_without_dot:
                    adjusted_response_text = sentence_without_dot.replace(number_str + ".", number_str)
        return adjusted_response_text


    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.tokenizer_encoding.encode(string))
        return num_tokens