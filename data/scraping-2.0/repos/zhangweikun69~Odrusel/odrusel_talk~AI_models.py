import configparser
from error_handler import ModelNotSupportError, IllegalInputError
# Langchain
from langchain.schema import (
    get_buffer_string
)

# OpenAI
from langchain.chat_models import ChatOpenAI

# Tiktoken
import tiktoken



class AIModels:
    def __init__(self, model_name:str):
        AImodels_config = configparser.ConfigParser()
        AImodels_config.read('config.ini', encoding='utf-8')
        self.config = ''
        self.model_name = ''
        self.model = None
        self.type = ''

        if model_name == "gpt-3.5-turbo":
            self.type = 'chat'
            self.model_name = str(model_name)
            config = AImodels_config['OPENAI']
            self.api_key = config['gpt_3.5_turbo_openai_api_key']
            self.model = OpenAIModels(0, self.api_key)

        elif model_name == "gpt-4":
            self.type = 'chat'
            self.model_name = str(model_name)
            config = AImodels_config['OPENAI']
            self.api_key = config['gpt_4_openai_api_key']
            self.model = OpenAIModels(1, self.api_key)

        elif model_name == 'cl100k_base':
            self.type = 'tiktoken'
            self.model_name = str(model_name)
            self.model = TiktokenModels(0)

        elif model_name == 'p50k_base':
            self.type = 'tiktoken'
            self.model_name = str(model_name)
            self.model = TiktokenModels(1)

        elif model_name == 'r50k_base':
            self.type = 'tiktoken'
            self.model_name = str(model_name)
            self.model = TiktokenModels(2)

        else:
            # Model not Supported
            raise ModelNotSupportError(f"Model {model_name} is not supported!")


    def use_model(self, messages):
        if self.type == 'chat':
            # The model should have chat method
            response = self.model.chat(messages)
            return response
        elif self.type == 'tiktoken':
            if isinstance(messages,list):
                response = self.model.tik_token_messages(messages)
            elif isinstance(messages,str):
                sentence = messages
                response = self.model.tik_token_sentence(sentence)
            else:
                raise IllegalInputError(f"The input type is {str(type(messages))}, which is illegal!")
        else:
            raise ModelNotSupportError(f"Model {self.model_name} doesn't have this method!")


    def test_api_key(self):

        model_name = self.model_name
        model_api_key = self.api_key
        if model_name == "gpt-3.5-turbo":

            return 1
        elif model_name == "gpt-4":

            return 1
        else:

            return 2
        

class OpenAIModels:
    def __init__(self, index, api_key, max_length = 1000):
        self.openai_models = ['gpt-3.5-turbo', 'gpt-4']
        self.openai_model_name = self.openai_models[index]
        self.openai_api_key = api_key

    def chat(self, messages):
        chat = ChatOpenAI(model_name=self.openai_model_name, temperature=0, openai_api_key=self.openai_api_key, model_kwargs={"stop": ["\n"]})
        return_msg = chat(messages)
        response = return_msg.content
        return response

    def quick_test(self):
        """
            quick_test the validity of models.
        """
        test_text = "Give me just a word"
        result = ''
        chat = ChatOpenAI(model_name=self.openai_model_name, temperature=0, openai_api_key=self.openai_api_key)
        result = chat.predict(test_text)
        if result != '':
            return True
        else:
            return False

class TiktokenModels:
    def __init__(self, index):
        self.tiktoken_models = ['cl100k_base', 'p50k_base', 'r50k_base']
        self.tiktoken_model_name = self.tiktoken_models[index]
        self.encoding = tiktoken.get_encoding(self.tiktoken_model_name)

    def tik_token_messages(self, messages):
        num_of_tokens = len(self.encoding.encode(get_buffer_string(messages)))
        return num_of_tokens

    def tik_token_sentence(self, sentence):
        num_of_tokens = len(self.encoding.encode(str(sentence)))
        return num_of_tokens


"""
#test OpenAIModels
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string
)
llm = OpenAIModels(0, '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')
llm.quick_test()
messages2 = [
        SystemMessage(content="You are an assistant."),
        HumanMessage(content="Hello, my name is Larry Zhang."),
        AIMessage(content="Hi Larry Zhang, How can I help you."),
        HumanMessage(content="Who am I?")
    ]
llm.chat(messages2)


#test AIModels
llm2 = AIModels("gpt-3.5-turbo")
messages2 = [
        SystemMessage(content="You are an assistant."),
        HumanMessage(content="Hello, my name is Larry Zhang."),
        AIMessage(content="Hi Larry Zhang, How can I help you."),
        HumanMessage(content="Who am I?")
    ]
llm2.use_model_chat(messages2)

#test Tiktoken
t0 = TiktokenModels(0)
t1 = TiktokenModels(1)
t2 = TiktokenModels(2)
t0.tik_token_messages(messages2)
t1.tik_token_messages(messages2)
t2.tik_token_messages(messages2)
"""