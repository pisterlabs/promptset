
from .benchmark_keywords_config import BenchmarkKeywordsConfig
from openai_model.chatgpt import run_conversation
from interface.service_interface import ServiceInterface

class BenchmarkKeywordsService(ServiceInterface):
    def __init__(self):
        self.__benchmark_config = BenchmarkKeywordsConfig()

    def execute(self, opt="1"):          
        i = 0;
        question = self.__benchmark_config.get_benchmark().get_question(i)    
        while question != None:
            #print(question)
            run_conversation(question, self.__benchmark_config.get_config_function(), opt, show_result=False)
            i += 1
            question = self.__benchmark_config.get_benchmark().get_question(i)   
        self.__benchmark_config.get_benchmark().save_result()
