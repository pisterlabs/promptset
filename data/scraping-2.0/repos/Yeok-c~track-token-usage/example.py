from src.track_usage import TokenTrack
from langchain.callbacks import get_openai_callback

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class YourClass(TokenTrack):
    def __init__(self):
        super(YourClass, self).__init__()

    def your_function(self):
        with get_openai_callback() as cb:            
            # any langchain function to be calculated
            results = LLMChain(
                llm=ChatOpenAI(), 
                prompt=ChatPromptTemplate.from_template("What's the capitol of {country}? List 20 top attractions there"),
                verbose=True
            ).run("France")
            print(results)

        self.update_usage(cb)

    def whenever_you_want_to_check(self):
        self.print_usage()
        
if __name__ == "__main__":
    yourclass = YourClass()
    yourclass.your_function()