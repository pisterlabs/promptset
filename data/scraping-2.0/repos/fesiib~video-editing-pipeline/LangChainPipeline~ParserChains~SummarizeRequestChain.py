from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from langchain.chains import LLMChain

from LangChainPipeline.PromptTemplates.summarize_request_prompt import get_summarize_request_prompt_chat as get_summarize_request_prompt

class SummarizeRequestChain():

    def __init__(
        self,
        verbose,
        temperature=0.7,
        model_name="gpt-3.5-turbo-16k-0613",
    ):
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)

        self.prompt_template = get_summarize_request_prompt()    

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=verbose,
        )
        print("Initialized SummarizeRequestChain")

    def run(self, request):
        # #dummy
        # return "Some long summary"

        if request == "":
            return "No Request Given"
        try:
            response = self.chain.predict(
                request=request,
            )
        except:
            print("ERROR: Failed to summarize: ", request)
            return ""
        return response