from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from langchain.chains import LLMChain

from LangChainPipeline.PromptTemplates.parser_prompt import get_parser_prompt_chat as get_parser_prompt
from LangChainPipeline.PydanticClasses.References import References

#langchain.llm_cache = InMemoryCache()

class IntentParserChain():

    def __init__(
        self,
        verbose,
        temperature=0.1,
        model_name="gpt-4-1106-preview",
    ):
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)
        self.parser = PydanticOutputParser(pydantic_object=References)

        self.prompt_template = get_parser_prompt({
            "format_instructions": self.parser.get_format_instructions(),
        })    

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=verbose,
            output_parser=self.parser,
            # memory=ConversationBufferMemory()
        )

        # self.chain = create_structured_output_chain(
        #     References,
        #     llm=self.llm,
        #     prompt=self.prompt_template,
        #     verbose=True,
        #     # output_parser=self.parser,
        #     # memory=ConversationBufferMemory()
        # )
        print("Initialized IntentParserChain")

    def run(self, command):
        if command == "":
            return References()
        try:
            references = self.chain.predict(command=command)
        except:
            print("ERROR: Failed to parse: ", command)
            return References()
        return references