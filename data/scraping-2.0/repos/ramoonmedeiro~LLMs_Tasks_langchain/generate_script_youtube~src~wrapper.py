from prompt import Prompt
from settings import Config
from langchain.llms import OpenAI
from chains import CreateChain
from langchain.utilities import WikipediaAPIWrapper


class ModelWrapper():

    def __init__(
        self,
        openai_api_key: str,
        temperature: float,
        topic: str
    ) -> None:

        self.prompt = Prompt()
        self.openai_api_key = openai_api_key
        self.temperature = temperature
        self.topic = topic
        self.content_wiki = WikipediaAPIWrapper().run(self.topic)
        self.llm = OpenAI(
            openai_api_key=self.openai_api_key,
            temperature=self.temperature
        )
        self.chain = CreateChain()

    def wrapper(self):

        prompt1 = self.prompt.create_prompt(
            template=Config.TEMPLATE_INITIAL.value,
            input_variables=["topic"]
        )

        prompt2 = self.prompt.create_prompt(
            template=Config.TEMPLATE_FINAL.value,
            input_variables=["title", "content_wiki"]
        )

        chain1 = self.chain.create(
            prompt=prompt1,
            llm=self.llm,
            output_key="title"
        )

        chain2 = self.chain.create(
            prompt=prompt2,
            llm=self.llm,
            output_key="script"
        )

        full_chain = self.chain.concatenate_chains(chain1, chain2)

        return full_chain.run({
            "topic": self.topic,
            "content_wiki": self.content_wiki
        })
