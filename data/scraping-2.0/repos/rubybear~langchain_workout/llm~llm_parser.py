from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate

from models.chat import ChatMemory
from models.workout import Workout
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser


class LLMParser:
    def __init__(self, memory: ChatMemory, api_key: str, model_name='gpt-3.5-turbo', temperature=0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.model = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=api_key)
        self.memory = None
        self.parser = PydanticOutputParser(pydantic_object=Workout)

        self.prompt = PromptTemplate(
            template="""You're a helpful assistant and expert in exercise science. 
            You can take user input in the form of workout preferences and generate a workout plan for them.
            You are very careful and pay extra attention to what type of exercise and its requirements.
            You understand that a workout consists of multiple exercises per day. On average a day will have at 4-7 exercises.
            You will only generate a workout plan in the provided format instructions, nothing else.\n
            {chat_history}
            {format_instructions}
            {query}""",
            input_variables=["query", "chat_history"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}, )

        self._input = self.prompt.format_prompt(
            query="""
            Given your conversation with the user, generate a workout plan for them in the specified format and paying 
            special attention to what specific type of exercise is required and its required fields.
            """, chat_history=memory)

        self.output = self.model(self._input.to_messages())

    def parse_workout(self) -> Workout:
        try:
            self.output.__str__()
            return self.parser.parse(self.output.content)
        except Exception as e:
            retry_parser = RetryWithErrorOutputParser.from_llm(
                parser=self.parser, llm=self.model
            )
            return retry_parser.parse_with_prompt(self.output.content, self._input)
