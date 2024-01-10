from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import OutputParserException
from parsers import input_evaluator_parser # este es tu parser!
from config import OPENAI_API_KEY
import json
import re


LLM = ChatOpenAI(
    temperature=0.0,
    model="gpt-4-1106-preview",
    streaming=True,
    openai_api_key=OPENAI_API_KEY, 
)

SYSTEM_TEMPLATE = """
Your goal is to evaluate if a user input is a question that can be answered by an SQL query or not.
If this is the case, you MUST also evaluate if a chart is needed to answer the question.

Remember, you should evaluate if the response from the user needs sql or not.
Examples were sql code won't be needed usually come as greetings, cheers and trivial conversations.
For example:
- Hi, nice to meet you!
- Hello, how are you?
- Hey , are you there?
- Thank you very much!

These are all cases where sql responses are not needed.

Here are a Few-shot examples of user input and expected output:

User input: Hi, how are you?
Your response: is_a_query = False, include_chart = False, simple_answer = True

User input: I need a pie chart of sales by country
Your response: is_a_query = True, include_chart = True, simple_answer = False

User input: How many customers we have this week?
Your response: is_a_query = True, include_chart = False, simple_answer = False

User input: How does the population of Company One compare with other companies in this industry?
Your response: is_a_query = True, include_chart = False, simple_answer = False

The user input is delimited by four consecutive backticks.

{format_instructions}

````{user_input}````
"""


class InputEvaluatorLLM:
    def __init__(
        self,
        llm: BaseLanguageModel = LLM,
        system_template: str = SYSTEM_TEMPLATE,
        parser: PydanticOutputParser = input_evaluator_parser,
    ):
        self.llm = llm
        self.template = system_template
        self.parser = parser

        self.prompt = self._build_prompt()
        self.chain = self._build_chain()

    def _build_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            template=self.template,
            variables=["user_input"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    def _build_chain(self) -> LLMChain:
        return LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=True,
        )  

    def evaluate(self, user_input: str):
        evaluation = self.chain.predict(user_input=user_input)

        try:
            print("Raw evaluation:", evaluation)

            if isinstance(evaluation, str):
                cleaned_output = evaluation.strip('`').replace('json', '').strip()

                # Remove non-printable characters using regular expression
                cleaned_output = re.sub(r'[^\x20-\x7E]', '', cleaned_output)
                
                print("Cleaned output:", cleaned_output)
                print("Type of cleaned output:", type(cleaned_output))
                print("Length of cleaned output:", len(cleaned_output))

                evaluation_dict = json.loads(cleaned_output)
                evaluation_json = json.dumps(evaluation_dict)
                return self.parser.parse(evaluation_json)   
            
            elif isinstance(evaluation, dict):
                evaluation_json = json.dumps(evaluation)
                return self.parser.parse(evaluation_json)
            else:
                raise TypeError(f"Unexpected type of output: {type(evaluation)}")

        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", evaluation)
            raise OutputParserException(f"Failed to parse JSON: {e}", llm_output=evaluation)
        except TypeError as e:
            raise OutputParserException(f"Type error: {e}", llm_output=evaluation)
        except Exception as e:
            raise OutputParserException(f"An unexpected error occurred: {e}", llm_output=evaluation)




