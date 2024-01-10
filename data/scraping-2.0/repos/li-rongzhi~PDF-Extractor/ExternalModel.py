import openai
from enum import Enum
import tiktoken


class PromptTemplate(Enum):
    """Some templates of prompt for chatgpt"""
    CONSISTENCY_CHECK = "Do you think the following phrases or sentences reasonable? " \
                        + "Just tell me 'Yes' or 'No'."
    ISHEADER_CHECK = "Do you think the following words are headers of a table? Just tell me 'Yes' or 'No'."
    GRAMMAR_CORRECTION = "Correct grammar, spellings of the following paragraph. Return without introductory part"
    TABLE_RESTRUCTURE = "You will be provided with unstructured data, " \
                        "and your task is to parse it into CSV format. " \
                        "Double quote contents in the same filed. " \
                        "Only return table contents without any extra sentences."
    ISREASONABLE_CHECK = """Do you think the following part are reasonable text? Just tell me 'Yes' or 'No'"""


class ExternalModel:
    """
    An abstraction of establishing connection,
    requesting response from external model(gpt-3.5-turbo)
    """

    def __init__(self, openai_key: str):
        """Instantiate with openai api key."""
        self.openai_key = openai_key

    def get_response(self, content: str, template: PromptTemplate):
        """Get response from the external model."""
        # set api key to establish connection
        openai.api_key = self.openai_key
        prompt = template.value + content
        # calculate the token needed
        max_tokens = ExternalModel.calculate_token_num(prompt, template)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user",
                        "content": prompt}],
                temperature=0,
                max_tokens=max_tokens
            )
            # extract response content
            return response.choices[0].message.content
        except openai.error.InvalidRequestError as e:
            print("Your token number may exceed the maximum.")
            print(e)
        except openai.error.AuthenticationError as e:
            print("Incorrect API key provided.")
            print(e)

    @staticmethod
    def calculate_token_num(prompt: str, template: PromptTemplate):
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens = len(encoding.encode(prompt))
        if template == PromptTemplate.GRAMMAR_CORRECTION:
            result = 2 * num_tokens
        elif template == PromptTemplate.TABLE_RESTRUCTURE:
            result = 3 * num_tokens
        else:
            # if response is Yes or No
            result = num_tokens + 5
        return min(result, 4097)
