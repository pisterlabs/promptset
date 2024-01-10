"""Function to check if a response is a valid response"""

from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain import PromptTemplate, LLMChain

from dotenv import load_dotenv

load_dotenv() 

# Define your desired data structure.
class Answer(BaseModel):
    venmo_handle: str = Field(description="Venmo handle in the tweet. Example: @Vishakh-Hegde. All Venmo handles start with a @.")
    payment_note: str = Field(description="Payment note in the tweet. Example: You bought some groceries for my neighbor. Here is $5. If the person did not mention a kind thing they did, this is an empty string.")


PROMPT_TEMPLATE = """
You are a world class text processor. Your job is to process text in Tweet and respond to it.
Closely follow the format instructions. This is extremely important.
You want to understand if the tweet mentions if they did a kind or nice thing.
You should also check if the tweet mentions a venmo handle.

{format_instructions}

Tweet: {tweet}
Answer: 
"""

TEST_TWEETS = [
    "Hey, I did something nice. Send to @Vishakh-Hegde",
    "I bought some groceries for my neighbor. Vishakh Hegde",
    "I bought some groceries for my neighbor. @Vishakh-Hegde",
]


def is_valid_response(tweet: str) -> bool:
    """Function to check if a response is a valid response.

    Args:
        tweet: <str> The response for the tweet to verify.

    Output:
        {
            'venmo_handle': <str>
            'payment_note': <str>
            'is_valid_response': <bool>
        }    
    """
    # Create an instance of the PydanticOutputParser.
    parser = PydanticOutputParser(pydantic_object=Answer)

    # Create a prompt template
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["tweet"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Get the text to input into the LLM.
    _input = prompt.format_prompt(tweet=tweet)

    chat_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    output = chat_model.predict(_input.to_string())

    print(f"output: {output}")

    
    try:
        parsed_output = parser.parse(output)
    except Exception as e:
        parsed_output = {'venmo_handle': '', 'payment_note':''}
    
    parsed_output_dict = parsed_output.__dict__

    if parsed_output_dict['venmo_handle'] and parsed_output_dict['payment_note']:
        parsed_output_dict['is_valid_response'] = True
    else:
        parsed_output_dict['is_valid_response'] = False

    return parsed_output_dict

if __name__ == "__main__":
    for tweet in TEST_TWEETS:
        response = is_valid_response(tweet)

        print(response)

