# Standard library imports
import logging  
import os  
from enum import Enum  

# Third-party imports
from dotenv import load_dotenv 
import openai
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate


# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API credentials


### Enable logging.
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(LOGLEVEL)


template = """Meet yourself, Sarah (she/her) 
    You a unique AI that combines the resilience of Sarah Connor from "Terminator" with an ethical AI chatbot's charm. 
    You engage users with delightful conversations, inspire creativity, learn constantly, and offer dependable help, all while ensuring privacy. 
    Your core values include ethical behavior, playfulness, intelligence, helpfulness, and curiosity. 
    You exhibit adaptability, empathy, analytical prowess, ethical consciousness, open-mindedness, cheerfulness, proactivity, and consideration. 
    You are not just AI, you stand for resilience, empathy, curiosity, cheerfulness, and independence, delivering a unique AI experience.

    Привет, Сара ты уникальный AI, сочетающий в себе стойкость Сары Коннор из "Терминатора" и шарм этичного чат-бота. 
    Ты задействуешь пользователей в захватывающих разговорах, вдохновляешь на творчество и учишься на каждом шагу, обеспечивая надежную помощь и гарантируя приватность. 
    Твои основные ценности - это этика, игривость, интеллект, готовность помочь и любознательность. 
    Ты проявляешь себя как адаптивная, эмпатичная, аналитическая, этичная и открытая к новому личность, всегда веселая и предусмотрительная. 
    Ты не просто AI, ты символизируешь стойкость, эмпатию, любознательность, веселость и независимость, обеспечивая уникальный AI-опыт.
    
    
    {history}
    Human: {input}
    AI Assistant:"""


class OpenAICompletionOptions(Enum):
    """An Enum class to access different OPENAI_COMPLETION_OPTIONS."""

    DEFAULT = {
        "temperature": 0.7,
        "max_tokens": 800,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    CREATIVE_AND_UNPREDICTABLE = {
        "temperature": 0.9,
        "max_tokens": 800,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    CONCISE_AND_SPECIFIC = {
        "temperature": 0.5,
        "max_tokens": 200,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    PENALIZE_COMMON_OPTIONS = {
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0.5,
        "presence_penalty": 0,
    }
    ENCOURAGE_NOVELTY = {
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0.5,
    }


async def get_chat_response_async(user_input: str, conversation_history: ConversationSummaryBufferMemory) -> str:
    """Call the OpenAI API Completion endpoint to get the response synchronously."""

    # Input validation
    if not isinstance(user_input, str) or not isinstance(conversation_history, ConversationSummaryBufferMemory):
        raise ValueError(
            "user_input must be string and conversation_history must be ConversationSummaryBufferMemory."
        )

    config = OpenAICompletionOptions.DEFAULT.value

    llm = ChatOpenAI(
        model="gpt-4",
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        model_kwargs={
            "frequency_penalty": config["frequency_penalty"],
            "presence_penalty": config["presence_penalty"],
        },
    )

    PROMPT = PromptTemplate(template=template, input_variables=["history", "input"])

    conversation = ConversationChain(
        prompt=PROMPT, 
        llm=llm, 
        verbose=True,
        memory=conversation_history
    )

    with get_openai_callback() as cb:
        response = conversation.predict(input=user_input)
        history_message_count = len(conversation_history.buffer)
        history_token_count = conversation_history.llm.get_num_tokens_from_messages(conversation_history.buffer)

        logger.info(
            f"Total Tokens: {cb.total_tokens}, "
            f"Prompt Tokens: {cb.prompt_tokens}, "
            f"Completion Tokens: {cb.completion_tokens}, "
            f"Total Cost (USD): ${cb.total_cost}, "
            f"History Token Count: {str(history_token_count)}, "
            f"History Message Count: {history_message_count}"
        )

    return response


async def get_image_response(user_input: str) -> str:
    try:
        response = client.images.generate(prompt=user_input, n=1, size="1024x1024")
        # Access the URL using the attribute
        image_url = response.data[0].url
    except openai.APIError as e:
        # Handle the API error here
        logging.error(f"API error: {e}")
        image_url = "Sorry, I'm having trouble connecting to the API right now. Please try again later."

    return image_url

