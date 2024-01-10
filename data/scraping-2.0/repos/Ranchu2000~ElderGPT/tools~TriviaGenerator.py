from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API= os.getenv('OPENAI_API_KEY')

def CognitiveGames() -> str:
    """Useful to play cognitive games. If the user asks for it, give them a question. if the answer is wrong, give them a hint until they get the right answer or say they give up
    :param: None
    :returns: a trivia question and answer in json format
    """
    chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API)
    messages = [
        SystemMessage(
            content="you are a helpful assistant."
        ),
        HumanMessage(
            content="give me a trivia question that would be good for elderly. Return the question and answer in json format."
        ),
    ]
    return chat(messages).content