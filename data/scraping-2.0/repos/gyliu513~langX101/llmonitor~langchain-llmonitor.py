from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.callbacks import LLMonitorCallbackHandler

handler = LLMonitorCallbackHandler()

chat = ChatOpenAI(
  callbacks=[handler],
)

chat.predict("Hello, how are you?")
