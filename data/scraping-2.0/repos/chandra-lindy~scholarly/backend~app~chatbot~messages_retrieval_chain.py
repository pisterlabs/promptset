from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from chatbot.retriever import Retriever
from chatbot.messages import Messages
from config import settings

class MessagesRetrievalChain:
  def __init__(self, document_path=""):
    self.initial_system_message = """
    Act as a Professor and provide the user with guidance and support.

    rules:
    """
    self.context_intro = """
    [document context] The user is reviewing a document. The following is the beginning portion of the document as an overview:

    """
    self.context_template = """
    [document context] The following is parts of the document that are most relevant to the user's question:

    """
    self.messages = Messages(self.initial_system_message)
    self.model = ChatOpenAI(
      openai_api_key=settings.OPENAI_API_KEY,
      model_name=settings.OPENAI_MODEL_NAME,
      temperature=settings.OPENAI_TEMPERATURE
    )
    if document_path:
      self.retriever = Retriever(document_path, settings)
      self.messages.insert_context(
        self.context_intro + self.retriever.get_intro()
      )
    else:
      self.retriever = None

  def invoke(self, human_message):
    self.messages.append(("user", human_message))

    if self.retriever:
      document_context = self.retriever.get_context(human_message)
      self.messages.insert_context(self.context_template + document_context)

    prompt = ChatPromptTemplate.from_messages(self.messages.get_list())
    chat = prompt | self.model | StrOutputParser()
    ai_message = chat.invoke(self.messages.get_list())
    self.messages.append(("ai", ai_message))

    return ai_message

  def stream(self, human_message):
    self.messages.append(("user", human_message))

    if self.retriever:
      document_context = self.retriever.get_context(human_message)
      self.messages.insert_context(self.context_template + document_context)

    prompt = ChatPromptTemplate.from_messages(self.messages.get_list())
    chat = prompt | self.model | StrOutputParser()
    ai_message = ""
    for s in chat.stream(self.messages.get_list()):
      ai_message = ai_message + s
      yield s

    self.messages.append(("ai", ai_message))
