from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import AzureChatOpenAI
content = PyPDFLoader("jjad179.pdf").load()

AzureChatOpenAI(mo)