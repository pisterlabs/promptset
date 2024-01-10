from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.prompts import Prompt

from langchain.chat_models import ChatOpenAI
from llama_index import ServiceContext

class PdfWord:
    def __init__(self):
        self._folder_name = ""
        self._prompt = ""

    def analyze(self, temp_dir, user_prompt_text):
        self._folder_name = temp_dir + '/'
        self._prompt = user_prompt_text

        documents = SimpleDirectoryReader(self._folder_name).load_data()
        index = VectorStoreIndex.from_documents(documents)
        # query_engine = index.as_query_engine()
        chat_engine = index.as_chat_engine(chat_mode='react', verbose=True)
        response = chat_engine.chat('Use the tool to answer: ' + self._prompt)
        # response = query_engine.query(
        #     "You're an intelligent subject expert and humorist, Respond the following query the data "
        #     + self._prompt)
        return str(response)

# if __name__ == "__main__":
#     pw = PdfWord()
#     documents = SimpleDirectoryReader('tmp/').load_data()
#     index = VectorStoreIndex.from_documents(documents)
#     chat_engine = index.as_chat_engine(verbose=True)
#     pw.analyze(temp_dir='tmp', user_prompt_text=chat_engine.chat_repl())
