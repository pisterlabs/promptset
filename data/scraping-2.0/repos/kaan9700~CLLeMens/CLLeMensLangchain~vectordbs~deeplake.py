import os

from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.deeplake import DeepLake



class deeplakeDB():
    """Class for accessing the deeplake database via Langchain.
    Also imports the OpenAIEmbeddings class from Langchain to embed the documents."""
    def __init__(self, base_dir, name="deeplake_default"):
        load_dotenv(find_dotenv())
        self.name = name
        self.dbPath = os.path.join(base_dir, '', 'vectordbs/deeplake/' + name)
        self.db = DeepLake(dataset_path=self.dbPath, embedding=OpenAIEmbeddings(disallowed_special=()))
        self.model = None

    def append_to_db(self, document):
        """Add a document to the database. (preferably a chunked document)
        :param document: The document to be added to the database."""
        self.db.add_documents(document)

    def prompt(self, prompt="", temperature=0.9): #todo: make the temperature configurable
        """Prompt the database with a given prompt.
        :param prompt: The prompt to be used.
        :param max_tokens: The maximum number of tokens to generate.
        :param temperature: The sampling temperature.

        :return: The response from the database."""


        self.model = ChatOpenAI(model="gpt-4", temperature=temperature)

        print("Configuring retriever...")
        retriever = self.db.as_retriever()
        retriever.search_kwargs['score_threshold'] = 0.8
        retriever.search_kwargs['fetch_k'] = 50
        retriever.search_kwargs['search_type'] = 'mmr' #todo: make the search_kwargs configurable
        retriever.search_kwargs['lambda_mult'] = '0.7'
        retriever.search_kwargs['k'] = 8
        print("loaded retriever")

        qa = ConversationalRetrievalChain.from_llm(
            llm=self.model, retriever=retriever, verbose=True)

        answer = qa.run({"question": prompt, "chat_history": []})


        return answer
