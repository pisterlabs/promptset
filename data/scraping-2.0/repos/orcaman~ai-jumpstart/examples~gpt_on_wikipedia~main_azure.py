import dotenv

dotenv.load_dotenv()

import sys
from langchain.text_splitter import CharacterTextSplitter
from langchain.utilities import WikipediaAPIWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import AzureOpenAI
from langchain.chains import RetrievalQA


class WikipediaAnalyzer:
    def __init__(self, search_term, query):
        self.search_term = search_term
        self.query = query
        self.documents = None
        self.texts = None
        self.chain = None
        self.api_client = WikipediaAPIWrapper()

    def load_wikipedia_page(self):
        if ',' in self.search_term:
            search_terms = self.search_term.split(',')
            self.documents = []
            for term in search_terms:
                self.documents.extend(self.api_client.load(term.strip()))
        else:
            self.documents = self.api_client.load(self.search_term.strip())

    def split_documents_into_chunks(self, chunk_size=800, chunk_overlap=0):
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.texts = text_splitter.split_documents(self.documents)

    def prepare_model_embedding(self):
        embeddings = OpenAIEmbeddings(
            chunk_size=1,
            deployment='text-embedding-ada-002',
        )

        doc_search = Chroma.from_documents(self.texts, embeddings)

        llm = AzureOpenAI(deployment_name='text-davinci-003', model_name="text-davinci-003")

        self.chain = RetrievalQA.from_chain_type(llm=llm, retriever=doc_search.as_retriever())

    def analyze_wikipedia_page(self):
        self.load_wikipedia_page()
        self.split_documents_into_chunks()
        self.prepare_model_embedding()

        print('\n\n\n\n\n-----------------')
        print('wikipedia search terms:', self.search_term)
        print('question:', self.query)
        print('answer:', self.chain.run(self.query))
        print('-----------------\n\n')


def main():
    if len(sys.argv) < 3:
        print("Missing arguments: wikipedia search term and/or query")
        sys.exit(1)

    search_term = sys.argv[1]
    query = sys.argv[2]

    analyzer = WikipediaAnalyzer(search_term, query)
    analyzer.analyze_wikipedia_page()


if __name__ == "__main__":
    main()
