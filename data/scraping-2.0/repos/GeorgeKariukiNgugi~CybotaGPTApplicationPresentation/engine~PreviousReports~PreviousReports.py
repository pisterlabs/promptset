import os

from langchain.document_loaders import CSVLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

os.environ["OPENAI_API_KEY"]

index_creator = VectorstoreIndexCreator()

malwareEventsLoader = UnstructuredExcelLoader(
    "/var/www/html/serianu_projects/projects/LangChain_StreamLit/Langchain/Data/malwareEvents.xlsx")
malwareEventsDocsearch = index_creator.from_loaders([malwareEventsLoader])
malwareEventsChain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff",
                                                 retriever=malwareEventsDocsearch.vectorstore.as_retriever(),
                                                 input_key="question")

print("Finished INDEX malwareEventsChain")

userLogonsLoader = UnstructuredExcelLoader(
    "/var/www/html/serianu_projects/projects/LangChain_StreamLit/Langchain/Data/userLogons.xlsx")
userLogonsDocsearch = index_creator.from_loaders([userLogonsLoader])
userLogonsChain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff",
                                              retriever=userLogonsDocsearch.vectorstore.as_retriever(),
                                              input_key="question")

print("Finished INDEX userLogonsChain")


def PreviousReports():
    print("PreviousReports")


def callchain(chain, query):
    if chain == 'malwareEvents':
        response = malwareEventsChain({"question": query})
        print(response)
        return response

    else:
        response = userLogonsChain({"question": query})
        print(response)
        return response


callchain('malwareEvents', "Your query here")
