import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import create_qa_with_sources_chain
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Qdrant
class VectaraAgent:
    def __init__(self):
        self.customer_id = '1464859115'

    def get_qa_agent(self, corpus_id: str):
        '''
        Will get qa agent
        :param corpus_id:
        :return:
        '''
        llm_src = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        corpus_api_key = os.environ['CORPUS_API_KEY']
        qdrant_client = None
        qdrant = Qdrant(
            client=qdrant_client,
            collection_name="docs"
        )


        doc_prompt = PromptTemplate(
            template="Content: {page_content}\nSource: {source} \n",  # look at the prompt does have page#
            input_variables=["page_content", "source"],
        )

        qa_chain = create_qa_with_sources_chain(llm_src)
        qa_chain = StuffDocumentsChain(
            llm_chain=qa_chain,
            document_variable_name='context',
            document_prompt=doc_prompt,
        )
        chat = RetrievalQA(
            retriever=qdrant.as_retriever(),
            combine_documents_chain=qa_chain,

        )

        return chat