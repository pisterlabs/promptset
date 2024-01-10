from langchain.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
import db

class QueryLLM():

    db = None

    def __init__(self):
        self.db = db.DB()

    def ask(self, query, history, filter):
        llm = OpenAI(
			verbose=True,
			temperature=0
			)
        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
        doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_PROMPT)

        qa = ConversationalRetrievalChain(
    	    retriever=self.db.vectorstore.as_retriever(),
		    combine_docs_chain=doc_chain,
		    question_generator=question_generator,
		    return_source_documents=True
		)

        result = qa({"question": query, "chat_history": history})
        result['history'] = [(query, result["answer"])]
        sources = []
        # TODO filter
        for doc in result["source_documents"]:
            sources.append(doc.metadata["source"])
        sources = list(set(sources))
        response = {"answer":result['answer'], "history":result['history'], "sources":sources}
        return response