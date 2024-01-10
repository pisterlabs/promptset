# RetrievalQAChain: chains together retriever and gpt 4 to generate responses to questions

from langchain.chains import RetrievalQA

def create_retrievalqa_chain(llm, chain_type, retriever):
    qa_chain = RetrievalQA.from_chain_type(llm = llm, chain_type = chain_type,
                                           retriever = retriever)
    return qa_chain

# Pass this description of the qa chain to the conversational agent. 
# It will be able to use the qa chain as a tool to answer questions.
retrievalqa_desc = """Use this tool to answer questions that are relevant to local information
                    and sources provided by the user."""