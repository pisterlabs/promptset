from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from Modules.prompt_template import system_prompt, conversation_prompt

def run_conversationalretrieval_chain(vectorstore, llm, memory):

    question_generator = LLMChain(llm=llm, prompt=system_prompt, memory=memory)
    doc_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=conversation_prompt)
    conv_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator
    )
    return conv_chain