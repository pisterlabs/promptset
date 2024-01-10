from langchain.chains.question_answering import load_qa_chain  # To work with openAI or any other language models
from langchain.llms import OpenAI  # Wrapper of OpenAI language model

def answer_question(knowledge_base, user_question, chain_type="stuff", model_name="gpt-3.5-turbo"):
    """
    Function to generate the answer to a user question.

    Parameters:
    knowledge_base (object): FAISS object with the embeddings.
    user_question (str): User's question.
    chain_type (str): Type of chain to be used for question answering.

    Returns:
    response (str): Generated answer.
    """
    docs = knowledge_base.similarity_search(user_question)
    llm = OpenAI(model_name=model_name) # Specify the model name
    chain = load_qa_chain(llm, chain_type=chain_type)
    response = chain.run(input_documents=docs, question=user_question)
    return response
