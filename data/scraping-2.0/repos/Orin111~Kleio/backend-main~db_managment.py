from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
import prompts

basic_information = "John Doe, a person with dementia"

questions_answers = []


def query(vector_store, query, llm):
    chain = load_qa_chain(llm, chain_type="stuff")
    similar_vectors = vector_store.similarity_search(query)
    beauty_prompt = prompts.create_prompt(query, basic_information)
    answer = chain.run(input_documents=similar_vectors, question=beauty_prompt)
    questions_answers.append((query, answer))
    return answer


def insert(text, embeddings, index_name, text_key):
    Pinecone.from_texts([text], embeddings, index_name=index_name, text_key=text_key)

