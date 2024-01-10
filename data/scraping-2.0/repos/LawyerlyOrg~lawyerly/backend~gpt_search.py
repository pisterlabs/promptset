import openai
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI

def chat_with_gpt(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def chat_with_index(query, index, file_name):
    meta_filter = {'source':file_name}
    #index = getExistingIndex(index_name, embeddings, name_space)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=OpenAI(temperature=0.0), chain_type="stuff", retriever=index.as_retriever(search_kwargs={'filter': meta_filter}))
    result = chain({'question': query}, return_only_outputs=True)
    return result

def get_existing_index(index_name, embeddings, collection_name=""):
    if not collection_name:
        pindex = Pinecone.from_existing_index(
        index_name=index_name, embedding=embeddings)
    else:
        pindex = Pinecone.from_existing_index(
            index_name=index_name, embedding=embeddings, namespace=collection_name)
    return pindex