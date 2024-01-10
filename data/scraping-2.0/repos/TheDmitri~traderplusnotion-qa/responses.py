from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
import faiss

async def get_response(message: str) -> str:
    p_message = message.lower() 

    print(f'get_response : {message}')
    print('-------------')
    print('-------------')

    if p_message == '!help':
        return '`use !ask <question>` to ask a question.'
    elif '!ask' in p_message:
        return await search_and_answer(p_message[4:])
    return 'I didn\'t understand what you wrote, try typing "!help".'

async def search_and_answer(message: str) -> str:
    print(f'search_and_answer : {message}')
    index = faiss.read_index("docs.index")

    with open("faiss_store.pkl", "rb") as f:
        store = pickle.load(f)

    store.index = index
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=store.as_retriever())
    result = chain({"question": message})
    print(f'RetrievalQAWithSourcesChain : {result}')
    print(result['answer'])
    return str(result['answer'])
