import os
#from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
#from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
#from dotenv import load_dotenv
from langchain.llms import GPT4All
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler



MODEL_PATH='C:\coding\docGPT2\YouTube-Document-ChatGPT\models\ggml-mpt-7b-base.bin'

embedding_model_name = "hkunlp/instructor-large"
callbacks = [StreamingStdOutCallbackHandler()]


llm = GPT4All(model=MODEL_PATH, callbacks=callbacks, verbose=True, temp=0.0, top_k=10, top_p=0.75)


def create_conversation() -> ConversationalRetrievalChain:

    persist_directory = 'db'

    # embeddings = OpenAIEmbeddings(
    #     openai_api_key=os.getenv('OPENAI_API_KEY')
    # )
    
    embedding_function = HuggingFaceInstructEmbeddings(
        model_name=embedding_model_name,
        encode_kwargs={"show_progress_bar": True}
    )

    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=False
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(),
        memory=memory,
        get_chat_history=lambda h: h,
        verbose=True
    )

    return qa