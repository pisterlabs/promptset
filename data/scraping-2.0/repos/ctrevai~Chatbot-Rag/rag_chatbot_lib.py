from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
# from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage
from langchain.memory import ConversationBufferMemory


def get_llm():

    model_kwargs = {  # anthropic
        "max_tokens_to_sample": 512,
        "temperature": 0,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman:"]
    }

    llm = Bedrock(
        credentials_profile_name="default",
        region_name="us-east-1",
        model_id="anthropic.claude-v2:1",  # set the foundation model
        model_kwargs=model_kwargs,)  # configure the properties for Claude

    return llm


pdf_path = "2022-Shareholder-Letter.pdf"


def select_pdf():
    pdf_path = "uploaded_file.pdf"
    return


def get_index():

    # create embeddings for the index
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default",
        region_name="us-east-1",
    )  # Titan Embedding by default

    loader = PyPDFLoader(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=100,
    )

    # create the index
    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        text_splitter=text_splitter,
    )

    index_from_loader = index_creator.from_loaders([loader])

    return index_from_loader


def get_memory():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    return memory


def get_rag_chat_response(input_text, memory, index):  # chat client function

    llm = get_llm()

    _ROLE_MAP = {"human": "\n\nHuman: ", "ai": "\n\nAssistant: "}

    def _get_chat_history(chat_history):
        buffer = ""
        for dialogue_turn in chat_history:
            if isinstance(dialogue_turn, BaseMessage):
                role_prefix = _ROLE_MAP.get(
                    dialogue_turn.type, f"{dialogue_turn.type}: ")
                buffer += f"\n{role_prefix}{dialogue_turn.content}"
            elif isinstance(dialogue_turn, tuple):
                human = "\n\nHuman: " + dialogue_turn[0]
                ai = "\n\nAssistant: " + dialogue_turn[1]
                buffer += "\n" + "\n".join([human, ai])
            else:
                raise ValueError(
                    f"Unsupported chat history format: {type(dialogue_turn)}."
                    f" Full chat history: {chat_history} "
                )
        return buffer

    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=index.vectorstore.as_retriever(),
        memory=memory,
        get_chat_history=_get_chat_history,
        # condense_question_prompt=condense_prompt_claude,
        verbose=True,
        chain_type='stuff'
    )

    # pass the user message, history, and knowledge to the model
    chat_response = conversation_with_retrieval({"question": input_text})
    return chat_response['answer']
