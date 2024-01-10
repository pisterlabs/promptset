# Successfully used structure data to embedded to vector database,
# there is an issue when during search it will not search all the chunks for structure data
# much use database to search

from dotenv import load_dotenv
import os
import tiktoken
import openai
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import chromadb
from langchain.document_loaders import DirectoryLoader

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY"),


def load_document(file_path: str):
    loader = DirectoryLoader('services/', glob="*.txt", loader_cls=TextLoader)
    # loader = TextLoader(file_path)
    documents = loader.load()
    return documents


def split_documents(document, chunk_size=10000, chunk_overlap=5):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(document)


def calculate_embedding_token(chunks):
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    model_cost = 0.0001 / 1000
    total_token = 0
    for chunk in chunks:
        total_token = total_token + len(encoding.encode(chunk.page_content))
    print(f"{total_token * model_cost:.7f}")


def new_embedded_document(embedding_model):
    documents = load_document("clinics/clinic_data.txt")
    chunks = split_documents(documents)
    calculate_embedding_token(chunks)
    new_client = chromadb.EphemeralClient()
    return Chroma.from_documents(chunks, embedding_model,
                                 collection_name="openai_collection",
                                 persist_directory="store/")


def load_from_desk(embedding_model):
    persistent_client = chromadb.PersistentClient()
    # collection = persistent_client.get_or_create_collection("openai_collection")
    # collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

    return Chroma(
        collection_name="openai_collection",
        embedding_function=embedding_model,
        persist_directory='store/'
    )


# select embedding model to use
embedding_model_1 = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

# openai_lc_client = load_from_desk(embedding_model_1)
openai_lc_client = new_embedded_document(embedding_model_1)

with get_openai_callback() as cb:

    pre_prompt = """[INST] <<SYS>>\n
    You can only answer my question using the context I provided if you don't find the answer just answer 'I do not know your question'.
    You will based on the question to find which service is suitable for me.

    <<Response format follow below>>
    Services:
    Information:
    ...
    \n\n"""
    context = "CONTEXT:\n\n{context}\n" + "Question: {question}" + "[\INST]"
    prompt = pre_prompt + context
    rag_prompt_custom = PromptTemplate(template=prompt, input_variables=["context", "question"])

    # integrate prompt with LLM
    qa = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0),
                                               retriever=openai_lc_client.as_retriever(),
                                               return_source_documents=True,
                                               combine_docs_chain_kwargs={"prompt": rag_prompt_custom},
                                               verbose=True)

    chat_history = []
    query = "I have yellowish tooth problem"
    # query = "I have toothache problem"
    result = qa({"question": query, "chat_history": chat_history})
    print(result["answer"])
    print(f"{result}")
    print(f"--------------------------------")
    # query = "Format the result based on instructions"
    # result = qa({"question": query, "context": result["answer"], "chat_history": chat_history})
    # result = qa({"question": "where got dentist do laser", "chat_history": chat_history})
    chat_history = [("question", result["answer"])]
    # print(result["answer"])
    # print(f"{result}")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")


# with get_openai_callback() as cb:
#     # template = """You can only answer my question using the data I provided if you don't find the answer just answer 'I do not know your question.'
#     # For example:
#     # User Location:
#     # Latitude: 3.139003
#     # Longitude: 101.849489
#     # Clinic A:
#     # Latitude: 3.139003
#     # Longitude: 101.857662
#     # User Latitude - Clinic A Latitude = Latitude distance
#     # 3.139003 - 3.139003 = Math.abs(0.00)
#     # User Longitude - Clinic A Longitude = Longitude distance
#     # 101.849489 - 101.857662 = Math.abs(-0.008173)
#     # Clinic B:
#     # Latitude: 3.139003
#     # Longitude: 101.859489
#     # User Latitude - Clinic B Latitude = Latitude distance
#     # 3.139003 - 3.139003 = Math.abs(0.00)
#     # User Longitude - Clinic B Longitude = Longitude distance
#     # 101.849489 - 101.859489 = Math.abs(-0.01)
#     #
#     # Clinic A is the nearest answer. the lesser is the closer
#     # {context}"""
#     # rag_prompt_custom = PromptTemplate.from_template(template)
#     #
#     # # integrate prompt with LLM
#     # rag_prompt_custom.format(context="""
#     # Provide your response as a JSON object with the following schema, and don't reply other than JSON:
#     #  {"clinics": [{"name":"", "differentLat": 0, "differentLong": 0}], "nearestClinic": {},
#     #     "instructions": ["", "", ... ]}
#     # """)
#
#     pre_prompt = """[INST] <<SYS>>\n
#     You can only answer my question using the context I provided if you don't find the answer just answer 'I do not know your question'.
#
#     <<Calculate Distance>>
#     lat=lat2−lat1 (difference in latitude)
#     long=long2−long1 (difference in longitude)
#     R is the radius of the Earth (mean radius = 6,371 km)
#     lat1 and long1 are the coordinates of the first point
#     lat2 and long2 are the coordinates of the second point
#     The result distance must be in kilometers.
#     The closest to the user is the answer.
#
#     <<Response format follow below>>
#     Clinic Name:
#     Services:
#     Location Latitude:
#     Location Longitude:
#     Operation Time:
#     --
#     Clinic Name:
#     Services:
#     Location Latitude:
#     Location Longitude:
#     Operation Time:
#     ...
#     \n\n"""
#     context = "CONTEXT:\n\n{context}\n" + "Question: {question}" + "[\INST]"
#     prompt = pre_prompt + context
#     # print(prompt)
#     # print(f"--------------------------------")
#     rag_prompt_custom = PromptTemplate(template=prompt, input_variables=["context", "question"])
#
#     # integrate prompt with LLM
#     qa = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0),
#                                                retriever=openai_lc_client.as_retriever(),
#                                                return_source_documents=True,
#                                                combine_docs_chain_kwargs={"prompt": rag_prompt_custom},
#                                                verbose=True)
#
#     chat_history = []
#     query = "Where got dentist do Laser Dentistry, near me, User Location: Latitude=3.139003 | Longitude=101.849489"
#     result = qa({"question": query, "chat_history": chat_history})
#     print(result["answer"])
#     print(f"{result}")
#     print(f"--------------------------------")
#     # query = "Format the result based on instructions"
#     # result = qa({"question": query, "context": result["answer"], "chat_history": chat_history})
#     # result = qa({"question": "where got dentist do laser", "chat_history": chat_history})
#     chat_history = [("question", result["answer"])]
#     # print(result["answer"])
#     # print(f"{result}")
#     print(f"Total Tokens: {cb.total_tokens}")
#     print(f"Prompt Tokens: {cb.prompt_tokens}")
#     print(f"Completion Tokens: {cb.completion_tokens}")
#     print(f"Total Cost (USD): ${cb.total_cost}")
