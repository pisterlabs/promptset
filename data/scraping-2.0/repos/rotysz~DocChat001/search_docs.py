import os
import sys
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.vectorstores import Pinecone
import pinecone


def GetEmbeddings(_directory_path,_index_file_name, _splitter_class, _embeddings, _gen_emb=False, _chunk_size=1000, _chunk_overlap=0,_separators=None):
    if _gen_emb:
        # List all files in the directory
        files = os.listdir(_directory_path)
        # Iterate over each file in the directory
        docs = []
        for file_name in files:
            if file_name.endswith('.txt'):
                loader = TextLoader(os.path.join(_directory_path, file_name))
                documents = loader.load()
                text_splitter = _splitter_class(chunk_size= _chunk_size, chunk_overlap=_chunk_overlap, separators=_separators)
                docs.extend(text_splitter.split_documents(documents))
        db = FAISS.from_documents(docs, _embeddings)
        db.save_local(_index_file_name)
    else:
        db = FAISS.load_local(_index_file_name, embeddings)
    return db

def GetEmbeddingsPineCone(_directory_path,_index_name, _splitter_class, _embeddings, _gen_emb=False, _chunk_size=1000, _chunk_overlap=0,_separators=None):

    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],  # find at app.pinecone.io
        environment="us-east4-gcp"  # next to api key in console
    )
    if _gen_emb:
        # List all files in the directory
        files = os.listdir(_directory_path)
        # Iterate over each file in the directory
        docs = []
        for file_name in files:
            if file_name.endswith('.txt'):
                loader = TextLoader(os.path.join(_directory_path, file_name))
                documents = loader.load()
                text_splitter = _splitter_class(chunk_size= _chunk_size, chunk_overlap=_chunk_overlap, separators=_separators)
                docs.extend(text_splitter.split_documents(documents))
        db = Pinecone.from_documents(docs, _embeddings,index_name=_index_name)
    else:
        db = Pinecone.from_existing_index(index_name=_index_name,embedding=_embeddings)
    return db


def GetQuestion( _query, _memory, _temperature=0,_max_tokens=256):
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. Question should be in Polish. 
    Do not repeat the question from the conversation.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question in Polish:"""

    Q_PROMPT = PromptTemplate.from_template(_template)
    chain = LLMChain(llm=ChatOpenAI(temperature=_temperature, max_tokens=_max_tokens), memory=_memory, prompt=Q_PROMPT)
    output = chain.predict(question=_query)
    return output

def GetAnswer(_query:str, vectorstore, _temperature=0,_max_tokens=256 ,_search_elements=4):

    docs = vectorstore.similarity_search(query, k=_search_elements)
    total_words = 0
    for i in range(len(docs)):
        total_words += len(docs[i].page_content.split())
        if total_words > 1200:
            docs = docs[:i]
            break

    prompt_template_p = """ Użyj poniższego kontekstu do wygenerowania wyczerpującej odpowiedzi na końcu. Po podaniu odpowiedzi zasugeruj zbliżone zagadnienia zgodne z kontakstem.
                    Jeżeli nie znasz odpowiedzi odpowiedz Nie wiem, nie staraj się wymyślić odpowiedzi.
                    {context}

                    Pytanie: {question}
                    Odpowiedź:"""

    PROMPT = PromptTemplate(
        template=prompt_template_p, input_variables=["context", "question"]
        )

    print(f'Pytanie -> {_query}\n')
    chain = load_qa_chain(ChatOpenAI(temperature=_temperature, max_tokens=_max_tokens), chain_type="stuff", prompt=PROMPT,verbose=False)
    output = chain({"input_documents": docs, "question": _query}, return_only_outputs=False)
    return output

def PrintAnswer(output, _print_context=False):
    print(f'Odpowiedź -> {output["output_text"]}\n')

    print("Zrodła:")
    for doc in output["input_documents"]:
        print(f'[{len(doc.page_content.split())}, {doc.metadata}]')
    if _print_context:
        print('Konteksty:')
        for doc in output["input_documents"]:
            print(
                f'Kontekst [{len(doc.page_content)},{len(doc.page_content.split())}, {doc.metadata}]-> {doc.page_content}\n')
    print("")
    return

GEN_EMBEDDINGS = False
print_context = False

if sys.argv[1].lower() == "gen":
    GEN_EMBEDDINGS = True

if sys.argv[2].lower() == "trace":
    print_context = True

if sys.argv[3].lower() == "PINECONE":
    vestorstore= "PINECONE"
else:
    vestorstore= "FAISS"

print (f" ===== DocBot V .001 ====== [gen embeddings: {GEN_EMBEDDINGS} trace: {print_context}]")

embeddings = OpenAIEmbeddings()
history = ChatMessageHistory()
#memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")
memory = ConversationBufferWindowMemory(return_messages=True,memory_key="chat_history",k=4)
if vestorstore == "FAISS":
    db = GetEmbeddings("input", "srch_idx", RecursiveCharacterTextSplitter, embeddings, GEN_EMBEDDINGS,
                   _chunk_size=3000, _chunk_overlap=0, _separators=[ "\n\n", "\n"," "])
elif vestorstore == "PINECONE":
    db = GetEmbeddingsPineCone("input", "docchat", RecursiveCharacterTextSplitter, embeddings, GEN_EMBEDDINGS,
                   _chunk_size=3000, _chunk_overlap=0, _separators=[ "\n\n", "\n"," "])

while True:
    #get query from user
    query = input("Pytanie: ")
    if query.lower() == 'q':
        break
    output_q = GetQuestion(query, memory)
    query = output_q

    #query = "Jakie kryteria wziąć pod uwagę wybierając księgowość dla małej spółki ?"
    #query="Na jakie wspatrcie unijne może liczyć mała firma?"

    output = GetAnswer(query, db,_temperature=0, _max_tokens=512 ,_search_elements=4)
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(output["output_text"])
    PrintAnswer(output,print_context)

print ("Bot stopped.")

