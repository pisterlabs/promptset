# 라이브러리 불러오기
import gradio as gr
import openai
import mysql.connector
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.docstore.document import Document
from langchain.chat_models import AzureChatOpenAI
from PyPDF2 import PdfReader
from langchain.prompts.chat import (
        ChatPromptTemplate,      
        SystemMessagePromptTemplate,       
        HumanMessagePromptTemplate,        
    )


# OpenAI API 키 설정
openai.api_type = "azure"
openai.api_base = ""
openai.api_version = "2023-03-15-preview"
openai.api_key = ""

# MySQL 연결
db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="sample"
)
cursor = db.cursor()


# 테이블 생성
def create_table_if_not_exists():
    cursor.execute("""CREATE TABLE IF NOT EXISTS chat_history (user_input TEXT, bot_response TEXT)""")

create_table_if_not_exists()

# 정보 저장
def save_to_database(user_input, bot_response):
    query = "INSERT INTO chat_history (user_input, bot_response) VALUES (%s, %s)"
    values = (user_input, bot_response)
    cursor.execute(query, values)
    db.commit()

def extract_pdf_text(pdf_file):
    pdf_text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
    return pdf_text


def openAiGPT(message, chat_history) :
    # pdf 불러오기 및 전처리
    documents_pro = [] # 저장 리스트

    reader = PdfReader("Puzzle Mix_Exploiting Saliency and Local Statistics for Optimal Mixup.pdf")
    raw_text = "" # raw_text 초기화

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
            documents_pro.append(raw_text) # documents_pro 에 추가


    # 청크 분할
    doc_chunks = []

    for line in documents_pro:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000, # 최대청크의 최대 길이를 3000자로 제한
                separators=["\n\n", "\n", ".", ""], # 분할 기준
                chunk_overlap=0, # 청크 사이의 중첩
            )
            chunks = text_splitter.split_text(line)
            for i, chunk in enumerate(chunks):  # doc형식으로 변환
                doc = Document(
                    page_content=chunk, metadata={"page": i, "source": "Puzzle Mix_Exploiting Saliency and Local Statistics for Optimal Mixup.pdf"}
                )
                doc_chunks.append(doc) 

    # ChromaDB에 임베딩
    embeddings = OpenAIEmbeddings(model="embeddingada002 ", chunk_size=1)   
    vector_store = Chroma.from_documents(doc_chunks, embeddings) 
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 프롬프트
    system_template="""To answer the question at the end, use the following context. 
        If you don't know the answer, just say you don't know and don't try to make up an answer.
        you tell me the exact information and figures of the monster.
        I want you to act as Monster expert.

        you only answer in Korean

        {summaries}

        """
    messages = [
        SystemMessagePromptTemplate.from_template(system_template), # 시스템 사전 설정
        HumanMessagePromptTemplate.from_template("{question}") # 내 질문 설정
    ]
    prompt = ChatPromptTemplate.from_messages(messages) # prompt 변수에 저장

    chain_type_kwargs = {"prompt": prompt}

    # Langchain & Azure GPT 연결
    llm = AzureChatOpenAI(deployment_name='gpt354k', temperature=0.0, max_tokens=500)

    chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
                                                    chain_type='stuff',
                                                    retriever=retriever,
                                                    return_source_documents=True,
                                                    chain_type_kwargs=chain_type_kwargs,
                                                    reduce_k_below_max_tokens=True
                                                    )

    result = chain(message)
    gpt_message = result['answer']
    print(gpt_message)

    chat_history.append((message, gpt_message))
    return "", chat_history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    pdf_upload = gr.inputs.File(label="Upload PDF")
    user_input = gr.inputs.Textbox(lines=2, label="Your Question")
    pdf_content = gr.outputs.HTML(label="PDF Content")
    bot_output = gr.outputs.HTML(label="Bot Response")
    
    msg.submit(openAiGPT, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()