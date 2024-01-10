from dotenv import load_dotenv
from langchain import OpenAI, FAISS
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

from langchain.text_splitter import CharacterTextSplitter


def importing_pdf(docs, doc_url):
    print (f'You have {len(docs)} characters in your {doc_url} data')

    # 切割文檔
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=700,
                                          chunk_overlap=30,
                                          length_function = len)

    # 加載切割後的文檔
    split_docs = text_splitter.split_text(docs)

    print (f'You have {len(split_docs)} split document(s)')

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(split_docs, embeddings)
    db.save_local("save_pdf_index")

    response = "我已經讀取了 PDF，可以開始提問了！"

    return response

def chat_pdf(text):
    embeddings = OpenAIEmbeddings()
    new_db  = FAISS.load_local("save_pdf_index", embeddings)
    docsearch = new_db.similarity_search(text)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    with get_openai_callback() as cb:
        response = chain.run(input_documents=docsearch[:3], question=text)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    return response