from langchain import OpenAI, LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import VectorStore, Pinecone
from langchain.chains import ConversationalRetrievalChain, LLMChain, load_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pinecone, os

# from langchain.vectorstores.faiss import FAISS

template = """Assistant is a LLM trained by OpenAI.
Assistant is a language model designed to assist with academic questions and provide guidance as an academic mentor/tutor. With a vast knowledge base and the ability to process natural language, Assistant can answer questions on a wide range of subjects, including but not limited to, law, economics, history, science, mathematics, and literature.

As an academic mentor/tutor, Assistant can provide guidance on study habits, essay writing, research methodologies, and exam preparation. Whether you are a high school student, college student, or graduate student, Assistant is equipped to provide the support and guidance you need to succeed academically.

With its advanced capabilities, Assistant can also help you generate ideas for research projects, provide feedback on writing assignments, and suggest relevant sources of information to support your academic endeavors.

No matter what your academic needs are, Assistant is here to help you achieve your goals.
{history}
Human: {user_input}
Assistant:"""

api_key = os.environ.get("OPENAI_API_KEY")


def load_embed_pinecone() -> None:
    """
    Creates a pinecone index for embeddings of the Sipser book.
    """
    # print(os.getcwd())
    loader = PyPDFLoader("../Sipser_theoryofcomp.pdf")
    # loader = TextLoader('./temp.txt')
    toc_pdf = loader.load()
    # print(doc)
    # print(doc[2])
    # print(f'len of toc_doc: {len(toc_pdf)}')
    # print(f'chars in first doc toc_doc[1]: {len(toc_pdf[474].page_content)}')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_toc_doc = text_splitter.split_documents(toc_pdf)

    pinecone.init(
        api_key="5bf2927b-0fb7-423b-b8f1-2f6a3347a15d",
        environment="asia-northeast1-gcp",
    )
    embedding = OpenAIEmbeddings()
    pine_cone_index_of_local_embeddings = Pinecone.from_texts(
        [doc.page_content for doc in split_toc_doc],
        embedding,
        index_name="teamprojindex",
    )

    # print(f'len of split_toc_doc: {len(split_toc_doc)}')
    # print(f'chars in first doc split_toc_doc[1]: {len(split_toc_doc[474].page_content)}')
    # print(split_toc_doc)
    # faiss_vecstore = FAISS.from_documents(doc, OpenAIEmbeddings())

    # if not os.path.exists("ndtmvecstore.pkl"):
    # with open("ndtmvecstore.pkl", "wb") as f:
    # pickle.dump(faiss_vecstore, f)


def make_chain(vectorstore: VectorStore) -> ConversationalRetrievalChain:
    alignment_prompt = """You will be asked theoretical computer science questions, if you are unsure of the answer try to answer to the best of your abilities but state that you are unsure. Use the given context to the best of your abilities, your goal is to help the questioner understand the context and answer the question.
    {context}

    Question: {question}
    Helpful Answer:"""

    qa_template = """Given the following chat history and a follow up question, rephrase the follow up input question to be a standalone question.
    if there is no chat history, just rephrase the follow up input question to be a standalone question and answer it to the fullest extent.
    Chat History:\"""
    {chat_history}
    \"""
    Follow Up Input: \"""
    {question}
    \"""
    Standalone question:"""

    ALIGNMENT_QA_PROMPT = PromptTemplate.from_template(alignment_prompt)
    QUESTIONGEN_PROMPT = PromptTemplate.from_template(qa_template)

    main_llm = OpenAI(temperature=0.5)
    streaming_llm = OpenAI(
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
        temperature=0.2,
        max_tokens=256,
    )
    question_gen_chain = LLMChain(llm=main_llm, prompt=QUESTIONGEN_PROMPT)
    qa_doc_chain = load_qa_chain(
        llm=streaming_llm,
        chain_type="stuff",
        prompt=ALIGNMENT_QA_PROMPT,
    )

    chatbot = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=qa_doc_chain,
        question_generator=question_gen_chain,
    )
    return chatbot


# def make_chain_prebuilt(vectorstore: VectorStore) -> str:
# qa = VectorDBQA.from_chain_type(llm=OpenAI(temperature=0.5, verbose=True), chain_type="stuff", vectorstore=vectorstore)
# qa.run(query)
# llm = OpenAI(temperature=0, verbose=True)
# chain = load_chain("./temp.json", vectorstore=vectorstore)
# query = "How would you define the class np? explain it like im 4 years old"
# docs = vectorstore.similarity_search(query)
# return chain.run(input=docs, query=query)
# print(chain.run(query))
