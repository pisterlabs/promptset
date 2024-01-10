from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import tqdm

idx_folder = r'C:\Users\Tim\Documents\gpt_index'

def get_context(pdf_path, question, idx_folder):

    documents = PyPDFLoader(pdf_path).load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    faiss_index = FAISS.from_documents(texts, embeddings)
    faiss_index.save_local(idx_folder)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # load vector store
    #print("loading indexes")
    faiss_index = FAISS.load_local(idx_folder, embeddings)
    #print("index loaded")
    gpt4all_path = r'C:\Users\jc463253\OneDrive - James Cook University\Models\LLM\ggml-model-gpt4all-falcon-q4_0.bin'

    # # Set your query here manually
    # question = "What are the most significant challenges for computer vision in agriculture?"
    matched_docs = faiss_index.similarity_search(question, 4)
    context = ""
    for doc in matched_docs:
        context = context + doc.page_content + " \n\n "

    return context

def get_memoryless_output(llm, prompt, question):

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    output = llm_chain.run(question)
    return output


def ask_pdf(pdf_path, model_path, question):

    all_context = []
    all_prompts = []
    all_output = []
    template = """
    
      # Please use the following context to answer questions.
      # Context: {context}
      #  - -
      # Question: {question}
      # Answer: Let's think step by step."""


    print("Getting Context")

    if isinstance(pdf_path, list):
        for i in tqdm.tqdm(range(len(pdf_path))):
            context = get_context(pdf_path[i], question, idx_folder)
            all_context.append(context)
    else:
        context = get_context(pdf_path, question, idx_folder)
        all_context.append(context)

    print("Getting prompts")

    # for i in tqdm.tqdm(range(len(all_context))):
    #     all_prompts.append(PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=all_context[i]))
    #
    # print("Building model")
    #
    # callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler()])
    # llm = GPT4All(model=model_path, max_tokens=1000, callback_manager=callback_manager, verbose=True, repeat_last_n=0)
    #
    # print("Asking model")
    #
    # for i in tqdm.tqdm(range(len(all_context))):
    #     llm_chain = LLMChain(prompt=all_prompts[i], llm=llm)
    #     output = llm_chain.run(question)
    #     all_output.append(output)
    # return all_output

    callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler()])
    llm = GPT4All(model=model_path, max_tokens=1000, n_predict=1000, callback_manager=callback_manager, verbose=True, repeat_last_n=0)

    print("Getting prompts")
    for i in tqdm.tqdm(range(len(all_context))):
        all_prompts.append(PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=all_context[i]))

    print("Getting output")
    for i in tqdm.tqdm(range(len(all_context))):
        all_output.append(get_memoryless_output(llm, all_prompts[i], question))

    return all_output



