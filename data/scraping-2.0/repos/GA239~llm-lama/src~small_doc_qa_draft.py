import os
import unicodedata
from functools import partial

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import NLTKTextSplitter, MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma

from runner import get_cpp_lama
from runner import get_local_model_path

DOC_NUM = 3


def get_data_from_dir(path):
    text_loader_kwargs = {"autodetect_encoding": True}
    loader = DirectoryLoader(path, show_progress=True, loader_cls=UnstructuredMarkdownLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    print(f"len(docs): {len(docs)}")
    for d in docs:
        d.page_content = unicodedata.normalize("NFKD", d.page_content)
    return docs


def split_text(data):
    # RecursiveCharacterTextSplitter does not create overlaps. I don't know why.
    text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(data)
    return all_splits


def split_text_md(data):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_docs = [markdown_splitter.split_text(x.page_content) for x in data]
    print(md_docs)

    merged = sum(md_docs, [])
    print(len(merged))
    return split_text(merged)


def vectorize_text(splits):
    # TODO: test other vector stores
    vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceInstructEmbeddings())
    return vectorstore


def test_vector_store(vectorstore: Chroma):
    question = "What are the Steps to Create an Effective Template?"
    docs = vectorstore.similarity_search_with_score(question, k=DOC_NUM)
    for doc, score in docs:
        print(score, " " * 10, doc)


def get_model(config: dict):
    m_name = config["m_name"]
    model_file = config["model_file"]
    model_path = os.path.join(get_local_model_path(m_name), model_file)
    return get_cpp_lama(m_name, model_path, lang_chain=True)


def create_chain(model, vectorstore):
    """
    Run an LLMChain with either model by passing in the retrieved docs and a simple prompt.

    It formats the prompt template using the input key values provided and passes the formatted string
    to LLama-V2, or another specified LLM.

    We can use a QA chain to handle our question above.
    chain_type="stuff" means that all the docs will be added (stuffed) into a prompt.
    """
    # Prompt
    #     Answer in one or maximum two sentences.
    template = """Use the following context to answer the question.

    Context delimited by triple backticks:
    ```{context}```

    Question delimited by triple backticks: ```{question}```
    Answer: """

    qa_chain_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=template,
    )
    search_kwargs = {"k": DOC_NUM}

    # Chain
    chain_type_kwargs = {"prompt": qa_chain_prompt, "verbose": True}
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",  # todo: debug. "stuff" just adds all documents in the prompt. Test more smart approaches.
        chain_type_kwargs=chain_type_kwargs,
        retriever=vectorstore.as_retriever(search_kwargs=search_kwargs),  # TODO: test mmr
        return_source_documents=True,
    )

    return chain


def check_similarity_search(question):
    vs, _ = prepare_vector_store()
    print("get relevant docs")
    docs = vs.similarity_search_with_score(question, k=DOC_NUM)
    print("::DEBUG::")
    for doc in docs:
        print(doc)
    print("::DEBUG::")


def prepare_vector_store(directory):
    print(f"get data.. from {directory}")
    dt = get_data_from_dir(directory)
    sp_dt = split_text(dt)
    print(sp_dt)

    print("vectorisation")
    return vectorize_text(sp_dt), sp_dt


def prepare_chain(config):
    vs, sp_dt = prepare_vector_store(config["directory"])
    model = get_model(config)

    print("create chain")
    return create_chain(model, vs), sp_dt


def ask_question(question, chain, sp_dt, json_output=True):
    repl = chain({"input_documents": sp_dt, "query": question}, return_only_outputs=False)
    # print(repl['result'])
    # print(len(repl['input_documents']))
    # print(len(repl['source_documents']))
    sources = {x.metadata["source"] for x in repl["source_documents"]}
    # print(sources)
    if json_output:
        return {"question": question, "result": repl["result"], "sources": sources}

    answer = repl["result"]
    sources_list = "\n * ".join(sources)
    return f"{answer} \n\nSources:\n * {sources_list}"


def create_chat(config: dict):
    ch, sp_dt = prepare_chain(config=config)
    return partial(ask_question, chain=ch, sp_dt=sp_dt, json_output=config.get("json_output", True))


def experiment(config: dict, questions):
    chat = create_chat(config)
    return [chat(q) for q in questions]


def run_experiments():
    questions = ["Where I can put information in case I know that person is going to leave project?"]
    confs = [
        {
            "m_name": "Llama-2-13B-chat-GGUF",
            "model_file": "llama-2-13b-chat.Q4_K_S.gguf",
        },
        {
            "m_name": "Llama-2-7b-Chat-GGUF",
            "model_file": "llama-2-7b-chat.Q4_K_S.gguf",
        },
    ]
    for conf in confs:
        print(f"Running experiment with {conf}")
        output = {"results": experiment(conf, questions), "config": conf}
        m_name = conf["m_name"]
        with open(f"{m_name}_results.txt", "w") as f:
            f.write(str(output))


def main(config: dict):
    chat = create_chat(config)
    while True:
        question = input("Ask a question: ")
        if question == "exit":
            break
        print(chat(question))


if __name__ == "__main__":
    cfg = {
        "m_name": "Llama-2-13B-chat-GGUF",
        "model_file": "llama-2-13b-chat.Q4_K_M.gguf",
        "json_output": False,
        "directory": "../docs/DeliveryCentral",
    }
    main(cfg)
