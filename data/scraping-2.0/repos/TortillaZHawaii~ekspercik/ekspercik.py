import streamlit as st

from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOllama
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.embeddings import OpenAIEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.globals import set_debug

set_debug(True)

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")
    openai_model_name = st.text_input("OpenAI Model Name", key="langchain_search_openai_model_name", value="gpt-3.5-turbo")
    db_prefix = st.text_input("DB Prefix", key="langchain_search_db_prefix", value="kosmetologia")
    ollama_model_name = st.text_input("Ollama Model Name", key="langchain_search_ollama_model_name", value="mistral:7b")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/TortillaZHawaii/ekspercik)"
    "[OG Repo](https://codespaces.new/streamlit/llm-examples)"

st.title("🔎 Ekspercik")

is_ollama = ollama_model_name and len(ollama_model_name) > 1
is_openai = openai_api_key and len(openai_api_key) > 1

if is_ollama:
    st.info("🤖 Używam Ollamy: " + ollama_model_name)
    persist_directory = f"./data/db_{ollama_model_name}"
    if db_prefix:
        persist_directory += f"_{db_prefix}"
    embeddings = OllamaEmbeddings(model=ollama_model_name)
    llm = ChatOllama(model=ollama_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    st.session_state["db"] = db

elif openai_api_key and len(openai_api_key) > 1:
    st.info("🤖 Używam OpenAI: " + openai_model_name)
    persist_directory = f"./data/db_openai"
    if db_prefix:
        persist_directory += f"_{db_prefix}"
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")
    llm = ChatOpenAI(model_name=openai_model_name, openai_api_key=openai_api_key, streaming=True)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    st.session_state["db"] = db

else:
    st.info("Prosze podaj nazwę modelu Ollama lub klucz OpenAPI w sidebarze, aby kontynuować")
    st.stop()


uploaded_file = st.file_uploader(
    "Wklej wykład",
    type=["pdf"],
    help="Wklej wykład, który chcesz przeszukać.",
    accept_multiple_files=False,
)


if uploaded_file:
    if st.session_state.get("uploaded_file", None) != uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file

        with st.spinner("📥 Pobieram..."):
            tmp_location = f"/tmp/{uploaded_file.name}"
            with open(tmp_location, "wb") as f:
                f.write(uploaded_file.getbuffer())

        with st.spinner("👀 Czytam z PDFa..."):
            loader = UnstructuredPDFLoader(
                file_path=tmp_location, ocr_languages="eng+pl", strategy="ocr_only",
            )
            raw_documents = loader.load()

        with st.spinner("🔪 Dzielę na zdania..."):
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=120)
            documents = text_splitter.split_documents(raw_documents)

        with st.spinner("🔎 Zapisuje w bazie..."):
            db: Chroma = st.session_state["db"]
            db.add_documents(documents)
            db.persist()
            st.session_state["db"] = db
            st.success("🎉 Gotowe!")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Heja, jestem chatbotem, który czyta PDFy. Jak mogę Ci pomóc?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Podsumuj wykład w jednym zdaniu"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    memory = ConversationBufferMemory(
        llm=llm,
        return_messages=True,
        input_key='question', output_key='answer',
    )
    db = st.session_state["db"]

    retriever= db.as_retriever(
        search_type="similarity_score_threshold",
        k=5,
        search_kwargs={"score_threshold": 0.5},
        return_metadata=True,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True, max_thought_containers=6)

        chat_history = st.session_state.get("chat_history", [])
        result = qa(
            {"question": prompt, 'chat_history': chat_history}, callbacks=[st_cb],
        )

        response = result["answer"]
        sources = result["source_documents"]
        # https://github.com/langchain-ai/langchain/issues/2303#issuecomment-1499646042
        # it has to be a tuple, not a list or a dict
        chat_history.append((prompt, response))
        st.session_state["chat_history"] = chat_history

    full_answer = result["answer"] + "\n\n" + "\n\n".join([f"📚 {source.metadata}\n\n {source.page_content}" for source in sources])
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    st.chat_message("assistant").write(full_answer)
    