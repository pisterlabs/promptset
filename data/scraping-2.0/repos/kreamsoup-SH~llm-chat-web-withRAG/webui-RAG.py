import streamlit as st
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer,
)

# Title
st.title('LLAMA RAG Demo')
st.divider()

# Load Tokenizer and Model
st.title('Model name and auth token')
model_name = st.text_input('Enter your Hugging Face model name', value="meta-llama/Llama-2-7b-chat-hf")
auth_token = st.text_input('Enter your Hugging Face auth token', value="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")
system_prompt = st.text_area('Enter your system prompt', value="You are a helpful, respectful and honest assistant.")
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Type a message to start a conversation"}]

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Type a message to start a conversation"}]
st.button('Clear Chat History', on_click=clear_chat_history)

# Load Tokenizer and Model
@st.cache_resource
def get_tokenizer_model():
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./model/', token=auth_token)
    # Create model
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    )
    model = AutoModelForCausalLM.from_pretrained(model_name,
            cache_dir='./model/', token=auth_token,
            quantization_config=quantization_config,
            # rope_scaling={"type":"dynamic", "factor":2},
            max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
    )
    return tokenizer, model



def get_rag_queryengine(_tokenizer, model, system_prompt):
    from llama_index.prompts.prompts import SimpleInputPrompt
    from llama_index.llms import HuggingFaceLLM
    system_prompt_ = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")
    llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=256,
                    system_prompt=system_prompt_,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=_tokenizer
                    )
    
    # Create embeddings
    from llama_index.embeddings import LangchainEmbedding
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    from llama_index import ServiceContext
    from llama_index import set_global_service_context
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embeddings
    )
    set_global_service_context(service_context)

    from llama_index import VectorStoreIndex, download_loader
    PyMuPDFReader = download_loader("PyMuPDFReader")
    loader = PyMuPDFReader()
    documents = loader.load_data(file_path='./data/SNUST_international_guidelines.pdf', metadata=True)
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    return query_engine


tokenizer, model = get_tokenizer_model()

engine = get_rag_queryengine(tokenizer, model, system_prompt)

# prompt ="### User:How many majors in SEOULTECH?### Assistant:"
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input('User: ')
if prompt:
    # update(append) chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Here... text streamer does not work as well as I intended with streamlit
# I will try to fix this later
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        # model inference
        output_text = engine.query(prompt)

        placeholder = st.empty()
        placeholder.markdown(output_text)
    st.session_state.messages.append({"role": "assistant", "content": output_text})


