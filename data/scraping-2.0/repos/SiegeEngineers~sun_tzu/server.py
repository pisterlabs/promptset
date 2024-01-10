import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

st.set_page_config(page_title="Sun Tzu", page_icon="⚔️", layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title("Chat with Sun Tzu")
st.info("Sun Tzu is an instance of Open AI's GPT 3.5/4 model, trained on custom data sourced from multiple sources like SOTL, Hera, TheViper, and more YouTubers, Discord messages and Reddit posts")
st.write("More info [here](https://github.com/divine-architect/sun_tzu)")

# Get the Open AI API key from the user
api_key = st.text_input('Enter Open AI API key here', type='password')

# Check if the API key is provided by the user
if api_key:
    openai.api_key = api_key

    # Rest of your code remains unchanged
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about build orders, pushing deer, unit counters etc!",
             "avatar": "https://cdn.discordapp.com/attachments/1175850099188977738/1178400584223699055/suntzu.jpg?ex=65760210&is=65638d10&hm=da03f03d65744543ff37e11f147ae5a906da537c54af1c381fdf709353d43ae0&"}
        ]

    @st.cache_resource(show_spinner=False)
    def load_data():
        with st.spinner(text="Loading and indexing data – hang tight! This should take 1-2 minutes."):
            reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
            docs = reader.load_data()
            service_context = ServiceContext.from_defaults(
                llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5,
                           system_prompt="You are an expert on Age of Empires 2, you know everything about Age of Empires 2 etc. Keep your answers technical and based on facts – do not hallucinate features."))
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            return index

    index = load_data()

    if "chat_engine" not in st.session_state.keys():
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        if message['role'] == 'assistant':
            with st.chat_message(message["role"],
                                 avatar='https://cdn.discordapp.com/attachments/1175850099188977738/1178400584223699055/suntzu.jpg?ex=65760210&is=65638d10&hm=da03f03d65744543ff37e11f147ae5a906da537c54af1c381fdf709353d43ae0&'):
                st.write(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant",
                             avatar='https://cdn.discordapp.com/attachments/1175850099188977738/1178400584223699055/suntzu.jpg?ex=65760210&is=65638d10&hm=da03f03d65744543ff37e11f147ae5a906da537c54af1c381fdf709353d43ae0&'):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)
else:
    st.warning("Please enter your Open AI API key.")
