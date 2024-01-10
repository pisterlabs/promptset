from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(page_title="Chat with Llama 2", page_icon="🦙")

# Sidebar contents
with st.sidebar:
    st.title('🦙💬 Chat with Llama 2')
    st.markdown('''
    ## About
    This app is a Llama2-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LLAMA2 CPP](https://github.com/abetlen/llama-cpp-python)
    - [LangChain](https://github.com/langchain-ai/langchain)

    💡 Note: No API key required!
    ''')
    add_vertical_space(5)
    st.write('Made by [Ziyue Li](https://github.com/feynlee)')

# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm HugChat, How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text
## Applying the user input box
with input_container:
    user_input = get_text()

# Model

## Metal set to 1 is enough.
n_gpu_layers = 1
## Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
n_batch = 512

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="./llama-2-13b-chat.ggmlv3.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)

# set up a chatbot
# the conversation history will be automatically inserted into the template
template = """Act as an AI Assistant that only completes the Assistant response to human prompt.
{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

chatbot = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=5),
)

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    response = chatbot.predict(human_input=prompt)
    return response

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))