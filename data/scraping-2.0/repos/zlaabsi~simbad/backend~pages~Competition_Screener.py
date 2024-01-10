import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

st.sidebar.success("Select an agent above.")
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

#st.title("🔎 Competition Screener")

#st.set_page_config(page_title="Competition Screener", page_icon="🔎")

st.markdown("# Competition Screener")
st.sidebar.header("Competition Screener")
st.write(
    """Identify the market leaders and their Value Propositions. Enjoy!"""
)



if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Competition Screener agent who can search the web for market competition. Please enter a target product."}
    ]

if target_product := st.chat_input(placeholder="Write a target product, for example : fitness app"):
    st.session_state.messages.append({"role": "user", "content": target_product})
    st.chat_message("user").write(target_product)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Define your prompt template here
    competition_prompt = f'''
    You are a Competition Screener
    GOALS: Identify the market leaders for {target_product} and their Value Propositions
    INSTRUCTION: 
    - Inform me about the latest information about the list of Most popular {target_product}, add the link to the website that you used",
    - Identify the Value Proposition, Weaknesses and Strengths of each most popular {target_product} in the list. Add a link to your information source. ",
    CONSTRAINTS :
    - If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
    '''

    st.session_state["messages"].append({"role": "assistant", "content": competition_prompt})

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)