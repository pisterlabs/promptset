import streamlit as st
import streamlit.components.v1 as components

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")

    # Input for weight in kg
    weight_kg = st.sidebar.number_input('Enter your weight (in kg)', value=0.0)

    # Input for height in meters
    height_m = st.sidebar.number_input('Enter your height (in meters)', value=0.0)

    # Conversion factors
    kg_to_lb = 2.20462
    m_to_ft = 3.28084

    # Convert weight to lb
    weight_lb = weight_kg * kg_to_lb

    # Convert height to ft
    height_ft = height_m * m_to_ft

    # Calculate BMI
    if height_m > 0:
        bmi = weight_kg / (height_m ** 2)
    else:
        bmi = 0

    # Display the results
    st.sidebar.write(f'Your weight in pounds: {format(weight_lb, ".1f")} lb')
    st.sidebar.write(f'Your height in feet: {format(height_ft, ".1f")} ft')
    st.sidebar.write(f'Your BMI: {format(bmi, ".1f")}')

    html_code = """
    <iframe src="https://uspreventiveservicestaskforce.org/apps/widget/USPSTFwidget.jsp" title="Prevention TaskForce Widget" allowtransparency="true" style="border: 0; width: 178px; height: 250px; overflow: hidden;" frameborder="0" scrolling="no"></iframe>
    """

    components.html(html_code, height=250)


st.title("🔎 LangChain - FTWR with search")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who is Agent Smith?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)


