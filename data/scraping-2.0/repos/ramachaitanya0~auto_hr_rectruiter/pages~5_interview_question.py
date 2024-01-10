import streamlit as st
from langchain.chat_models import  AzureChatOpenAI
from langchain.callbacks import get_openai_callback
from dotenv import  load_dotenv
load_dotenv()

if "Amount_Spent" not in st.session_state :
    st.session_state["Amount_Spent"] = 0.0

st.header("Get Question and Answers",divider='rainbow')
cols = st.columns(3)
with cols[0] :
    no_of_questions = st.number_input(label="Number of Questions",step=1)
with cols[1] :
    yoe = st.number_input(label="Years of Experience", step=1)
with cols[2] :
    selected_role = st.selectbox(label="Role",
                                 options=['Software Development Engineer', 'Data Scientist', 'Data Engineer', 'QA Tester'])

qa_input = f"""Write {no_of_questions} questions and its answers to ask an interview \
candidate for {selected_role} role with {yoe} Years of Experience  """
jd_button = st.button("Submit")


if jd_button :
    with get_openai_callback() as cr :
        model = AzureChatOpenAI()
        val = model.invoke(qa_input, engine="gpt_4_32k")
        st.session_state.Amount_Spent += cr.total_cost

    st.write(val.content)


