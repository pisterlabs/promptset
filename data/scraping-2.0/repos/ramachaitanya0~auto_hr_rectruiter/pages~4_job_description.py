import streamlit as st
from langchain.chat_models import  AzureChatOpenAI
from langchain.callbacks import get_openai_callback

if "Amount_Spent" not in st.session_state :
    st.session_state["Amount_Spent"] = 0.0

st.header("Get Job Description ",divider='rainbow')

cols = st.columns(3)
with cols[0] :
    selected_role = st.selectbox(label="Role",
                                 options=['Software Development Engineer', 'Data Scientist', 'Data Engineer',
                                          'QA Tester'])
with cols[1] :
    yoe = st.number_input(label="Years of Experience", step=1)

with cols[2] :
    key_skills = st.text_input(label="Key Skills")



job_description_input = f""" Write a Job Description of {selected_role} with {yoe} Years of Experience with key skills as {key_skills}  """

# job_description_input =  st.text_input(label="Get a Job Description")
jd_button = st.button("Submit")



if jd_button :
    with get_openai_callback() as cr :
        model = AzureChatOpenAI()
        val = model.invoke(job_description_input, engine="gpt_4_32k")
        st.session_state.Amount_Spent += cr.total_cost

    st.write(val.content)


