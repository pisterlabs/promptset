import streamlit as st
import openai

st.markdown("# Create Resumes :card_index_dividers:")
st.sidebar.markdown("# create resumes :card_file_box:")

model_list = openai.Model.list()
model_names = [model["id"] for model in model_list["data"]]


model_names = ["text-davinci-003"] + model_names

model = st.sidebar.selectbox("Select a model", model_names, index=0)

# Get the model for the selected model
st.sidebar.write(openai.Model.retrieve(model))

st.markdown("## Create a resume!")
prompt = st.text_area(
    "Modify the prompt to your liking", 
    value="Generate in Markdown a resume for a Python Software Engineer with 5+ years of experience in software engineering. Include an introduction of the candidate, the education, the experience, the skills and the contact information."
    )

@st.cache
def create_resume(prompt):
    result = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.9,
        max_tokens=550,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
    )
    return result['choices'][0]['text']

if prompt:
    st.markdown("### Resume")
    st.write(create_resume(prompt))