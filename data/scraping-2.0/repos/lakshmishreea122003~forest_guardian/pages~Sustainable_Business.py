import streamlit as st 
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

st.set_page_config(
    page_title="Doubt Solver",
    page_icon="🤔",
)

st.markdown("<h1 style='color: #3B444B; font-style: italic; font-family: Comic Sans MS; font-size:4rem' >Sustainable Scope</h1> <h3 style='color:#00755E; font-style: italic; font-family: Comic Sans MS; font-size:2rem'>Amazon Forest Sustainability Analysis & Alternatives forBusiness and Agriculture using LLM</h3>", unsafe_allow_html=True)

prompt = st.text_input('Have a doubt 🤯! Ask here🧠') 
doubt_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Solve doubt related to sustainability on {topic} for amazon forest protection'
)

check_template = PromptTemplate(
    input_variables = ['topic'], 
    template='check is business description {topic} is sustainable or no in amazon forest. Will it harm the forest too much. If yes, give the solutions'
)
llm = OpenAI(temperature=0.9) 

doubt_chain = LLMChain(llm=llm, prompt=doubt_template, verbose=True, output_key='doubt')
check_chain = LLMChain(llm=llm, prompt=check_template, verbose=True, output_key='check')


# Initialize language model
llm1 = OpenAI(model_name="text-davinci-003", temperature=0)
summarize_chain = load_summarize_chain(llm1)

uploaded_file = st.file_uploader(':green[Upload your business report 👇]')


# Show stuff to the screen if there's a prompt
if prompt: 
    doubt = doubt_chain.run(prompt)
    st.write(doubt) 

elif uploaded_file is not None:
    # Save the file to a directory
    with open(os.path.join('D:/llm projects/HealthyWealthy2', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.read())
    st.write('File saved successfully!')
    document_loader = PyPDFLoader(file_path=os.path.join('D:/llm projects/HealthyWealthy2', uploaded_file.name))
    document = document_loader.load()

    # Summarize the document
    summary = summarize_chain(document)
    st.write(summary['output_text'])

    pdf_summary= summary['output_text']
    res = check_chain.run(pdf_summary)
    st.write(res)
