import streamlit as st
import os
import re
from llama_index import StorageContext, load_index_from_storage, GPTVectorStoreIndex
from llama_index import LLMPredictor, ServiceContext
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from langchain import OpenAI
os.environ['OPENAI_API_KEY'] = "<OPEN API KEY>"

st.title("Fiskehelserapporter 2019-2022")

storage_2019 = StorageContext.from_defaults(persist_dir="./storage_2019")
storage_2020 = StorageContext.from_defaults(persist_dir="./storage_2020")
storage_2021 = StorageContext.from_defaults(persist_dir="./storage_2021")
storage_2022 = StorageContext.from_defaults(persist_dir="./storage_2022")
#load index
index_2019 = load_index_from_storage(storage_2019)
index_2020 = load_index_from_storage(storage_2020)
index_2021 = load_index_from_storage(storage_2021)
index_2022 = load_index_from_storage(storage_2022)

#Build engine
engine_2019 = index_2019.as_query_engine(similarity_top_k = 3)
engine_2020 = index_2020.as_query_engine(similarity_top_k = 3)
engine_2021 = index_2021.as_query_engine(similarity_top_k = 3)
engine_2022 = index_2022.as_query_engine(similarity_top_k = 3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=engine_2019, 
        metadata=ToolMetadata(name='year_19', description='Inneholder informasjon om fiskehelsen i 2019')
    ),
    QueryEngineTool(
        query_engine=engine_2020, 
        metadata=ToolMetadata(name='year_20', description='Inneholder informasjon om fiskehelsen i 2020')
    ),
    QueryEngineTool(
        query_engine=engine_2021, 
        metadata=ToolMetadata(name='year_21', description='Inneholder informasjon om fiskehelsen i 2021')
    ),
    QueryEngineTool(
        query_engine=engine_2022, 
        metadata=ToolMetadata(name='year_22', description='Inneholder informasjon om fiskehelsen i 2022')
    ),
]
# Setup av språkmodell.
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=-1, streaming=True))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

sub_question_query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools, service_context= service_context)
question = st.text_input("Still spørsmål om fiskehelse fra 2019 til 2022")
if question == '':
    st.stop()
response_s = sub_question_query_engine.query(question)
st.markdown(f'''<h4>{response_s}</h4>''', unsafe_allow_html=True)

st.header("Genererte spørsmål og svar") 
string = re.sub(r"(\n)", "", str(response_s.get_formatted_sources))
split_string = re.split(r"(Sub question:|Response|, doc_id)", string)

def generate_pattern(threshold):
    sequence = [6]  # Start with the initial value of 6
    add_value = 2  # Initial value to add

    while sequence[-1] + add_value <= threshold:  # Check if the next element exceeds the threshold
        sequence.append(sequence[-1] + add_value)  # Add the next element to the sequence
        if add_value == 2:
            add_value = 4  # Change the add_value to 4 if the previous add_value was 2
        else:
            add_value = 2  # Change the add_value to 2 if the previous add_value was 4

    sequence = [x for x in sequence if x <= threshold]  # Remove elements greater than the threshold

    return sequence

n = len(split_string)
pattern = generate_pattern(n)
for index in pattern:
    if index < len(split_string):
        st.markdown(f'''<p>{split_string[index]}</p>''', unsafe_allow_html=True)
