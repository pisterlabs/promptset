import streamlit as st
from llama_index import download_loader, VectorStoreIndex, StorageContext, ServiceContext, LLMPredictor, load_index_from_storage
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import urllib.request
import urllib.parse
import os, json

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY'] 
chat = ChatOpenAI(model_name="gpt-3.5-turbo")
llm_predictor = LLMPredictor(llm=chat)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

if not os.path.exists('book'):
    os.mkdir('book')
# if not os.path.exists('storage'):
#     os.mkdir('storage')

url = "https://www.impromptubook.com/wp-content/uploads/2023/03/impromptu-rh.pdf"

def llm(prompt):
    res = chat([HumanMessage(content=prompt)])
    return res.dict()['content']

@st.cache_data
def get_book(url):
    # Parse the URL
    parsed_url = urllib.parse.urlparse(url)
    # Extract the path from the parsed URL
    path = parsed_url.path
    # Use os.path to get the file name from the path
    file_name = os.path.basename(path)
    # Unquote the file name to convert %20 to spaces and handle other special characters
    pdf = urllib.parse.unquote(file_name)
    urllib.request.urlretrieve(url, os.path.join('book',pdf))
    return pdf

# load document with LLaMa Index
def load_doc(pdf):
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    documents = loader.load_data(pdf)
    index = VectorStoreIndex.from_documents(documents,service_context=service_context)
    return index

def load_index(folder = 'storage'):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=folder)
    # load index
    index = load_index_from_storage(storage_context)
    return index

def change_prompt(prompt):
    st.session_state.prompt = prompt


st.title("🤖 Question Answering on Book")
# st.header("📖 Impromptu")
st.markdown(f"Source: 📖 [Impromptu]({url})")

# folder = 'chap4/storage'
folder = 'storage'

try:
    with st.spinner('Loading index...'):
        index = load_index(folder)
    # st.write(f'Index loaded from: {folder}')
except:
    pdf = get_book(url)
    with st.spinner('Creating index...'):
        index = load_doc(os.path.join('book',pdf))
        index.storage_context.persist(folder)
    # st.write(f'Creating and saving index to: {folder}')

if 'prompt' not in st.session_state:
    st.session_state.prompt = "what is the potential of AI in education?"

query = st.text_area('Query',value=st.session_state.prompt)
if st.button('Answer'):
    query_engine = index.as_query_engine()
    res = query_engine.query(query)
    st.write(res.response)
    with st.expander('Sources:'):
        for source in res.source_nodes:
            st.write(source.node.get_text())
    st.write('Follow-up questions:')
    follow_up = f'''
    Provide a list of of 3 follow-up questions the initial query and the result associated. 
    Here is the initial query:
    {query}
    Here is the result associated:
    {res.response}
    Format the output as a JSON file with a list of the 3 follow-up questions in a field called follow-up
    '''
    q2 = llm(follow_up)
    jason = json.loads(q2)
    # st.write(jason)

    for element in jason['follow-up']:
        st.button(element, on_click=change_prompt, args=(element,))