import streamlit as st
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI



# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    import io
    import pypdf
    import docx2txt

    name, extension = os.path.splitext(file.name)
    bytes_data = file.read()

    if extension == '.pdf':
        stream = io.BytesIO(bytes_data)
        pdf_reader = pypdf.PdfReader(stream)
        num_pages = len(pdf_reader.pages)
        data = []
        for i in range(num_pages):  
            page = pdf_reader.pages[i]  
            data.append(page.extract_text())
    elif extension == '.docx':
        stream = io.BytesIO(bytes_data)
        data = [docx2txt.process(stream)]
    elif extension == '.txt':
        data = [bytes_data.decode()]
    else:
        st.error(f"Unsupported file format: \"{extension}")
        return None

    return data


def validate_openai_api_key(api_key):
    import openai

    openai.api_key = api_key

    with st.spinner('Validating API key...'):
        try:
            response = openai.Completion.create(
                engine="davinci",
                prompt="This is a test.",
                max_tokens=5
            )
            # print(response)
            validity = True
        except:
            validity = False

    return validity


# splitting data in chunks
def chunk_data(data, chunk_size=512, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.create_documents(data) # use create_doc without loader
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_with_memory(llm, vector_store, question, chat_history, k=3):
    from langchain.chains import ConversationalRetrievalChain

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k' : k})
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({"question": question, "chat_history": chat_history})

    return question, result["answer"]

def stuff_summary_response(llm, chunks):
    from langchain import PromptTemplate
    from langchain.chains.summarize import load_summarize_chain

    prompt_template = '''
    Write a concise summary of the following text that covers the key points.
    Add a title to the summary.
    Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
    by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.
    Text: `{text}`
    '''
    initial_prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='stuff',
        prompt=initial_prompt,
        verbose=False
    )
    stuff_summary = summary_chain.run(chunks)

    return stuff_summary

def mapReduce_summary_response(llm, chunks):
    from langchain import PromptTemplate
    from langchain.chains.summarize import load_summarize_chain

    map_prompt = '''
    Write a short and concise summary of the following:
    Text: `{text}`
    CONCISE SUMMARY:
    '''
    map_prompt_template = PromptTemplate(
        input_variables=['text'],
        template=map_prompt
    )

    combine_prompt = '''
    Write a concise summary of the following text that covers the key points.
    Add a title to the summary.
    Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
    by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.
    Text: `{text}`
    '''
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=['text'])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=False
    )
    mapReduce_summary = summary_chain.run(chunks)

    return mapReduce_summary


def refine_summary_response(llm, chunks):
    from langchain import PromptTemplate
    from langchain.chains.summarize import load_summarize_chain

    prompt_template = """Write a concise summary of the following extracting the key information:
    Text: `{text}`
    CONCISE SUMMARY:"""
    initial_prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

    refine_template = '''
        Your job is to produce a final summary.
        I have provided an existing summary up to a certain point: {existing_response}.
        Please refine the existing summary with some more context below.
        ------------
        {text}
        ------------
        Start the final summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
        by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.

    '''
    refine_prompt = PromptTemplate(
        template=refine_template,
        input_variables=['existing_response', 'text']
    )

    chain = load_summarize_chain(
    llm=llm,
    chain_type='refine',
    question_prompt=initial_prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=False
    )

    refine_summary = chain.run(chunks)

    return refine_summary


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


# clear the chat history from streamlit session state
def clear_history():
    if 'doc_history' in st.session_state:
        del st.session_state.doc_history
        st.session_state.doc_history = list()



if __name__ == "__main__":

    ############################################################ System Configuration ############################################################

    system_msg = '''Hi customer, I am a Document Consultant powered by OpenAI's GPT-3.5-Turbo model.
            \n\nI can help you with the following:
            \n- Ask any question about the content of your document.
            \n- Generate a concise summary of your document with a click of the above button.
            \n\nPlease enter your OpenAI API Key and upload your file to get started.
            You could also adjust the length, creativity and relevance of the responses.
            '''

    # creating objects in the Streamlit session state
    if 'doc_history' not in st.session_state:
        st.session_state.doc_history = []

    if 'chunks' not in st.session_state:
        st.session_state.chunks = ''

    if 'vs' not in st.session_state:
        st.session_state.vs = ''

    # better to set all the object requiring api keys in the beginning


    ############################################################ SIDEBAR widgets ############################################################ (for params configuration)
    with st.sidebar:
        
        # Setting up the OpenAI API key via secrets manager
        if 'OPENAI_API_KEY' in st.secrets:
            api_key_validity = validate_openai_api_key(st.secrets['OPENAI_API_KEY'])
            if api_key_validity:
                os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
                st.success("✅ API key is valid and set via Encrytion provided by Streamlit")
            else:
                st.error('🚨 API key is invalid and please input again')
        # Setting up the OpenAI API key via user input
        else:
            api_key_input = st.text_input("OpenAI API Key", type="password")
            api_key_validity = validate_openai_api_key(api_key_input)

            if api_key_input and api_key_validity:
                os.environ['OPENAI_API_KEY'] = api_key_input
                st.success("✅ API key is valid and set")
            elif api_key_input and api_key_validity == False:
                st.error('🚨 API key is invalid and please input again')

            if not api_key_input:
                st.warning('Please input your OpenAI API Key')
        
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

        st.divider()

        # Response Configuration Expander
        with st.expander('Response Configuration'):

            # chunk size number widget
            chunk_size = st.number_input('Chunk size:', min_value=100, max_value=10000, value=512) # max 2048

            # k number slider widget
            k = st.slider('Response Length', min_value=1, max_value=20, value=3)
            st.info('More comprehensive response will cost more tokens. (Recomm 3-5)')
            
            temperature = st.slider('Temperature:', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
            st.info('Larger the number, More Creative is the response.')

        # Summary Configuration Expander
        with st.expander('Summary Configuration'):
            summary_chains = ['Stuff', 'Map Reduce', 'Refine']
            summary_chain = st.radio('Summary Chain:', summary_chains, index=0)
            st.info('''Map Reduce & Refine are for large documents, giving a better summarization but require a Paid Tier of your OpenAI account for higher rate limit
                    \n\n(Free Tier only allows 2 chunks for Map Reduce & 3 chunks for Refine)''')

        st.divider()

        if st.button('Clear Chat History'):
            clear_history()


    ############################################################ MAIN PAGE widgets ############################################################
    
    st.title("📂 AI Document Consultant")

    # Upload Doc Expander
    with st.expander('Upload a Document'):
        # File uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt']) 

        if uploaded_file:
            with st.spinner('Reading file ...'):
                data = load_document(uploaded_file)
                st.session_state.chunks = chunk_data(data, chunk_size=chunk_size) # saving the chunks in the streamlit session state (to be persistent between reruns)
                chunks = st.session_state.chunks
                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.info(f'Chunk size: {chunk_size}\n\nChunks: {len(chunks)}\n\nTotal Tokens: {tokens}\n\nEmbedding Cost: ${embedding_cost:.4f}')

        # Add data button widget
        add_data = st.button('Add Data', on_click=clear_history) 

        if api_key_validity and uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Chunking and Embedding file ...'):
                st.session_state.vs = create_embeddings(chunks) # creating the embeddings and returning the Chroma vector store in the streamlit session state (to be persistent between reruns)
                st.success('File uploaded, chunked and embedded successfully.')
        elif not api_key_validity and add_data:
            st.warning('Please enter a valid OpenAI API Key to continue.')
        elif not uploaded_file and add_data:
            st.warning('Please upload your file first.')

    # Summarize document button widget
    summary_button = st.button('Generate Summary') # summarize document button widget

    st.divider()

    st.chat_message('assistant').write(system_msg)
    
    # The process of generating ChatGPT response

    # Display chat messages from history on app rerun
    for question, response in st.session_state.doc_history:
        user_message = {"role": "user", "content": question}
        st.chat_message(user_message["role"]).markdown(user_message["content"])
        AI_message = {"role": "assistant", "content": response}
        st.chat_message(AI_message["role"]).markdown(AI_message["content"])

    # Question input widget
    question = st.chat_input(placeholder="What is the document about?")

    if question and st.session_state.vs:
        if api_key_validity:
            st.chat_message('user').write(question)

            # Generating the question response
            with st.spinner('Generating response ...'):
                vector_store = st.session_state.vs
                llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=temperature)
                question, response = ask_with_memory(llm, vector_store, question, st.session_state.doc_history, k)
                st.session_state.doc_history.append((question, response))
                st.chat_message('assistant').markdown(response)
        elif not api_key_validity: 
            st.warning('Please enter a valid OpenAI API Key to continue.')
    elif question and not st.session_state.vs:
        st.warning('Please upload and add the data of your document first.')

    # Generating the summary of the document
    if summary_button and st.session_state.vs:
        with st.spinner('Summarizing document ...'):
            llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=temperature)
            if summary_chain == 'Stuff':
                summarize_response = stuff_summary_response(llm, st.session_state.chunks)
            elif summary_chain == 'Map Reduce':
                summarize_response = mapReduce_summary_response(llm, st.session_state.chunks)
            elif summary_chain == 'Refine':
                summarize_response = refine_summary_response(llm, st.session_state.chunks)

            question = 'Generate a concise summary of my document'
            st.session_state.doc_history.append((question, summarize_response))

            st.chat_message('user').write(question)
            st.chat_message('assistant').markdown(summarize_response)
    elif summary_button:
        st.warning('Please upload your file first.')