import os
import streamlit as st
from llama_index import Document, SummaryIndex, LLMPredictor, ServiceContext, load_index_from_storage, VectorStoreIndex
from llama_index.llms import OpenAI

class Temp:
    def __init__(self):
        self.DEFAULT_TERM_STR = (
        "Make a list of terms and definitions that are defined in the context, "
        "with one pair on each line. "
        "If a term is missing it's definition, use your best judgment. "
        "Write each line as as follows:\nTerm: <term> Definition: <definition>"
    )
    
    def get_llm(self, llm_name, model_temperature, api_key, max_tokens=256):
        os.environ['OPENAI_API_KEY'] = api_key
        return OpenAI(temperature=model_temperature, model=llm_name, max_tokens=max_tokens)

    def extract_terms(self, documents, term_extract_str, llm_name, model_temperature, api_key):
        llm = self.get_llm(llm_name, model_temperature, api_key, max_tokens=1024)

        service_context = ServiceContext.from_defaults(llm=llm,
                                                    chunk_size=1024)

        temp_index = SummaryIndex.from_documents(documents, service_context=service_context)
        query_engine = temp_index.as_query_engine(response_mode="tree_summarize")
        terms_definitions = str(query_engine.query(term_extract_str))
        terms_definitions = [x for x in terms_definitions.split("\n") if x and 'Term:' in x and 'Definition:' in x]
        # parse the text into a dict
        terms_to_definition = {x.split("Definition:")[0].split("Term:")[-1].strip(): x.split("Definition:")[-1].strip() for x in terms_definitions}
        return terms_to_definition

    def insert_terms(self, terms_to_definition):
        for term, definition in terms_to_definition.items():
            doc = Document(text=f"Term: {term}\nDefinition: {definition}")
            st.session_state['llama_index'].insert(doc)

    @st.cache_resource
    def initialize_index(self, llm_name, model_temperature, api_key):
        """Create the VectorStoreIndex object."""
        llm = self.get_llm(llm_name, model_temperature, api_key)

        service_context = ServiceContext.from_defaults(llm=llm)

        index = VectorStoreIndex([], service_context=service_context)

        return index

    def display(self):
        if 'all_terms' not in st.session_state:
            st.session_state['all_terms'] = self.DEFAULT_TERM_STR
        st.title("🦙 Llama Index Term Extractor 🦙")
        setup_tab, terms_tab, upload_tab, query_tab = st.tabs(
            ["Setup", "All Terms", "Upload/Extract Terms", "Query Terms"]
        )

        with terms_tab:
            st.subheader("Current Extracted Terms and Definitions")
            st.json(st.session_state["all_terms"])
            

            
        with setup_tab:
            st.subheader("LLM Setup")
            api_key = st.text_input("Enter your OpenAI API key here", type="password")
            llm_name = st.selectbox('Which LLM?', ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"])
            model_temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, step=0.0001)
            term_extract_str = st.text_area("The query to extract terms and definitions with.", value=self.DEFAULT_TERM_STR)

        with query_tab:
            st.subheader("Query for Terms/Definitions!")
            st.markdown(
                (
                    "The LLM will attempt to answer your query, and augment it's answers using the terms/definitions you've inserted. "
                    "If a term is not in the index, it will answer using it's internal knowledge."
                )
            )
            if st.button("Initialize Index and Reset Terms", key="init_index_2"):
                st.session_state["llama_index"] = self.initialize_index(
                    llm_name, model_temperature, api_key
                )
                st.session_state["all_terms"] = {}

            if "llama_index" in st.session_state:
                query_text = st.text_input("Ask about a term or definition:")
                if query_text:
                    query_text = query_text + "\nIf you can't find the answer, answer the query with the best of your knowledge."
                    with st.spinner("Generating answer..."):
                        response = st.session_state["llama_index"].query(
                            query_text, similarity_top_k=5, response_mode="compact"
                        )
                    st.markdown(str(response))

        with upload_tab:
            st.subheader("Extract and Query Definitions")
            
            if st.button("Initialize Index and Reset"):
                st.session_state['llama_index'] = self.initialize_index(llm_name, model_temperature, api_key)
                st.session_state['all_terms'] = {}
            
            if "llama_index" in st.session_state:
                data_type = st.selectbox('What type of data are you sharing?', ["pdf", "website", "github repo", "string"])
                
                if data_type == "pdf":
                    st.markdown("Upload a PDF file")
                    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
                    if uploaded_file:
                        st.session_state['terms'] = {}
                        terms_docs = {}
                        with st.spinner("Extracting..."):
                            terms_docs.update(self.extract_terms([Document(file=uploaded_file)], term_extract_str, llm_name, model_temperature, api_key))
                        st.session_state['terms'].update(terms_docs)
                elif data_type == "website":
                    st.markdown("Enter a URL")
                    url = st.text_input("Enter a URL")
                    if url:
                        st.session_state['terms'] = {}
                        terms_docs = {}
                        with st.spinner("Extracting..."):
                            terms_docs.update(self.extract_terms([Document(url=url)], term_extract_str, llm_name, model_temperature, api_key))
                        st.session_state['terms'].update(terms_docs)
                elif data_type == "github repo":
                    st.markdown("Enter a GitHub repo URL")
                    github_repo = st.text_input("Enter a GitHub repo")
                    if github_repo:
                        st.session_state['terms'] = {}
                        terms_docs = {}
                        with st.spinner("Extracting..."):
                            terms_docs.update(self.extract_terms([Document(github_repo=github_repo)], term_extract_str, llm_name, model_temperature, api_key))
                        st.session_state['terms'].update(terms_docs)
                elif data_type == "string":
                    st.markdown("Enter a string")
                    string = st.text_input("Enter a string")
                    
                
                
                st.markdown("Either upload an image/screenshot of a document, or enter the text manually.")
                document_text = st.text_area("Or enter raw text")
                if st.button("Extract Terms and Definitions") and (uploaded_file or document_text):
                    st.session_state['terms'] = {}
                    terms_docs = {}
                    with st.spinner("Extracting..."):
                        terms_docs.update(self.extract_terms([Document(text=document_text)], term_extract_str, llm_name, model_temperature, api_key))
                    st.session_state['terms'].update(terms_docs)

                if "terms" in st.session_state and st.session_state["terms"]:
                    st.markdown("Extracted terms")
                    st.json(st.session_state['terms'])

                    if st.button("Insert terms?"):
                        with st.spinner("Inserting terms"):
                            self.insert_terms(st.session_state['terms'])
                        st.session_state['all_terms'].update(st.session_state['terms'])
                        st.session_state['terms'] = {}
                        st.experimental_rerun()