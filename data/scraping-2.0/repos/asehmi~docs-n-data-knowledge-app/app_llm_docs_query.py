import json

from langchain.prompts import PromptTemplate

import tiktoken
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores import WeaviateVectorStore
from llama_index import (
    VectorStoreIndex, SimpleDirectoryReader, 
    StorageContext, ServiceContext, 
    load_index_from_storage
)
import weaviate

import streamlit as st

from app_state import (state, init_app_state, _set_state_cb)
init_app_state() # ensure all state variables are initialized

from globals import (
    VECTOR_STORE, OPENAI_MODELS_COMPLETIONS, 
    DEFAULT_MODEL_CONFIG, LANG_MODEL_PRICING
)
from common import scrape_articles

# DOCS CHAT PAGE ----------------------------------------------------------------

wc = None
# WEAVIATE CLOUD STORE
if VECTOR_STORE == 'Weaviate':
    auth_config = weaviate.AuthApiKey(api_key=state.weaviate_api_key)
    wc = weaviate.Client(
        url=state.WEAVIATE_URL,
        auth_client_secret=auth_config,
        additional_headers={
            "X-OpenAI-Api-Key": state.openai_api_key,
        }
    )

@st.cache_data(ttl=60*60, show_spinner=False)
def get_llm_doc_query_response(
    query_prompt, model_name: str = DEFAULT_MODEL_CONFIG['completions_model'], 
    _service_context=ServiceContext.from_defaults()
):
    # load index
    # LOCAL STORE
    if VECTOR_STORE == 'Local':
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir='./storage')
        index = load_index_from_storage(storage_context)

    # WEAVIATE CLOUD STORE
    elif VECTOR_STORE == 'Weaviate':
        vector_store = WeaviateVectorStore(weaviate_client = wc, index_name="Documents", text_key="content")
        # set up the index
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=_service_context)

    else:
        raise ValueError(f'Unknown vector store {VECTOR_STORE}')

    # get query engine over the index
    query_engine = index.as_query_engine()
    # query the index
    response = query_engine.query(query_prompt)
    response = response.response.replace('•', '*')
    return response

def main(title, user_input_confirmed=False):
    # Count token usage for cost estimation
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(state.completions_model).encode,
        verbose=False  # set to true to see usage printed to the console
    )
    callback_manager = CallbackManager([token_counter])
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)
    
    def _index_documents():
        # load the documents 
        documents = SimpleDirectoryReader('docs').load_data()

        # LOCAL STORE
        # NOTE: Disallow if cloud deployment (temporary fix for public demo and/or if you 
        # don't have required file permissions or disk space)
        if not json.loads(st.secrets['IS_CLOUD_DEPLOYMENT']) and VECTOR_STORE == 'Local':
            # construct an index over these documents... saved in memory
            index = VectorStoreIndex.from_documents(documents, show_progress=True, service_context=service_context)
            # save index on disk
            index.storage_context.persist(persist_dir='./storage')

        # WEAVIATE CLOUD STORE
        elif VECTOR_STORE == 'Weaviate':
            wc.schema.delete_class("Documents")
            class_obj = {
                "class": "Documents",
                "vectorizer": "text2vec-openai",
                "moduleConfig": {
                    "text2vec-openai": {},
                    "generative-openai": {}
                }
            }
            wc.schema.create_class(class_obj)
            # chunk up the documents into nodes 
            parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            # construct vector store
            vector_store = WeaviateVectorStore(weaviate_client=wc, index_name="Documents", text_key="content")
            # setting up the storage for the embeddings
            storage_context = StorageContext.from_defaults(vector_store = vector_store)
            # set up the index
            index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True, service_context=service_context)

        else:
            raise ValueError(f'Unknown vector store {VECTOR_STORE}')

        print('---- Document Q&A  ----', '\n',
              'Indexing Embedding Tokens: ', token_counter.total_embedding_token_count, '\n')

    with st.sidebar:
        st.markdown(f'#### {title} Settings')
        st.selectbox(
            'OpenAI model', options=OPENAI_MODELS_COMPLETIONS,
            on_change=_set_state_cb, kwargs={'completions_model': 'selectbox_docs_completions_model_name'},
            index=OPENAI_MODELS_COMPLETIONS.index(state.completions_model),
            help='Allowed models. Accuracy, speed, token consumption and costs will vary.',
            key='selectbox_docs_completions_model_name'
        )
        include_history = st.checkbox('Include history in prompts', value=False)
        if st.button('Clear history'):
            state.questions = []
            state.past = []
        # NOTE: Hide indexing button if cloud deployment (temporary fix for public demo)
        if not json.loads(st.secrets['IS_CLOUD_DEPLOYMENT']) and st.button('Index documents'):
            with st.spinner("Indexing..."):
                _index_documents()

    # GPT completion models can not handle web sites, so we scrape the URL in the user input
    user_input = state.user_input
    if user_input.strip().startswith('http'):
        scraped_texts = scrape_articles([user_input])['text']
        user_input = scraped_texts[0] if scraped_texts else user_input
        user_input = user_input.replace('\n', ' ').replace('\r', '') if user_input else user_input

    if include_history:
        context = '\n\n'.join([f'| Question: "{q}" | Answer: "{a}" |' for q, a in zip(state.questions, state.past)])
        refinement = \
            'Finally, return results in markdown text, include bullet point format where appropriate. ' + \
            'Add additional web links at the end of the response if this is useful.'
        prompt_template = "Given this context ### {context} ###. Answer or summarize this: ### {doc_query} ###. {refinement}"
        prompt = PromptTemplate(input_variables=['context', 'doc_query', 'refinement'], template=prompt_template)
        query_prompt = prompt.format(context=context, doc_query=user_input, refinement=refinement)
    else:
        refinement = \
            'Return results in markdown text, include bullet point format where appropriate. ' + \
            'Add additional web links at the end of the response if this is useful.'
        prompt_template = "Answer or summarize this: ### {doc_query} ###. {refinement}"
        prompt = PromptTemplate(input_variables=['doc_query', 'refinement'], template=prompt_template)
        query_prompt = prompt.format(doc_query=user_input, refinement=refinement)

    if user_input_confirmed and state.user_input:
        with st.spinner("Generating query answer..."):
            try:
                response = get_llm_doc_query_response(query_prompt, model_name=state.completions_model, _service_context=service_context)
                print('---- Document Q&A  ----', '\n',
                      'Embedding Tokens: ', token_counter.total_embedding_token_count, '\n',
                      'LLM Prompt Tokens: ', token_counter.prompt_llm_token_count, '\n',
                      'LLM Completion Tokens: ', token_counter.completion_llm_token_count, '\n',
                      'Total LLM Token Count: ', token_counter.total_llm_token_count)
            except Exception as ex:
                st.warning(f'Index does not exist. Please index some documents.')
                st.error(str(ex))
                return

    if state.user_input:
        st.subheader('🙋🏽 Answer')
        with st.spinner("Generating query answer..."):
            try:
                # This will use cached response!
                response = get_llm_doc_query_response(query_prompt, model_name=state.completions_model, _service_context=service_context)
            except Exception as ex:
                st.warning(f'Index does not exist. Please index some documents.')
                st.error(str(ex))
                return
            
        if state.user_input not in state.questions:
            state.questions.append(state.user_input)
            state.generated.append((state.user_input, response))
            state.past.append(response)

        st.markdown(response)

        with st.expander('View conversation history', expanded=False):
            st.markdown('\n\n'.join([f'---\n**Question**\n\n{q}\n\n**Answer**\n\n{a}' for q, a in zip(state.questions, state.past)]))
            
        estimated_cost = ((token_counter.prompt_llm_token_count / 1000.0) * LANG_MODEL_PRICING[state.completions_model]['input']) + \
            ((token_counter.completion_llm_token_count / 1000.0) * LANG_MODEL_PRICING[state.completions_model]['output'])
        print('Document Q&A Estimated Cost: $', estimated_cost)
        state.estimated_cost_doc = estimated_cost
        state.cumulative_cost += estimated_cost

        return response
