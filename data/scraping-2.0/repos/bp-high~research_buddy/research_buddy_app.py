from llama_index import Document
from llama_index.chat_engine import CondenseQuestionChatEngine
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.node_parser import SimpleNodeParser
from llama_index import LangchainEmbedding, ServiceContext
from llama_index import VectorStoreIndex
from llama_index import StorageContext, load_index_from_storage
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import TreeSummarize,get_response_synthesizer
from llama_index.llms import ChatMessage

from langchain.llms import Clarifai
from langchain.embeddings import ClarifaiEmbeddings


from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2


import uuid

import streamlit as st

import modal

CLARIFAI_PAT = st.secrets.CLARIFAI_PAT
MODERATION_THRESHOLD = st.secrets.MODERATION_THRESHOLD
st.set_page_config(page_title="Research Buddy: Insights and Q&A on AI Research Papers using GPT and Nougat", page_icon="🧐", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title(body="AI Research Buddy: Nougat + GPT Powered Paper Insights 📚🤖")
st.info("""This Application currently only works with arxiv and acl anthology web links which belong to the format:- 
1) Arxiv:- https://arxiv.org/abs/paper_unique_identifier
2) ACL Anthology:- https://aclanthology.org/paper_unique_identifier/ 

This Application uses the recently released Meta Nougat Visual Transformer for processing Papers""", icon="ℹ️")
user_input = st.text_input("Enter the arxiv or acl anthology url of the paper", "https://aclanthology.org/2023.semeval-1.266/")


def initialize_session_state():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about the research paper"}
        ]

    if "paper_content" not in st.session_state:
        st.session_state.paper_content = None

    if "paper_insights" not in st.session_state:
        st.session_state.paper_insights = None


initialize_session_state()

st.write("This App has been currently disabled as I am out of the credits for LLM model vendors. Also I am working out on a better way to extract insights from research papers and Scientific Q&A, will restart this in some time.")
# Uncomment the below code if you are trying to build something similar to my app
# def get_paper_content(url: str) -> str:
#     with st.spinner(text="Using Nougat(https://facebookresearch.github.io/nougat/) to read the paper contents and get the markdown representation of the paper"):
#         f = modal.Function.lookup("streamlit-hack", "main")
#         output = f.call(url)
#         st.session_state.paper_content = output
#         return output


# def index_paper_content(content: str):
#     with st.spinner(text="Indexing the paper – hang tight! This should take 3-5 minutes"):
#         try:
#             LLM_USER_ID = 'openai'
#             LLM_APP_ID = 'chat-completion'
#             # Change these to whatever model and text URL you want to use
#             LLM_MODEL_ID = 'GPT-3_5-turbo'
#             llm = Clarifai(pat=CLARIFAI_PAT, user_id=LLM_USER_ID, app_id=LLM_APP_ID, model_id=LLM_MODEL_ID)

#             documents = [Document(text=content)]
#             parser = SimpleNodeParser.from_defaults()

#             nodes = parser.get_nodes_from_documents(documents)
#             USER_ID = 'openai'
#             APP_ID = 'embed'
#             # Change these to whatever model and text URL you want to use
#             MODEL_ID = 'text-embedding-ada'
#             embeddings = ClarifaiEmbeddings(pat=CLARIFAI_PAT, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)
#             embed_model = LangchainEmbedding(embeddings)
#             service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
#             index = VectorStoreIndex(nodes, service_context=service_context)
#             persist_dir = uuid.uuid4().hex
#             st.session_state.vector_store = persist_dir
#             index.storage_context.persist(persist_dir=persist_dir)
#             return "Paper has been Indexed"

#         except Exception as e:
#             print(str(e))
#             return "Unable to Index the Research Paper"


# def generate_insights():
#     with st.spinner(text="Generating insights on the paper and preparing the Chatbot"):
#         try:
#             LLM_USER_ID = 'openai'
#             LLM_APP_ID = 'chat-completion'
#             # Change these to whatever model and text URL you want to use
#             LLM_MODEL_ID = 'GPT-3_5-turbo'
#             llm = Clarifai(pat=CLARIFAI_PAT, user_id=LLM_USER_ID, app_id=LLM_APP_ID, model_id=LLM_MODEL_ID)

#             USER_ID = 'openai'
#             APP_ID = 'embed'
#             # Change these to whatever model and text URL you want to use
#             MODEL_ID = 'text-embedding-ada'
#             embeddings = ClarifaiEmbeddings(pat=CLARIFAI_PAT, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)
#             embed_model = LangchainEmbedding(embeddings)

#             service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

#             index = load_index_from_storage(
#                 StorageContext.from_defaults(persist_dir=st.session_state.vector_store),
#                 service_context=service_context
#             )

#             retriever = VectorIndexRetriever(
#                 index=index,
#                 similarity_top_k=2,
#             )
#             # configure response synthesizer
#             response_synthesizer = get_response_synthesizer(
#                 response_mode="simple_summarize", service_context=service_context
#             )

#             # assemble query engine
#             query_engine = RetrieverQueryEngine(
#                 retriever=retriever,
#                 response_synthesizer=response_synthesizer,
#             )

#             response_key_insights = query_engine.query("Generate core crux insights, contributions and results of the paper as Key Topics and thier content in markdown format where each Key Topic is in bold and has a markdown heading format followed by its content in bullets list form")

#         except Exception as e:
#             print(str(e))
#             response_key_insights = "Error While Generating Insights"

#         st.session_state.paper_insights = response_key_insights.response


# if st.button("Read and Index Paper"):
#     paper_content = get_paper_content(url=user_input)

#     if st.session_state.paper_content is not None:
#         with st.expander("See Paper Contents"):
#             st.markdown(paper_content)

#         result = index_paper_content(content=paper_content)
#         st.write(result)
#         generate_insights()


# if st.session_state.paper_content is not None:
#     with st.expander("See Paper Contents"):
#         st.markdown(st.session_state.paper_content)

# if st.session_state.paper_insights is not None:
#     st.sidebar.title("# 🚀 Illuminating Research Insights 📜💡")
#     st.sidebar.write(st.session_state.paper_insights)


# def reset_conversation():
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Ask me a question about the research paper"}
#     ]


# def moderate_text(text: str) -> tuple:
#     MODERATION_USER_ID = 'clarifai'
#     MODERATION_APP_ID = 'main'
#     # Change these to whatever model and text URL you want to use
#     MODERATION_MODEL_ID = 'moderation-multilingual-text-classification'
#     MODERATION_MODEL_VERSION_ID = '79c2248564b0465bb96265e0c239352b'

#     channel = ClarifaiChannel.get_grpc_channel()
#     stub = service_pb2_grpc.V2Stub(channel)

#     metadata = (('authorization', 'Key ' + CLARIFAI_PAT),)

#     userDataObject = resources_pb2.UserAppIDSet(user_id=MODERATION_USER_ID, app_id=MODERATION_APP_ID)

#     # To use a local text file, uncomment the following lines
#     # with open(TEXT_FILE_LOCATION, "rb") as f:
#     #    file_bytes = f.read()

#     post_model_outputs_response = stub.PostModelOutputs(
#         service_pb2.PostModelOutputsRequest(
#             user_app_id=userDataObject,
#             # The userDataObject is created in the overview and is required when using a PAT
#             model_id=MODERATION_MODEL_ID,
#             version_id=MODERATION_MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
#             inputs=[
#                 resources_pb2.Input(
#                     data=resources_pb2.Data(
#                         text=resources_pb2.Text(
#                             raw=text
#                         )
#                     )
#                 )
#             ]
#         ),
#         metadata=metadata
#     )
#     if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
#         print(post_model_outputs_response.status)
#         raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

#     # Since we have one input, one output will exist here
#     output = post_model_outputs_response.outputs[0]
#     moderation_reasons = ""
#     intervention_required = False
#     for concept in output.data.concepts:
#         if concept.value > MODERATION_THRESHOLD:
#             moderation_reasons += concept.name + ","
#             intervention_required = True

#     return moderation_reasons, intervention_required


# if st.session_state.vector_store is not None:
#     LLM_USER_ID = 'openai'
#     LLM_APP_ID = 'chat-completion'
#     # Change these to whatever model and text URL you want to use
#     LLM_MODEL_ID = 'GPT-3_5-turbo'
#     llm = Clarifai(pat=CLARIFAI_PAT, user_id=LLM_USER_ID, app_id=LLM_APP_ID, model_id=LLM_MODEL_ID)

#     USER_ID = 'openai'
#     APP_ID = 'embed'
#     # Change these to whatever model and text URL you want to use
#     MODEL_ID = 'text-embedding-ada'
#     embeddings = ClarifaiEmbeddings(pat=CLARIFAI_PAT, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)
#     embed_model = LangchainEmbedding(embeddings)

#     service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

#     index = load_index_from_storage(
#         StorageContext.from_defaults(persist_dir=st.session_state.vector_store),
#         service_context=service_context
#     )

#     retriever = VectorIndexRetriever(
#         index=index,
#         similarity_top_k=2,
#     )
#     # configure response synthesizer
#     response_synthesizer = get_response_synthesizer(
#         response_mode="simple_summarize", service_context=service_context
#     )

#     # assemble query engine
#     query_engine = RetrieverQueryEngine(
#         retriever=retriever,
#         response_synthesizer=response_synthesizer,
#     )

#     custom_chat_history = []
#     for message in st.session_state.messages:
#         custom_message = ChatMessage(role=message["role"], content=message["content"])
#         custom_chat_history.append(custom_message)

#     chat_engine = CondenseQuestionChatEngine.from_defaults(service_context=service_context, query_engine=query_engine,
#                                                            verbose=True,
#                                                            chat_history=custom_chat_history)

#     if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})

#     st.button('Reset Chat', on_click=reset_conversation)

#     for message in st.session_state.messages:  # Display the prior chat messages
#         with st.chat_message(message["role"]):
#             st.write(message["content"])

#     # If last message is not from assistant, generate a new response
#     if st.session_state.messages[-1]["role"] != "assistant":
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     reason, intervene = moderate_text(prompt)
#                 except Exception as e:
#                     print(str(e))
#                     reason = ''
#                     intervene = False
#                 if not intervene:
#                     response = chat_engine.chat(prompt)
#                     st.write(response.response)
#                     message = {"role": "assistant", "content": response.response}
#                     st.session_state.messages.append(message)  # Add response to message history
#                 else:
#                     response = f"This query cannot be processed as it has been detected to be {reason}"
#                     st.write(response)
#                     message = {"role": "assistant", "content": response}
#                     st.session_state.messages.append(message)
