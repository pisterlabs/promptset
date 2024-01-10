import os
from llama_index import LLMPredictor, StorageContext, load_index_from_storage, LangchainEmbedding, ServiceContext
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
import langchain
import gradio as gr

#debug
langchain.debug=False

#Based on your settings, see version, base, key in your Azure AI portal
api_type = "azure"
api_version = "2023-03-15-preview"
api_base = os.getenv("AZURE_API_BASE")
api_key = os.getenv("AZURE_API_KEY")
chat_deployment = "gpt35"
embedding_deployment= "text-embedding-ada-002"

# Chat model
llm = AzureChatOpenAI(deployment_name=chat_deployment, openai_api_base=api_base, openai_api_key=api_key, openai_api_type=api_type, openai_api_version=api_version)
llm_predictor = LLMPredictor(llm=llm)

# Embedding model
embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model=embedding_deployment,
        deployment=embedding_deployment,
        openai_api_key=api_key,
        openai_api_base=api_base,
        openai_api_type=api_type,
        openai_api_version=api_version,
    ),
    embed_batch_size=1
)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embedding_llm)

#load local index
#load local index
storage_context = StorageContext.from_defaults(persist_dir="local-index-azure")
index = load_index_from_storage(storage_context, service_context=service_context)

# define custom retrievers
query_engine = index.as_query_engine(service_context=service_context)

# tool config
tool_config = IndexToolConfig(
    query_engine=query_engine,
    name="VMware Index",
    description="Documents about VMware cloud services and cloud partner navigator",
    tool_kwargs={"return_direct": True}
)

toolkit = LlamaToolkit(
    index_configs=[tool_config]
)

memory = ConversationBufferMemory(memory_key="chat_history")
agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
    verbose=True
)

#If you want command line remove Gradio below and uncomment this
""" while True:
    text_input = input("User: ")
    response = agent_chain.run(input=text_input)
    print(f'Agent: {response}')
 """
#Gradio chatbot
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    prompt = gr.Textbox()

    def respond(message, chat_history):
        response = agent_chain.run(input=message)
        chat_history.append((message, response))
        return "", chat_history

    prompt.submit(respond, [prompt, chatbot], [prompt, chatbot])

demo.launch()
