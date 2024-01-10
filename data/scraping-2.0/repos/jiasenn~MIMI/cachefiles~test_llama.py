from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage
from llama_index.callbacks import CallbackManager, LlamaDebugHandler


template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
# Load the pre-trained LlamaIndex model and initialize the pipeline
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/Users/jiasenn/Downloads/llama-2-13b-chat.Q5_K_M.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
storage_directory = "./storage"

documents = SimpleDirectoryReader('./data').load_data()


service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1024,
                                               embed_model="local",
                                               callback_manager=callback_manager)

storage_context = StorageContext.from_defaults(persist_dir=storage_directory)


index = VectorStoreIndex.from_documents(documents, service_context=service_context)

index.storage_context.persist(persist_dir=storage_directory)
query_engine = index.as_query_engine(service_context=service_context,
                                        similarity_top_k=3)

response = query_engine.query("what does the coolies do?")

# prompt = """
# Question: A rap battle between Stephen Colbert and John Oliver
# """
# llm(prompt)