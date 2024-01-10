from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

template = _DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
                                    Use the following format:
    
                                    Question: "Question here"
                                    Answer: "SQL Query to run"
    
                                    Only use the following tables:
    
                                    {table_info}
    
                                    If someone asks for the table foobar, they really mean the employee table.
    
                                    Question: {input}"""
prompt= PromptTemplate(input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE)


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="./models/sqlcoder-7b/ggml-sqlcoder-7b-q4_k.gguf.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,
    prompt = prompt,
    callback_manager=callback_manager,
    verbose=True  # Verbose is required to pass to the callback manager
    )

# llm_chain = LLMChain(prompt=prompt,llm=llm)
# question = "How many rows are there in the actor table"
# print(llm_chain.run(question))

result = llm("How many actors were born between the years 2001 and 2005 in the actor table?")
print(result)