#from gpt4all import GPT4All
#model = GPT4All("/home/superadmin/Workspace/Reference/gpt4all-client/models/GPT4All-13B-snoozy.ggmlv3.q4_0.bin")
#output = model.generate("The capital of France is ", max_tokens=3)
#print(output)


from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
local_path = ("/home/superadmin/Workspace/Reference/gpt4all-client/models/GPT4All-13B-snoozy.ggmlv3.q4_0.bin")  # replace with your desired local file path

# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]

# Verbose is required to pass to the callback manager
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.run(question)