from langchain.llms import HuggingFacePipeline
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = HuggingFacePipeline.from_model_id(
  model_id="togethercomputer/LLaMA-2-7B-32K",
  # model_id="bigscience/bloom-1b7",
  task="text-generation",
  # callback_manager=callback_manager,
  callbacks=[StreamingStdOutCallbackHandler()],
  model_kwargs={"temperature": 0.9, "max_length": 256},
)

while True:
  inp = input("> ")
  print(f"INPUT:\n{inp}\n")
  print(llm(inp))