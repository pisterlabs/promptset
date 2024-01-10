from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# i dont know why but the callback_manager is not working
# i dont know to use it
callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler()])

prompt = PromptTemplate.from_file("prompt_vb6_select_clause4.txt")

print(f"prompt: {prompt.format()}")

# you must run "python ../../../llama.cpp/llamaapp.cpp/examples/server/api_like_OAI.py"
# that server its http://localhost:8081
# the openai_api_key is not used in this case because we are only using the openai framework
uri = "http://localhost:8081"
llm = OpenAI(openai_api_base=uri,
            openai_api_key="YOUR_API_KEY",
            streaming=True,
            callback_manager=callback_manager,
            verbose=False)
print(llm(prompt=prompt.format()))