from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(model="llama2",
            #  callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
             temperature=0.9
             )

from langchain.prompts import PromptTemplate

prompt=PromptTemplate(
    input_variables=["topic"],
    template="Bríndame 5 películas del género {topic}"
)

from langchain.chains import LLMChain
chain= LLMChain(llm=llm,
                prompt=prompt,
                verbose=False)

print(chain.run("Acción"))

llm("Puedes recomendarme 5 películas de terror?")