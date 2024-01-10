"""
instructions:
- download ollama `https://ollama.ai/download`
    - ollama pull llama2
- pip install langchain
"""


from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = Ollama(model='llama2',
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# llm2 = Ollama(model='mistral',
#                 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

prompt = PromptTemplate(
    input_variables=["sentence"],
    template= """Given this sentence: '{sentence}', classify if it is Singlish (0) or English (1),
                    strictly only output the JSON object with the following keys:
                        'sentence': (str) {sentence}
                        'label': (int) 0 or 1 
                        'explanation': (str) reason for classification""",
)
chain = LLMChain(llm=llm, 
                 prompt=prompt,
                 verbose=False)

# Run the chain only specifying the input variable.
sentence1 = "I already have a host and blogger"
sentence2 = "Meet after lunch la..."
print(chain.run(sentence1))
print(chain.run(sentence2))





