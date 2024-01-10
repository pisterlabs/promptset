from langchain import OpenAI, ConversationChain, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import GPT4All

from helper_dude.prompts.chat import chat_prompt
from helper_dude.ui.term import TerminalInterface

user_interface = TerminalInterface()

# Instantiate the model. Callbacks support token-wise streaming
model = GPT4All(model="./models/gpt4all-model.bin", n_ctx=512, n_threads=8, callback_manager=user_interface.callback_manager)

chat_chain = LLMChain(
    llm=model, 
    prompt=chat_prompt, 
    verbose=True, 
    memory=ConversationBufferWindowMemory(k=2)
)

# Start the conversation
response = chat_chain.predict(human_input=user_interface.prompt())
print()
print()
print()
print(response)
