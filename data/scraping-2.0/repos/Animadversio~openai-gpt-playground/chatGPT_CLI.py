# https://python.langchain.com/en/latest/modules/agents/agent_executors/examples/chatgpt_clone.html
import questionary

from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "human_input"],
    template=template
)


chatgpt_chain = LLMChain(
    llm=ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo"),#OpenAI(temperature=0),
    prompt=prompt,
    verbose=False,
    memory=ConversationBufferWindowMemory(k=2),
)

while True:
    # query = "For robotics purpose, which algorithm did they used, PPO, Q-learning, etc.?"
    query = questionary.text("Human: ", multiline=True).ask()
    if query == "" or query is None:
        is_exit = questionary.confirm("Exit?").ask()
        if is_exit:
            break
        else:
            continue
    output = chatgpt_chain.predict(human_input=query)
    print(output)

