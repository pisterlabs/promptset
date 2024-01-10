import langchain
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

langchain.verbose = True

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = """\
次のコマンドの概要を説明してください。
コマンド: {command}
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(template)

chat_message_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_message_prompt)
result = chain.run("echo")
print(result)