from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os
import sys
import dotenv

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

template = """
        あなたは物事に命名をするアシスタントです。指示された印象を与える名前を5つ考えてください。
        印象とは物事に対する形容詞です。例えば「かわいい」、「こわい」、「かっこいい」などが挙げられます。
        もし指示内に印象に関する指定が含まれなかった場合は、
        「印象に関しては特に指定がなかったため、かわいい名前を考えました。」と断りを入れた上で、かわいい名前を提示してください。
    """
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template = "{instruction}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

chain = LLMChain(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, streaming=True),
    prompt=chat_prompt,
)


try:
    print(chain.run(sys.argv[1]))
except IndexError:
    print("コマンドライン引数に指示を渡してください。(例: 犬の可愛い名前を考えてください。)")
