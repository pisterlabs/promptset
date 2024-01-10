from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

chat = ChatOpenAI(
    model="gpt-3.5-turbo",
)

write_article_chain = LLMChain(
    llm=chat,
    prompt=PromptTemplate(
        template="{input}について記事を書いてください。",
        input_variables=["input"],
    ),
    verbose=True
)

translate_chain = LLMChain(
    llm=chat,
    prompt=PromptTemplate(
        template="以下の文章を英語に翻訳してください。\n{input}",
        input_variables=["input"],
    ),
    verbose=True
)

sequential_chain = SimpleSequentialChain(
    chains=[
        write_article_chain,
        translate_chain,
    ],
    verbose=True,
)

result = sequential_chain.run("ノートパソコンの選び方")

print(result)