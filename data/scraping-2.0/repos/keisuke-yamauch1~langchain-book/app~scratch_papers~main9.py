import langchain
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

langchain.verbose = True
langchain.debug = True

if __name__ == "__main__":
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    cot_template = """以下の質問に回答してください。

質問：{question}

ステップバイステップで考えましょう。
"""

    cot_prompt = PromptTemplate(
        input_variables=["question"],
        template=cot_template,
    )

    cot_chain = LLMChain(llm=chat, prompt=cot_prompt)

    summarize_template = """以下の文章を結論だけ一言に要約してください

{input}    
"""

    summarize_prompt = PromptTemplate(
        input_variables=["input"],
        template=summarize_template,
    )

    summarize_chain = LLMChain(llm=chat, prompt=summarize_prompt)

    cot_summarize_chain = SimpleSequentialChain(chains=[cot_chain, summarize_chain])

    result = cot_summarize_chain(
        "私は市場に行って10個のリンゴを買いました。隣人に2つ、修理工に2つ渡しました。それから5つのリンゴを買って1つ食べました。残りは何個ですか？"
    )

    print(result["output"])

