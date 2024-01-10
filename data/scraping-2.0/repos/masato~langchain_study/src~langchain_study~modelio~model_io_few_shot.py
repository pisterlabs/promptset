from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.llms import OpenAI

examples = [
    {
        "input": "LangChainはChatGPT・Large Language Model (LLM) の実利用をより柔軟に簡易に行うためのツール群です",  # noqa: E501
        "output": "LangChainは、ChatGPT・Large Language Model (LLM) の実利用をより柔軟に、簡易に行うためのツール群です。",  # noqa: E501
    },
]

prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="入力: {input}\n出力: {output}\n",
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=prompt,
    prefix="以下の句読点の抜けた入力に句読点を追加してください。追加してよい句読点は「、」と「。」のみです。他の句読点は追加しないでください。",
    suffix="入力: {input_string}\n出力:",
    input_variables=["input_string"],
)

llm = OpenAI()
formatted_prompt = few_shot_prompt.format(
    input_string="私はさまざまな機能がモジュールとして提供されているLangChainを使ってアプリケーションを開発しています",
)

result = llm.predict(formatted_prompt)
print(f"formatted_prompt: {formatted_prompt}")
print(f"result: {result}")
