import typing as t

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.globals import set_debug
import os

set_debug(True)


class Reviewer:
    analyze_prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the following code"),
        ("human", "{code_input}")
    ])

    def __init__(self, api_key: t.Optional[str] = None, model: t.Optional[str] = None):
        api_key = os.getenv("OPENAI_API_KEY", api_key)
        model = model or "gpt-4-1106-preview"
        self.llm = ChatOpenAI(openai_api_key=api_key, model=model)

    def review(self, code: str):
        chain = self.analyze_prompt | self.llm
        response = chain.invoke({"code_input": code})
        print(response)

    def review_batch(self, paths: t.List[str]) -> t.List[t.Dict[str, t.Any]]:
        """

        Args:
            paths ():

        Returns:

        """
        input_dict = {}
        input_list = []
        for path in paths:
            with open(path, "r") as f:
                code = f.read()
            input_dict[path] = code
            input_list.append({"code_input": code})

        chain = self.analyze_prompt | self.llm
        output_list = []
        responses = chain.batch(input_list)
        for path, response in zip(paths, responses):
            output_list.append({"path": path, "review": response.content})
        return output_list


