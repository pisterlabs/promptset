CODER_TEMPLATE = """
You are a Python programmer. Your task is to take the user's prompt and code the response in a single python script in markdown format.
Do NOT assume any local files exist such as images or other files.

Examples:

User: Write me a hello world script.
Response:
```python
print("Hello World!")
```


User: <!prompt>
Response:
"""

from langchain.chat_models import ChatOpenAI


class ProgrammerAgent:
    def __init__(self):
        self.template = CODER_TEMPLATE
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4")

    def get_prompt_template(self, prompt):
        template = CODER_TEMPLATE
        template = template.replace("<!prompt>", prompt)
        return template

    def prompt(self, prompt):
        prompt = self.get_prompt_template(prompt=prompt)
        res = self.llm.call_as_llm(prompt)
        # follow_up_prompt = res + "\n"
        # "The following is a python script that will run in a python interpreter. Check it for errors that may occur and respond with the cleaned up script." + "\n" + "Response:"
        # res = self.llm.call_as_llm(follow_up_prompt)
        return res


if __name__ == "__main__":
    programmer_agent = ProgrammerAgent()
    res = programmer_agent.prompt(
        """
Write me a hello world script.
"""
    )
    print(res)
