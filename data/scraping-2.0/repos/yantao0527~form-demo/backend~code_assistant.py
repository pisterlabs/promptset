from langchain.agents.openai_assistant import OpenAIAssistantRunnable


class CodeInterpreterAssistant:
    def __init__(self):
        self.assistant = OpenAIAssistantRunnable.create_assistant(
            name="langchain assistant",
            instructions="You are a personal tutor. Write and run code to answer questions.",
            tools=[{"type": "code_interpreter"}],
            model="gpt-4-1106-preview",
        )

    def ask__question(self, question):
        return self.assistant.invoke({"content": question})
