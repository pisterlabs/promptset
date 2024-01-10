import os
import langchain
from langchain import (
    agents,
    prompts,
    chains,
    llms
)


class BOAgent:
    def __init__(
            self,
            tools,
            memory,
            model="text-davinci-003",
            temp=0.1,
            max_steps=30,
    ):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.memory = memory

        # Initialize LLM
        if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
            self.llm = langchain.chat_models.ChatOpenAI(
                temperature=temp,
                openai_api_key=self.openai_key,
                model_name=model,
            )
        else:
            self.llm = langchain.OpenAI(
                temperature=temp,
                openai_api_key=self.openai_key,
                model_name=model
            )

        # Initialize agent
        self.agent = agents.initialize_agent(
            tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            max_iterations=max_steps,
            memory=self.memory
        )

    def run(self, prompt):
        return self.agent.run(prompt)
