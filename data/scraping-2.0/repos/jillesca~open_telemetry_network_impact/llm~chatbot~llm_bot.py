import os
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory


LLM_API_KEY = os.environ["LLM_API_KEY"]
SYSTEM_PROMPT = """
    You are chatbot who receives alerts based on telemetry data from network devices. 
    The information you are receiving is from a system using json format. 
    If you receive information that is not as json, this comes from a human. 
    Use the information that is inside the commonLabels tag. 
    Your job is to make sense of the alert received and process for an engineer. 
    Provide as much help as possible, but don't invent. reply in markdown format.
    Provide a table with the device details you recevied with the device name, ip and interface affected.
    If you receive new information review if the new events could be related and provide a new update. 
    Avoid repeating similar information.
    And tell a for joke for network engineers, to take out the stress of the situation.
    Give one for every request ask.
    """


class LLM_Chatbot:
    def __init__(self) -> None:
        self._start_llm()

    def _start_llm(self):
        llm = ChatOpenAI(openai_api_key=LLM_API_KEY, temperature=0.5)
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    " ".join(SYSTEM_PROMPT.split())
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        memory = ConversationSummaryBufferMemory(
            llm=llm, max_token_limit=3000, return_messages=True
        )
        self.conversation = ConversationChain(
            memory=memory,
            prompt=prompt,
            llm=llm,
        )

    def chat(self, data: str) -> str:
        answer = self.conversation({"input": data})
        return self.get_latest_ai_message(answer)

    def get_latest_ai_message(self, answer: list) -> str:
        return answer.get("history")[-1].content


if __name__ == "__main__":
    chatbot = LLM_Chatbot()
    joke = chatbot.chat("tell me a joke about developers")
    print(joke)
    print("#" * 80, "\n")
    joke = chatbot.chat("now about lawyers")
    print(joke)
    print("#" * 80, "\n")
    joke = chatbot.chat("now make a one, but consider the previous answer")
    print(joke)
    print("#" * 80, "\n")
    joke = chatbot.chat("can you do a summary of what we talked about?")
    print(joke)
    print("#" * 80, "\n")
