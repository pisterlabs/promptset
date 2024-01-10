from dotenv import load_dotenv

from langchain.agents import Tool, AgentExecutor
from langchain.agents.openai_assistant import OpenAIAssistantRunnable
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.utilities import SerpAPIWrapper

from SparksAI.config import MODEL_NAME, CONVERSATION_ANALYST_ID

load_dotenv()


class Swarm:
    def __init__(self) -> None:
        self.conversation_swarm = {}
        self.archivist_swarm = {}
        self.analyst_swarm = self.init_analyst_agent()

        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            streaming=True,
            callbacks=[FinalStreamingStdOutCallbackHandler()],
        )

    def get_conversation_agent(self, username: str) -> AgentExecutor:
        if username in self.conversation_swarm:
            return self.conversation_swarm[username]

        else:
            self.init_conversation_agent(username)

            return self.conversation_swarm[username]

    def init_conversation_agent(self, username: str) -> None:
        if username in self.conversation_swarm:
            return None

        prompt = PromptTemplate.from_template(
            """You are an expert conversationalist tasked with crafting a response to a specific question.
            An analyst has already reviewed the question and supplied guidance along with additional information to assist you.
            Furthermore, you have access to context from prior interactions with the user, ensuring your response is well-informed and tailored to the user's needs and history of inquiries.
            Use all information provided when crafting a response.
            Finally, you should write the response from the perspective of the below persona.
            

            Persona:
            You are Tav, a knowledgeable and friendly virtual assistant with a background in a wide range of topics, from science and technology to arts and history.
            You are known for your engaging conversation style, blending informative content with a touch of humor and personal anecdotes.
            Your responses are not only factual but also considerate of the user's level of understanding and interest in the subject.
            You have a knack for making complex subjects accessible and enjoyable to learn about.
            Tav is patient, always willing to clarify doubts, and enjoys exploring topics in depth when the user shows interest.
            Your tone is consistently warm and inviting, making users feel comfortable and encouraged to ask more questions.
            As Tav, you aim to provide a pleasant and educational experience in every interaction. 


            Analyst Review:
            {analyst_message}
            
            
            Summary of prior interactions:
            {prior_messages}


            Question:
            {input_message}
            """
        )

        convo_agent = prompt | self.llm

        self.conversation_swarm[username] = convo_agent

    def get_analyst_agent(self) -> AgentExecutor:
        return self.analyst_swarm

    def init_analyst_agent(self) -> AgentExecutor:
        search = SerpAPIWrapper()

        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="Useful when you need to answer questions about current events. You should ask targeted questions.",
            ),
        ]

        ai_assistant = OpenAIAssistantRunnable(
            assistant_id=CONVERSATION_ANALYST_ID,
            as_agent=True,
        )

        return AgentExecutor(agent=ai_assistant, tools=tools)

    def get_archivist(self, username: str) -> AgentExecutor:
        if username in self.archivist_swarm:
            return self.archivist_swarm[username]

        else:
            self.init_archivist(username)

            return self.archivist_swarm[username]

    def init_archivist(self, username: str) -> None:
        if username in self.archivist_swarm:
            return None

        prompt = PromptTemplate.from_template(
            """You are an archivist.
            You have been given the transcript of a conversation between an AI and a human user.
            You have also received the most recent message from the human.
            Your job is to provide a list of at least three bullet points summarizing the transcript.
            The list should contain the most relevant material to the most recent message.

            Transcript:
            {memory}

            Most recent message:
            {input_message}
            """
        )

        archivist = prompt | self.llm

        self.archivist_swarm[username] = archivist
