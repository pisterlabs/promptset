from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.domain.models.message import Message, Role
from app.adapters.repositories.transcript_line_repository import TranscriptLinesRepository

# TODO: Template Router

class ChatAboutEpisodeService:
    def __init__(self):
        self.repository = TranscriptLinesRepository()
        self.llm = ChatOpenAI(temperature=0)
        self.default_template =  """Answer the question based only on the following context:
        {context}

        Question: {question}
        """


    def get_answer(self, episode_id: str, message: Message) -> Message:
        transcript_lines = self.repository.find_by_episode_id(episode_id,1000)
        context = "\n".join([line.text for line in transcript_lines])
        prompt = ChatPromptTemplate.from_template(self.default_template)
        chain = prompt | self.llm
        ai_message=chain.invoke({"context": context, "question": message.content})

        return Message(
            content = ai_message.content,
            role = Role.AI
        )