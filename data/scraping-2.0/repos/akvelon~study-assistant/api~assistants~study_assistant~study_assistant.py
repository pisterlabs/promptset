"""Study assistant"""
import time
import io
from pydantic import BaseSettings
import openai
from settings import settings
from api.endpoints.search import search_engine
from api.endpoints.user import User
from api.endpoints.schemas import (
    MessageAttachment,
    Message,
    MessagesRequest,
    MessagesResponse,
)
from api.assistants.history.manager import HistoryManager
from api.db.schools import schools_db

history_manager = HistoryManager()

openai.api_key = settings.openai_key


def parse_prompt(file: str) -> str:
    """Loads prompts for Chat"""
    with open(file, "r", encoding="utf-8") as promptfile:
        prompt = promptfile.read()
    return prompt


class StudyAssistantSettings(BaseSettings):
    """read file and return as string"""

    prompt = parse_prompt("api/assistants/study_assistant/study_assistant.txt")

    model: str = "gpt-3.5-turbo"
    greeting: str = ""

    temperature: float = 0.8
    max_tokens: int = 400

    prompt: str = prompt

    name: str = "Study Assistant"
    description: str = ""
    category: str = "language"


class StudyAssistant(StudyAssistantSettings):
    """Study assistant class"""

    async def generate_response(
        self,
        request: MessagesRequest,
        user: User,
        school_id: int,
    ) -> MessagesResponse:
        """Generates response for answer"""
        # Places the system prompt at beginning of list
        messages = [{"role": "system", "content": self.prompt}]
        # Appends the rest of the conversation
        for message in request.messages:
            messages.append({"role": message.role, "content": message.content})
        documents = await search_engine.search_text_vectors(request, school_id)
        document = documents[0] if len(documents) else None
        if user:
            school_name = schools_db.get_school_by_id(user.schoolId)
            messages.append(
                {
                    "role": "system",
                    "content": f"""
                      For the response, take into account the following information:"
                      the user studies at the {school_name}
                    """,
                }
            )

        if document:
            messages.append(search_engine.get_system_message(document))

        # Generate response
        gpt_response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        attachments = []
        # Attach document to user message content
        if document:
            include_doc = search_engine.should_search_docs(
                gpt_response["choices"][0]["message"]["content"],
                0.5,
                school_id,
                [document.document],
            )
            if include_doc:
                image_src = (
                    document.document.image_metadata[0]["src"]
                    if document.document.image_metadata
                    else None
                )
                attachments.append(
                    MessageAttachment(
                        id=document.document.id,
                        title=document.document.title,
                        summary=document.document.summary,
                        url=document.document.url,
                        image=image_src,
                    )
                )

        # Convert to Message schema
        response_message = Message(
            id=gpt_response["id"],
            role=gpt_response["choices"][0]["message"]["role"],
            timestamp=time.time(),
            content=gpt_response["choices"][0]["message"]["content"],
            attachments=attachments,
        )

        return history_manager.process_messages(request, response_message, user)

    async def generate_response_audio(
        self, audio_file, chat_id: int | None, user: User | None, school_id: int
    ) -> MessagesResponse:
        """Generate response for audio question."""

        # Convert audio file to byte stream, due to unknown reasons
        # OpenAI does not accept SpooledTemporaryFile's
        buffer = io.BytesIO(audio_file.file.read())
        buffer.name = audio_file.filename
        # Transcribe audio byte stream to text
        transcript = openai.Audio.transcribe("whisper-1", buffer, response_format="text")

        # Start new conversation if no chat_id is present
        if chat_id is None:
            request = MessagesRequest(
                messages=[
                    Message(
                        id="1",
                        role="user",
                        timestamp=time.time(),
                        content=transcript,
                    )
                ]
            )
            return await self.generate_response(request, user, school_id)

        # Otherwise append transcription to existing chat history
        # Attempt to get chat history
        chat_history = history_manager.get_history(chat_id, user.id)

        request = MessagesRequest(
            chat=chat_history.chat,
            messages=chat_history.messages,
        )
        request.messages.append(
            Message(
                id=str(len(request.messages) + 1),
                role="user",
                timestamp=time.time(),
                content=transcript,
            )
        )

        return await self.generate_response(request, user, school_id)


class UsersMessageMissingException(Exception):
    """Missing exception class"""
