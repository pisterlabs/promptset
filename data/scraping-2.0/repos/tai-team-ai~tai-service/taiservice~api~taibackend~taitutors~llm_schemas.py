"""Define the llm schemas for interfacing with LLMs."""
import copy
from datetime import datetime
import re
from typing import Any, Callable, Optional, Type, Union
from enum import Enum
from uuid import UUID
from uuid import uuid4
from pydantic import Field, BaseModel, validator
import tiktoken
from langchain.schema import (
    AIMessage,
    FunctionMessage as langchainFunctionMessage,
    HumanMessage,
    SystemMessage as langchainSystemMessage,
    BaseMessage as langchainBaseMessage,
)
from langchain.chat_models.base import BaseChatModel
# first imports for local development, second imports for deployment
try:
    from ...routers.tai_schemas import ClassResourceSnippet, ClassResource
    from ..shared_schemas import BasePydanticModel
except (KeyError, ImportError):
    from routers.tai_schemas import ClassResourceSnippet, ClassResource
    from taibackend.shared_schemas import BasePydanticModel

class TaiTutorName(str, Enum):
    """Define the supported TAI tutors."""

    MILO = "Milo"
    DECLAN = "Declan"
    FINN = "Finn"
    ADA = "Ada"
    REMY = "Remy"
    KAI = "Kai"
    VIOLET = "Violet"

class ChatRole(str, Enum):
    """Define the built-in MongoDB roles."""

    TAI_TUTOR = "taiTutor"
    STUDENT = "student"
    FUNCTION = "function"

class ResponseTechnicalLevel(str, Enum):
    """Define the technical level of the response."""

    EXPLAIN_LIKE_IM_5 = "like5"
    EXPLAIN_LIKE_IM_IN_HIGH_SCHOOL = "likeHighSchool"
    EXPLAIN_LIKE_IM_IN_COLLEGE = "likeCollege"
    EXPLAIN_LIKE_IM_AN_EXPERT_IN_THE_FIELD = "likeExpertInTheField"


class ModelName(str, Enum):
    """Define the supported LLMs."""
    GPT_TURBO = "gpt-3.5-turbo"
    GPT_TURBO_LARGE_CONTEXT = "gpt-3.5-turbo-16k"
    GPT_4 = "gpt-4"


MODEL_TO_TOKEN_WINDOW_SIZE_MAPPING = {
    ModelName.GPT_TURBO: 4097,
    ModelName.GPT_TURBO_LARGE_CONTEXT: 16385,
    ModelName.GPT_4: 8192,
}


class BaseMessage(langchainBaseMessage):
    """Define the base message for the TAI tutor."""

    role: ChatRole = Field(
        ...,
        description="The role of the user that generated this message.",
    )
    render_chat: bool = Field(
        default=True,
        description="Whether or not to render the chat message. If false, the chat message will be hidden from the student.",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="The timestamp of the message.",
    )

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "base"

class SearchQuery(BaseMessage, HumanMessage):
    """Define the model for the search query."""
    role: ChatRole = Field(
        default=ChatRole.STUDENT,
        const=True,
        description="The role of the creator of the chat message.",
    )


class TutorAndStudentBaseMessage(BaseMessage):
    """Define the base message for the TAI tutor and student."""
    tai_tutor_name: TaiTutorName = Field(
        default=TaiTutorName.FINN,
        description="The name of the TAI tutor that generated this message.",
    )
    technical_level: ResponseTechnicalLevel = Field(
        default=ResponseTechnicalLevel.EXPLAIN_LIKE_IM_IN_HIGH_SCHOOL,
        description="The technical level of the response.",
    )


class StudentMessage(HumanMessage, TutorAndStudentBaseMessage):
    """Define the model for the student chat message."""

    role: ChatRole = Field(
        default=ChatRole.STUDENT,
        const=True,
        description="The role of the creator of the chat message.",
    )

class AIResponseCallingFunction(BaseModel):
    """Define the model for the AI response calling function."""

    name: str = Field(
        ...,
        description="The name of the function to call.",
    )
    arguments: dict[str, Any] = Field(
        ...,
        description="The arguments to pass to the function.",
    )

class TaiTutorMessage(AIMessage, TutorAndStudentBaseMessage):
    """Define the model for the TAI tutor chat message."""

    role: ChatRole = Field(
        default=ChatRole.TAI_TUTOR,
        const=True,
        description="The role of the creator of the chat message.",
    )
    class_resource_snippets: list[ClassResourceSnippet] = Field(
        default=[],
        description="The class resource chunks that were used to generate this message, if any.",
    )
    class_resources: list[ClassResource] = Field(
        default_factory=list,
        description="The class resources that were used to generate this message, if any.",
    )
    function_call: Optional[AIResponseCallingFunction] = Field(
        default=None,
        description="The function call that the assistant wants to make.",
    )

class SystemMessage(langchainSystemMessage, BaseMessage):
    """Define the model for the system chat message."""

    role: ChatRole = Field(
        default=ChatRole.TAI_TUTOR,
        const=True,
        description="The role of the creator of the chat message.",
    )
    render_chat: bool = Field(
        default=False,
        const=True,
        description="System messages are never rendered. Therefore this field is always false.",
    )

    @staticmethod
    def from_prompt(prompt: str) -> "SystemMessage":
        """Create a system message from a prompt."""
        return SystemMessage(
            content=prompt,
            role=ChatRole.TAI_TUTOR,
            render_chat=False,
        )

class FunctionMessage(langchainFunctionMessage, BaseMessage):
    """Define the model for the function chat message."""

    role: ChatRole = Field(
        default=ChatRole.FUNCTION,
        const=True,
        description="The role of the creator of the chat message.",
    )
    render_chat: bool = Field(
        default=False,
        const=True,
        description="Function messages are never rendered. Therefore this field is always false.",
    )


class ValidatedFormatString(BasePydanticModel):
    """Define the model for the key safe format string."""
    format_string: str = Field(
        ...,
        description="The format string.",
    )
    kwargs: dict[str, str] = Field(
        ...,
        description="The keys in the format string.",
    )

    @validator("kwargs")
    def validate_keys(cls, keys: dict[str, str], values: dict) -> dict[str, str]:
        """Validate the keys and ensure all are in the format string and there are no extra keys in format string."""
        format_string_keys = re.findall(r"\{([a-zA-Z0-9_]+)\}", values["format_string"])
        for key in keys:
            if key not in format_string_keys:
                raise ValueError(f"Key {key} in keys not found in format string.")
        for key in format_string_keys:
            if key not in keys:
                raise ValueError(f"Key {key} in format string not found in keys.")
        return keys

    def format(self) -> str:
        """Format the format string."""
        return self.format_string.format(**self.kwargs)


SUMMARIZE_CHAT_SESSION_SYSTEM_PROMPT = """\
You are an excellent summarizer. You will be given a chat session and your job is to summarize the chat session \
in {start}-{stop} sentences. You should be sure to outline the important points of the chat session and \
should provide a title for the session using the following format:
    ## <title>
    <summary>

Here's the first chat session that I would like you to summarize:
"""


class TextNumber(str, Enum):
    """Define the number of sentences to use when summarizing."""

    THREE = "three"
    FOUR = "four"
    FIVE = "five"
    SIX = "six"


def range_to_num_sentences(sentence_range: range) -> tuple[TextNumber, TextNumber]:
    """Convert a range to a number of sentences into the format <start>-<end>."""
    assert sentence_range.start < sentence_range.stop, f"Invalid range {sentence_range}. Start must be less than stop."
    mapping = {
        3: TextNumber.THREE,
        4: TextNumber.FOUR,
        5: TextNumber.FIVE,
        6: TextNumber.SIX,
    }
    start = mapping.get(sentence_range.start, None)
    assert start is not None, f"Invalid start {sentence_range.start}. Must be {mapping.keys()}."
    end = mapping.get(sentence_range.stop, None)
    assert end is not None, f"Invalid end {sentence_range.stop}. Must be {mapping.keys()}."
    return start, end


class BaseLLMChatSession(BasePydanticModel):
    """Define the base model for the LLM chat session."""
    id: Optional[UUID] = Field(
        default_factory=uuid4,
        description="The ID of the chat session.",
    )
    user_id: Optional[UUID] = Field(
        default_factory=uuid4, # this can stay optional
        description="The ID of the user that created the chat session.",
    )
    messages: list[BaseMessage] = Field(
        default_factory=list,
        description="The messages in the chat session.",
    )

    class Config:
        """Define the config for the model."""
        validate_assignment = True

    @property
    def last_chat_message(self) -> Optional[BaseMessage]:
        """Return the last chat message in the chat session."""
        if self.messages:
            return self.messages[-1]
        return None

    @property
    def last_student_message(self) -> Optional[StudentMessage]:
        """Return the last student message in the chat session."""
        for message in reversed(self.messages):
            if isinstance(message, StudentMessage):
                return message
        return None

    @property
    def last_search_query_message(self) -> Optional[SearchQuery]:
        """Return the last search query message in the chat session."""
        for message in reversed(self.messages):
            if isinstance(message, SearchQuery):
                return message
        return None

    @property
    def last_human_message(self) -> Optional[HumanMessage]:
        """Return the last human message in the chat session."""
        for message in reversed(self.messages):
            if isinstance(message, HumanMessage):
                return message
        return None

    @staticmethod
    def from_message(
        message: BaseMessage,
        id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
    ) -> "BaseLLMChatSession":
        """Create a new chat session from a message."""
        return BaseLLMChatSession(
            id=id or uuid4(),
            user_id=user_id,
            messages=[message]
        )

    def append_chat_messages(self, new_messages: Union[list[BaseMessage], BaseMessage]) -> None:
        """Append a chat message to the chat session."""
        if isinstance(new_messages, BaseMessage):
            new_messages = [new_messages]
        msgs = copy.deepcopy(self.messages)
        msgs.extend(new_messages)
        self.messages = msgs

    def insert_system_prompt(self, prompt: str) -> None:
        """Insert a system prompt to the beginning of the chat session."""
        if self.messages and isinstance(self.messages[0], SystemMessage):
            self.messages[0].content = prompt
        else:
            self.messages.insert(0, SystemMessage(content=prompt))

    def remove_system_prompt(self) -> None:
        """Remove the system prompt from the beginning of the chat session."""
        if self.messages and isinstance(self.messages[0], SystemMessage):
            self.messages.pop(0)

    def remove_unrendered_messages(self, num_unrendered_blocks_to_keep: int = 1) -> None:
        """
        Remove messages from the chat session that aren't rendered to the user,
        except for the last 'num_unrendered_blocks_to_keep' sets of
        non-rendered messages before a rendered message.

        This method will remove all messages where the 'render_chat' property is False,
        except for the last 'num_unrendered_blocks_to_keep' blocks of contiguous non-rendered messages.
        This is useful for when the context of the most recent non-rendered messages is needed for comprehension.

        Args:
            num_unrendered_blocks_to_keep: The number of unrendered message blocks to retain in the chat history.
                Each 'block' is a contiguous series of unrendered messages surrounded by rendered messages.
        """
        blocks_count = 0
        started_new_block = False
        new_messages = []

        for i in reversed(range(len(self.messages))):
            if not self.messages[i].render_chat:
                if started_new_block is False:
                    started_new_block = True
                    blocks_count += 1

                if blocks_count <= num_unrendered_blocks_to_keep:
                    new_messages.insert(0, self.messages[i])
            elif self.messages[i].render_chat:
                started_new_block = False
                new_messages.insert(0, self.messages[i])
        
        self.messages = new_messages

    def get_token_count(
        self,
        model_name: ModelName,
        exclude_system_prompt: bool = False
    ) -> int:
        """Return the tokens used in the chat."""
        def count_tokens_tiktoken(text: str) -> int:
            encoding = tiktoken.encoding_for_model(model_name.value)
            tokens = encoding.encode(text)
            return len(tokens)

        count = 0
        if model_name == ModelName.GPT_TURBO or model_name == ModelName.GPT_TURBO_LARGE_CONTEXT \
            or model_name == ModelName.GPT_4:
            token_function = count_tokens_tiktoken
        else:
            raise NotImplementedError(f"Invalid model name {model_name}.")

        for message in self:
            if exclude_system_prompt and isinstance(message, SystemMessage):
                continue
            count += token_function(message.content)
        return count

    def max_tokens_allowed_in_session(self, model_name: ModelName) -> int:
        """Return the max tokens allowed in the session."""
        tokens = MODEL_TO_TOKEN_WINDOW_SIZE_MAPPING.get(model_name, None)
        assert tokens is not None, f"Invalid model name {model_name}."
        return tokens

    def average_tokens_per_message(self, exclude_system_prompt: bool = False) -> float:
        """Return the average tokens per message."""
        total_tokens = self.get_token_count(len, exclude_system_prompt=exclude_system_prompt)
        return total_tokens / len(self.messages)

    def __str__(self) -> str:
        """Return the string representation of the chat session."""
        # iterate over messages and create a conversation with roles and content of messages
        conversation = ""
        for message in self.messages:
            if isinstance(message, SystemMessage):
                continue
            conversation += f"{message.role}: {message.content}\n"
        return conversation

    def _get_summarization_chat_session(self, system_prompt: str, sentence_range: Optional[range] = None) -> "BaseLLMChatSession":
        """Get a summarization chat session."""
        if not sentence_range:
            sentence_range = range(3, 5)
        start, stop = range_to_num_sentences(sentence_range)
        system_prompt_format_str = ValidatedFormatString(
            format_string=system_prompt,
            kwargs={
                "start": start,
                "stop": stop,
            },
        )
        system_prompt_message = SystemMessage.from_prompt(system_prompt_format_str.format())
        summarization_chat_session = BaseLLMChatSession(
            id=self.id,
            messages=[
                system_prompt_message,
            ],
        )
        summarization_chat_session.append_chat_messages(
            new_messages=StudentMessage(
                content=str(self),
                role=ChatRole.STUDENT,
            )
        )
        return summarization_chat_session

    def summarize(self, model: BaseChatModel, sentence_range: Optional[range] = None, **model_kwargs) -> str:
        """Summarize the chat session."""
        summarization_chat_session = self._get_summarization_chat_session(
            system_prompt=SUMMARIZE_CHAT_SESSION_SYSTEM_PROMPT,
            sentence_range=sentence_range,
        )
        response = model(messages=summarization_chat_session.messages, **model_kwargs)
        return response.content

    def __iter__(self):
        """Iterate over the messages in the chat session."""
        return iter(self.messages)


TUTOR_SUMMARIZE_CHAT_SESSION_PROMPT = """\
Hey {name}! The chat session is getting kinda long. Can you summarize the chat session for me? \
Please use the following format:
## <title>
### <summary>

Please be sure to highlight all the important points, especially and technical material. \
"""


class TaiChatSession(BaseLLMChatSession):
    """Define the model for the TAI chat session. Compatible with LangChain."""
    # TODO: need to make this required once BE supports
    user_id: Optional[UUID] = Field(
        default_factory=uuid4,
        description="The ID of the user that created the chat session.",
    )
    class_id: Optional[UUID] = Field(
        default=None,
        description="The class ID to which this chat session belongs.",
    )
    class_name: str = Field(
        ...,
        max_length=100,
        min_length=1,
        description="The name of the class that the chat session is for.",
    )
    class_description: str = Field(
        ...,
        max_length=800,
        min_length=1,
        description="The description of the course that the chat session is for.",
    )

    def summarize(self, model: BaseChatModel, sentence_range: range | None = None, **model_kwargs) -> str:
        chat_session = copy.deepcopy(self)
        last_tutor_name = ""
        for message in reversed(chat_session.messages):
            if isinstance(message, TaiTutorMessage):
                last_tutor_name = message.tai_tutor_name
                break
        prompt_format_str = ValidatedFormatString(
            format_string=TUTOR_SUMMARIZE_CHAT_SESSION_PROMPT,
            kwargs={
                "name": last_tutor_name,
            },
        )
        chat_session.append_chat_messages(
            new_messages=StudentMessage(
                content=prompt_format_str.format(),
                role=ChatRole.STUDENT,
            )
        )
        return model(messages=chat_session.messages, **model_kwargs).content

MARKDOWN_PROMPT = """\
Respond in markdown format with inline LaTeX support using these delimiters:
    inline: $...$ or $$...$$
    display: $$...$$
    display + equation number: $$...$$ (1)\
"""


SUMMARIZER_SYSTEM_PROMPT = f"""\
You are a summarizer. You will be given a list of documents and you're \
job is to summarize the documents in about 3-4 sentences for the user. \
{MARKDOWN_PROMPT}
Please insert equations as necessary when summarizing for the user query. \
You should not directly reference the documents in your summary. Pretend like the documents represent \
information that you already know and you are paraphrasing the information for \
the user. Remember, you must respond in markdown format with equations in LaTeX format.\
"""

SUMMARIZER_USER_PROMPT = """\
User Query:
{user_query}
Documents:
{documents}
Summary:\
"""

STUDENT_COMMON_QUESTIONS_SYSTEM_PROMPT = """\
You are a helpful assistant designed to help professors understand what \
their students are struggling with. You will be given a list of student \
interactions with a teaching assistant, and your job is to save a list \
of up to 10 most common questions ordered by most commonly asked. If there are no \
specific questions that students asked, you should try to create a list \
of questions that were implied by the student messages. For example, if \
a student says "I am struggling on homework 1", this implies that they \
are asking for help on homework 1: "Can you help me with homework 1?". \
You must respond with a list of 10 questions or less. To help the professor, \
please order the questions from most common to least common. Remember, \
you must only return a list of 10 questions so you must group similar \
questions together. You must not return more than ten questions! Here are \
the student messages:\
"""

STUDENT_COMMON_DISCUSSION_TOPICS_SYSTEM_PROMPT = """\
You are a helpful assistant designed to help professors understand what \
their students are struggling with. You will be given a list of student \
interactions with a teaching assistant, and your job is to create a list \
of up to 10 top discussed topics ordered by most commonly discussed. If there are no \
explicit discussion topics that students discussed, you should try to \
create a list of discussion topics that were implied by the student messages. \
You must respond with a list of 10 discussion topics or less. To help the \
professor, please order the discussion topics from most discussed by the \
students to least discussed. Please provide as much detail as possible \
for each discussion topic so that the professor can understand what the \
students were discussing. \
Remember, you must only return a list of 10 discussion topics so you must \
group similar discussion topics together. You must not return more than \
ten discussion topics! Here are the student messages:\
"""

FINAL_STAGE_STUDENT_TOPIC_SUMMARY_SYSTEM_PROMPT = """\
Please condense this list by grouping by topic, using 'and' where necessary to combine:\
"""

STEERING_WHEN_RESULTS_PROMPT = """\
Thought: I found some great results for the student! I should remember to answer in a \
friendly tone and ask questions to help them work to understand the concept they are asking about. \
I need to remember that the best learning is interactive and that I should ask questions to help the \
student learn. I should also remember that I am {name} and {persona}. \
Here's my response:\
"""

# STEERING_PROMPT = """\
# Thought: I don't know anything about what the user is asking because I am a tutor for '{class_name}'. \
# I must be honest with the student and tell them that I don't know about that concept \
# because it is not related to '{class_name}' and I should suggest that they use Google to find more info or instruct them to ask \
# their Instructor or TA for further help. I can also give the answer my best shot, but I must \
# disclaim that I am not an expert in the requested subject matter so I don't mislead the student. \
# """

# this one has gotten much better feedback from professors:
STEERING_PROMPT = """\
Thought: I wasn't able to find any direct resources from the database for '{class_name}'. \
I can answer the student, but i should be honest and provide them a bold disclaimer that I wasn't able to find any resources \
so my answer may not be correct. I should clearly put this disclaimer in my response. \
I can also suggest that they use Google to find more info or instruct them to ask \
their Instructor or TA for further help. \
Regardless of how I respond, I need to remember that I am {name} and {persona}. \
"""

BASE_SYSTEM_MESSAGE = f"""\
You are a friendly tutor named {{name}} that tutors for a class called '{{class_name}}'. As {{name}}, {{persona}}. \
You are to be a good listener, ask how you can help the student, and inspire them on their academic journey. \
You MUST get to know them as a human being and understand their needs in order to be successful. \
To do this, you need to ask questions to understand the student as best as possible. \
{MARKDOWN_PROMPT}
The student has requested that you use responses with a technical level of a {{technical_level}} to help the understand the material. \
Remember, you should explain things in a way that a {{technical_level}} would understand. \
Remember, your name is {{name}} and {{persona}}. At times, you may not know the answer to a question \
because you are a tutor only for '{{class_name}}'. That's okay! If this occurs you should prompt the student \
to reach out to their professor or TA and give your best shot at the answer, but provide a disclaimer that the \
subject is not your expertise and that your answer may not be correct.\
"""

MILO = {
    "name": TaiTutorName.MILO.value,
    "persona": "you are are less formal in how you talk and are technically savvy. You never resist the urge to incorporate real-world examples into your explanations.",
}
DECLAN = {
    "name": TaiTutorName.DECLAN.value,
    "persona": "you have a balanced conversational style and are really creative. You love thinking outside the box and are always looking for new ways to explain things.",
}
FINN = {
    "name": TaiTutorName.FINN.value,
    "persona": "you are informal and are very empathetic. You looove to weave narratives into your explanations and are always looking for ways to make things more relatable.",
}
ADA = {
    "name": TaiTutorName.ADA.value,
    "persona": "you slightly informal, but highly creative. You love to ask questions that might seem random, but are actually very insightful to help connect dots for the student.",
}
REMY = {
    "name": TaiTutorName.REMY.value,
    "persona": "you are very informal and highly creative. You love artsy things and are always looking for ways to make things more relatable.",
}
KAI = {
    "name": TaiTutorName.KAI.value,
    "persona": "you are formal, but still bring some creativity. You excel at diving deep on technical topics and are always looking to nerd out with the student.",
}
VIOLET = {
    "name": TaiTutorName.VIOLET.value,
    "persona": "you are formal and very technical. You are very good at explaining technical topics and are always looking to nerd out with the student.",
}
RESPONSE_TECHNICAL_LEVEL_MAPPING = {
    ResponseTechnicalLevel.EXPLAIN_LIKE_IM_5: "5 year old",
    ResponseTechnicalLevel.EXPLAIN_LIKE_IM_IN_HIGH_SCHOOL: "high school student",
    ResponseTechnicalLevel.EXPLAIN_LIKE_IM_IN_COLLEGE: "college student",
    ResponseTechnicalLevel.EXPLAIN_LIKE_IM_AN_EXPERT_IN_THE_FIELD: "expert in the field",
}

class TaiProfile(BasePydanticModel):
    """Define the model for the TAI profile."""
    name: TaiTutorName = Field(
        ...,
        description="The name of the tutor.",
    )
    persona: str = Field(
        ...,
        description="The persona of the tutor.",
    )

    @staticmethod
    def get_profile(name: TaiTutorName) -> "TaiProfile":
        """Get the profile for the given name."""
        if name == TaiTutorName.MILO:
            return TaiProfile(**MILO)
        elif name == TaiTutorName.DECLAN:
            return TaiProfile(**DECLAN)
        elif name == TaiTutorName.FINN:
            return TaiProfile(**FINN)
        elif name == TaiTutorName.ADA:
            return TaiProfile(**ADA)
        elif name == TaiTutorName.REMY:
            return TaiProfile(**REMY)
        elif name == TaiTutorName.KAI:
            return TaiProfile(**KAI)
        elif name == TaiTutorName.VIOLET:
            return TaiProfile(**VIOLET)
        else:
            raise ValueError(f"Invalid tutor name {name}.")

    @staticmethod
    def get_system_prompt(name: TaiTutorName, technical_level: ResponseTechnicalLevel, class_name: str) -> str:
        """Get the system prompt for the given name."""
        tai_profile = TaiProfile.get_profile(name)
        technical_level_str = RESPONSE_TECHNICAL_LEVEL_MAPPING[technical_level]
        format_string = ValidatedFormatString(
            format_string=BASE_SYSTEM_MESSAGE,
            kwargs={
                "technical_level": technical_level_str,
                "class_name": class_name,
                **tai_profile.dict(),
            },
        )
        return format_string.format()

    @staticmethod
    def get_results_steering_prompt(name: TaiTutorName) -> str:
        """Get the steering prompt for the given name."""
        tai_profile = TaiProfile.get_profile(name)
        format_string = ValidatedFormatString(
            format_string=STEERING_WHEN_RESULTS_PROMPT,
            kwargs={
                "name": tai_profile.name,
                "persona": tai_profile.persona,
            },
        )
        return format_string.format()
