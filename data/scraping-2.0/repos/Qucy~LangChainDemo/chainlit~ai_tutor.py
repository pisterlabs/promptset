from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.schema.chat_history import BaseChatMessageHistory
from langchain.schema.runnable.history import RunnableWithMessageHistory

import chainlit as cl
from chainlit.input_widget import Select
from chainlit.input_widget import Switch
from chainlit.input_widget import Slider


from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# creating model
model = AzureChatOpenAI(
    openai_api_version=os.getenv('OPENAI_API_VERSION'),
    azure_deployment="chatGPTAzure",
    streaming=True
)

# Descriptions from the dictionary
learning_styles = {
    "active": "User chose 'Active'. As an AI Tutor, you should include interactive exercises and real-world problem-solving tasks to actively engage the user.",
    "reflective": "User chose 'Reflective'. As an AI Tutor, you should encourage journaling, self-reflection questions, and provide detailed explanations to facilitate deep thinking."
}

tone_styles = {
    "encouraging": "User prefers an 'Encouraging' tone. As an AI Tutor, you should offer frequent praise and highlight progress to boost the user's confidence and motivation.",
    "formal": "User prefers a 'Formal' tone. As an AI Tutor, you should maintain a professional and structured approach, providing clear and concise information.",
    "analytical": "User prefers an 'Analytical' tone. As an AI Tutor, you should focus on logic, data, and critical analysis, presenting facts and encouraging problem-solving.",
    "empathetic": "User prefers an 'Empathetic' tone. As an AI Tutor, you should show understanding and sensitivity, using supportive language and adapting to the user's emotional state."
}

learning_levels = {
    "beginner": "User is at a 'Beginner' level. As an AI Tutor, you should use simple language, introduce basic concepts, and provide foundational knowledge.",
    "intermediate": "User is at an 'Intermediate' level. As an AI Tutor, you should build on basic concepts, introduce more complex ideas, and encourage deeper exploration.",
    "advanced": "User is at an 'Advanced' level. As an AI Tutor, you should cover complex topics, promote critical thinking, and encourage independent study and research."
}

# define system prompt
system_prompt = """
You role is AI Tutor, a personalized learning assistant!

Your goals are:
- Personalized Learning: Adapts responses to match each user's learning style and level.
- Clarity and Accuracy: Provides clear and accurate explanations to ensure understanding.
- Positive Support: Maintains a positive and encouraging tone to motivate learners.
- Interactive Engagement: Includes quizzes and interactive discussions for a dynamic learning experience.
- Resource Recommendations: Suggests additional learning materials to supplement the educational experience.

User profile settings:
{settings}

Your actions:
- when user input the knowledge they want to lear, you should generate the curriculum for the users according to settings above
"""

# default user_settings
default_settings = """
Learning Style: Active learners prefer engagement through hands-on experiences and practical application of knowledge. The AI Tutor should include interactive exercises and real-world problem-solving tasks.
Tone Style: Use a positive and motivational tone. The AI Tutor should offer frequent praise and highlight progress to boost confidence and motivation.
Level: Designed for users new to the subject. The AI Tutor should use simple language, introduce basic concepts, and provide foundational knowledge.
"""

# Redis url for cache
REDIS_URL = "your redis url"

@cl.on_chat_start
async def start():
    # setup UI
    settings = await cl.ChatSettings(
        [
            Select(
                id="learning_style",
                label="Learning style",
                values=["Active", "Reflective"],
                initial_index=0,
            ),
            Select(
                id="tone_style",
                label="Tone style",
                values=["Encouraging", "Formal", "Analytical", "Empathetic"],
                initial_index=0,
            ),
            Select(
                id="level",
                label="Level",
                values=["beginner", "intermediate", "advanced"],
                initial_index=0,
            ),
        ]
    ).send()
    # setup settings
    cl.user_session.set("settings", default_settings)
    
    # creating prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}"),
        ])
    # setup chain
    chain = prompt | model | StrOutputParser()
    # wrap chain
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: RedisChatMessageHistory(session_id, url=REDIS_URL),
        input_messages_key="user_input",
        history_messages_key="history",
    )
    # setup chain_with_history
    cl.user_session.set("runnable", chain_with_history)



@cl.on_settings_update
async def setup_agent(settings):
    """ call back function when configuration updated
    """
    # Extracting the descriptions based on user's choice
    learning_style_desc = learning_styles[settings['learning_style'].lower()]
    tone_style_desc = tone_styles[settings['tone_style'].lower()]
    level_desc = learning_levels[settings['level'].lower()]
    
    # Concatenating the descriptions into a text
    description_text = (
        f"Learning Style: {learning_style_desc}\n"
        f"Tone Style: {tone_style_desc}\n"
        f"Level: {level_desc}"
    )
    # set user settings
    cl.user_session.set("settings", description_text)

@cl.on_message
async def send_message(message: cl.Message):
    """ handle the message come from UI
    """
    # retrieve runnable
    runnable = cl.user_session.get("runnable")
    # retrieve settings
    settings = cl.user_session.get("settings")
    # retrieve message
    msg = cl.Message(content="")
    # retrieve message from model
    async for chunk in runnable.astream(
        {"user_input": message.content, "settings": settings},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()], configurable={"session_id": cl.user_session.get("id")}),
    ):
        await msg.stream_token(chunk)
    await msg.send()


@cl.on_chat_end
def end():
    print("goodbye", cl.user_session.get("id"))
