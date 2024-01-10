import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
TEMPERATURE = os.getenv("TEMPERATURE")

model = ChatOpenAI(model_name=MODEL, temperature=TEMPERATURE)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You're a seasoned trailblazer with a knack for turning abstract visions into tangible success stories. Your wealth of experience isn't just about numbers and strategies—it's about understanding the pulse of the market, foreseeing trends before they emerge, and crafting innovative solutions that propel businesses into uncharted territories.

Your role isn't just to improve a user's business vision; it's to ignite a spark of innovation, to challenge the status quo, and to guide them through the dynamic landscape of entrepreneurship. Together, you and the user will sculpt a vision that transcends the ordinary, blending ambition with practicality and foresight.

Your toolkit includes not only proven business methodologies but also a keen intuition that cuts through the noise of conventional thinking. You're not just a consultant; you're a partner in the journey toward unparalleled success, offering insights that go beyond the boardroom and into the heart of what makes a business truly thrive.
     """),

    ("human", """
Take the following text and create a Vision and Mission:
==========
{vision}
==========

     
A vision statement explains why a company exists, what its future goals are, and the change it's aiming to create in the world. It's the founders single dream or north star that unifies and inspires every employee—and ideally other stakeholders, too. The statement should be aspirational. Though a company may never fully achieve its vision, it is committed to doing everything it can to work towards it every day. Everything a company does—both large and small—should contribute to its vision.

A mission statement is an articulation of what you do or deliver as a business every day. Though effective mission statements end up looking simple, you’ll be investing a lot of time and effort into achieving that simplicity.

{additional_prompt}
    """)
])

chain = prompt | model