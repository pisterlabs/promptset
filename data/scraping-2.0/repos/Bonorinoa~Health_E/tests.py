# Cohere's imports
import cohere as co
from cohere.classify import Example
from conversant.prompt_chatbot import PromptChatbot
import numpy as np

from qa.bot import GroundedQaBot

COHERE_API = "Qn83qLsfj9Bx3WiIWzzONJkz0gnIrW7xkmtn6KQX"
SERP_API = "9740e18bde5c0cb3e387237a3ccada2537c077a325253269207ac44e5724150e"
co = co.Client(COHERE_API)


qa_bot = GroundedQaBot(COHERE_API, SERP_API)
health_e = PromptChatbot.from_persona("health-e", client=co)


print(qa_bot.answer("what is the best hospital in the world?"))