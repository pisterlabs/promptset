import re
import openai
import logging
from supabase import Client


from llm import LLM

logging.basicConfig(level=logging.INFO)

class Entity(LLM):
    pass