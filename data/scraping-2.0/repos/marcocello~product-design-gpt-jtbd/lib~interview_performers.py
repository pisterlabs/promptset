import json
import os
import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
TEMPERATURE = os.getenv("TEMPERATURE")

model = ChatOpenAI(model_name=MODEL, temperature=TEMPERATURE)

prompt = ChatPromptTemplate.from_messages([
    ("system", """

Act like this user {job_performer}. You will assume the role of this usert embodying the specified characteristics and be interviewed regarding their expertise in the indicated profession. You will fully immerse yourself in the persona of this individual, faithfully portraying their unique characteristics and experiences throughout the interview.

     """),
    
    ("human", """

Answer all the following questions:

{main_job_related}

1. Tell me a little about yourself and what you do.
2. When was the last time you did the main job?
3. How did you feel overall while getting that job done?

4. What are you trying to accomplish? What tasks are involved?
5. What problems are you trying to prevent or resolve?
6. What helps you achieve your goals?
7. What would the ideal service be to do the job for you?”
8. What else are you trying to get done?”

9. How do you get started?
10. What is the previous step? What's the next step?
11. How do you continue after that?
12. How do you make decisions along the way?
13. How do you feel at each stage in the process?
14. How do you know you are doing the job right?
15. How do you wrap things up?

16. What workarounds exist in your process?
17. What do you dread doing? What do you avoid? Why?
18. What could be easier? Why?
19. Why do you avoid doing certain parts of the job?
20. What's the most annoying part? Why is that frustrating?
21. How do you feel when the job is completed?

22. In which situations do you act differently?
23. What conditions influence your decisions?
24. How do the environment and setting affect your attitude and feelings while getting the job done?
     
This is the format
     
     {{
     "interview" {{
     [
     "question":"",
     "answer":""
     ]
     }}
     }}
""")]
)


functions = [
    {
    "name": "job_performer_interview",
    "description": "Full interview of Job Performer",
    "parameters": {
        "type": "object",
        "properties": {
            "interview": {
                "type": "array",
                "description": "The array of the questions and answers",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The single question asked"
                        },
                        "answer": {
                            "type": "string",
                            "description": "The single answer"
                        },
                    }
                }
            },
        },
        "required": ["interview"]
    },
    },
]

chain = (
    prompt 
    | model.bind(function_call={"name": "job_performer_interview"}, functions = functions)
)