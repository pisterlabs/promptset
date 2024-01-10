import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
# Set GPT model

# openai.api_key = os.getenv("OPENAI_API_KEY")
# gpt_model = "gpt-3.5-turbo"

openai.api_key = os.environ.get("OPEN_API")
gpt_model = "gpt-4"
# gpt_model = "gpt-3.5-turbo"

def gpt_response_2_answer_elaboration(essay_question, answered_questions, essay_length):
    response = openai.ChatCompletion.create(model=gpt_model, temperature=0, messages=[
        ### SYSTEM PROMPT
        {"role": "system",
         "content":
             """
             You are a helpful assistant.
             """
         },

        ### USER PROMPT THAT IS ACTUALLY USED FOR A RESPONSE FROM THE API
        {"role": "user",
         "content":
             f"""

             I am writing a {essay_length} word essay, and I need help elaborating on my thoughts. Do not write the 
             essay, rather elaborate on the thoughts to make them more concise, they work better together, 
             and help answer the essay question at hand. 

            Additional Instructions, 
            
            Use the following guidelines to elaborate on the answers. You can integrate stories of personal history and family struggle.
            - Personal and Authentic
            - Use Detailed Anecdotes
            - Elaborate on Evolution of Character and Relationship
            - Use Emotional Resonance
            - Use Reflection and Insight
            
            Here is the essay question:
            {essay_question} ({essay_length} word essay.)
            
            Here are my thoughts:
            {answered_questions}
            
            """},

    ])

    return response.choices[0].message.content
