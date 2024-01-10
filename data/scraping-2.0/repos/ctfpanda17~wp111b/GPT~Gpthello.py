import os
import openai
openai.api_key = os.getenv("OPEN_AI_KEY")
response = openai.Completion.create(
    model = "text-davinci-003"
    
)