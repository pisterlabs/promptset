from cmd import PROMPT
from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv('GPT3_FINE_TUNE_MODEL')

start_prompt = '''
Hi ! The following is a conversation with Nerdo, an AI-Powered Bot trained on an open source dataset of COVID-19 Academia Knowledge. Nerdo is helpful, creative, clever, friendly and has a good grasp on concept related to coronavirus.

User: How has the number  of  childhood pneumonia been reduced?\n\n###\n\n
Bot: New conjugate vaccines against Haemophilus influenzae type b and Streptococcus pneumoniae have contributed to decreases in radiologic, clinical and complicated pneumonia cases. \n\n###\n\n
User: What is the treatment for MERS-COV?\n\n###\n\n
Bot: There is no specific treatment for MERS-CoV. Like most viral infections, the treatment options are supportive and symptomatic.\n\n###\n\n
User: What causes Q fever?\n\n###\n\n
Bot: Coxiella burnetii (C. burnetii) causes Q fever. \n\n####\n\n
User: Thank you for the help!\n\n###\n\n
Bot: You're welcome, come back anytime!'''


start_sequence = "\n\n###\n\nBot: "
restart_sequence = "\n\n###\n\nUser: "

#curl https://api.openai.com/v1/files/file-16xpLIF0ai4MsB4zFvoUKNyU -H 'Authorization: Bearer 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'
incoming_text = "What is PPE?"
prompt = f'{start_prompt}{restart_sequence} {incoming_text}{start_sequence}'
response = openai.Completion.create(
    prompt=prompt,
    model=model,
    temperature=0.7,
    max_tokens=96,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.3,
    stop=['\n\n###\n\n'])

story = response['choices'][0]
print(story)