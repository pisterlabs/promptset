# conda activate llama.cpp && which python
# python request_local_model_like_openai2.py
import openai
from langchain.prompts import PromptTemplate

uri = "http://rlaplnxml2:8080/v1"
client = openai.OpenAI(
    base_url=uri,
    api_key = "sk-no-key-required"
)

# rlapstudio
PROMPT_FOLDER_PATH="/Volumes/local-data/repos/utils/ai/prompts/TheBloke/deepseek-coder-6.7B-instruct-GGUF/vb6_select_clause"
# rlaplnxml2
# PROMPT_FOLDER_PATH="~/proyectos/rlab/utils/ai/prompts/TheBloke/deepseek-coder-6.7B-instruct-GGUF/vb6_select_clause"

PROMPT_NAME="prompt_vb6_select_clause1.txt"
PROMPT_PATH=f"{PROMPT_FOLDER_PATH}/{PROMPT_NAME}"
print(f"PROMPT_PATH: {PROMPT_PATH}")

prompt = PromptTemplate.from_file(PROMPT_PATH)
print(f"prompt: {prompt.format()}")

# model="text-davinci-003",
completion = client.chat.completions.create(
model="gpt-3.5-turbo",
messages=[
    {"role": "system", "content": "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."},
    {"role": "user", "content": f"{prompt.format()}"}
]
)

# print(completion)
print(completion.choices[0].message)