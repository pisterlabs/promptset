from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Replicate

# FastAPI scaffolding
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
)

# Initialize replicate endpoint
# Alternative LLM options at https://replicate.com/explore
r8_llm =  Replicate(model="replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b")

# LangChain initialization
todo_prompt = """
INSTRUCTIONS: 
Extract all potential to-dos or tasks from the following text in pure markdown. 
Do not make up information yourself, but you can summarize and suggest potential actions.

INPUT TEXT TO PROCESS:
{text}
"""

todo_prompt_template = PromptTemplate(
    template=todo_prompt,
    input_variables=['text'],
)
todo_chain = LLMChain(
    llm=r8_llm,
    prompt=todo_prompt_template,
)

@app.post('/extract-todos')
async def extract_todos(text: str):
    """Takes a piece of user text and returns to-dos."""
    return {'output': todo_chain.run(text=text)}

# Just needed for deployment to fly.io
@app.get("/healthcheck")
def read_root():
    return {"status": "ok"}

