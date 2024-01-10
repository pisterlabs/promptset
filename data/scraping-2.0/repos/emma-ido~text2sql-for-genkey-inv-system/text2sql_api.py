import os
from enum import Enum

from fastapi import FastAPI
from pydantic import BaseModel
import openai


openai.api_key = os.getenv("API_KEY")
api_password = os.getenv("PASSWORD")

app = FastAPI()

class UserRequest(BaseModel):
    request : str
    password: str

@app.get("/")
def index() -> str:
    return "Hello, welcome to my API"


@app.post("/text2sql/")
def generate_sql(user_request: UserRequest):
    prompt_text = "### Postgres SQL tables, with their properties:\n#\n# asset(id, cost, date_created, defects, model, name, serial_number, brand, category, department, assigned_to, location, status, sub_category)\n# branch(branch_name)\n# brand (brand_name)\n# category(category_name)\n# department(department_name)\n# status(status)\n# sub_category(sub_category_name)\n#\n### {}. Give me the query only."
    
    if(user_request.password != api_password):
        return "Incorrect Password!"
        
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_text.format(user_request.request),
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["#", ";"] 
    )
    
    # print(prompt_text)
    output_query = response.choices[0].text.strip();
    
    if(output_query[0] == '"' and output_query[-1] == '"'):
        output_query = output_query[1:]
        output_query = output_query[:-1]
        
    return output_query;
