from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
import openai
import json
import numpy as np
from pydantic import BaseModel
# Getting OpenAI API Key
OPENAI_API_KEY = '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'

openai.api_key = OPENAI_API_KEY

# Parameters for OpenAI
openai_model = "gpt-4"
max_responses = 1
temperature = 0.7
max_tokens = 2048
city_width = 12
city_height = 8


# Defining the FastAPI app and metadata
app = FastAPI(
    title="The City Creator",
    description="""### API specifications\n
CURL can support the stream.
              """,
    version=1.0,
)

# creates a websocket server



# create a 2d array of zeros, with size 12*8
app.current_city= np.zeros((city_width,city_height))


app.past_prompts = []

# load message from json file promt_past.json
with open("./data/prompt_past.json", "r") as f:
    app.past_prompts = json.load(f)

print ("past_prompts", app.past_prompts)

# Define the Pydantic model for the request body
class Item(BaseModel):
    action: str
    value: int

class SinglePrompt(BaseModel):
    role: str
    content: str

class PromptHistory(BaseModel):
    prompt_history: list[SinglePrompt]


# Defining error in case of 503 from OpenAI
error503 = "OpenAI server is busy, try again later"


def write_prompt(prompt):
    with open("./data/prompt_past.json", "w", encoding='utf8') as f:
        json.dump(prompt, f, ensure_ascii=False)

def get_response_openai(prompt):
    try:
        app.past_prompts.append({"role": "user", "content": prompt})
        # save the prompt_past.json file
        write_prompt(app.past_prompts)
        prompt = prompt
        response = openai.ChatCompletion.create(
            model=openai_model,
            temperature=temperature,
            max_tokens=max_tokens,
            n=max_responses,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            messages=app.past_prompts,
            stream=True,
        )
    except Exception as e:
        print("Error in creating campaigns from openAI:", str(e))
        raise HTTPException(503, error503)
    try:
        the_response =''
        for chunk in response:
            current_content = chunk["choices"][0]["delta"].get("content", "")
            the_response += current_content
            yield current_content

        # print out the overall response
        print("OpenAI Response (Streaming): " + the_response)
        append_prompt = {"role": "assistant", "content": the_response}
        app.past_prompts.append(append_prompt)
        print ("app.past_prompts", app.past_prompts)
        # save the prompt_past.json file
        write_prompt(app.past_prompts)
        print ("app.past_prompts", app.past_prompts)

    except Exception as e:
        print("OpenAI Response (Streaming) Error: " + str(e))
        raise HTTPException(503, error503)


@app.get("/map")
async def get_array():
    return {"map": app.current_city.tolist(),"city_width":city_width,"city_height":city_height}      # return as json

@app.post("/map/{x}/{y}")
async def update_value(x: int, y: int, item: Item):
    # The function parameters x, y will correspond to the path parameters
    # And the parameter item will be inferred from the JSON request body
    if item.action == "update":
        # Your code here to handle 'update' action using 'value' and the coordinates x and y
        app.current_city[x][y]=item.value
        # return the updated map
        return {"status":"sucess","map": app.current_city.tolist(),"city_width":city_width,"city_height":city_height}
    else:
        return {"status": "error", "message": "Invalid action provided."}

@app.get("/prompt_history/reset")
def reset_prompt():
    temp_text = ""

    with open("./data/promt_info.txt", "r") as f:
        temp_text = f.read()
    # write out the prompt_past.json file
    temp_promt= {"role": "system", "content": temp_text}
    app.past_prompts = [temp_promt]
    write_prompt(app.past_prompts)
    return {"status":"sucess","prompt_list":app.past_prompts }

@app.post("/prompt_history")
async def update_prompt_history(prompt_history: PromptHistory):
    app.past_prompts = prompt_history.dict()["prompt_history"]
    # write out the prompt_past.json file
    write_prompt(app.past_prompts)
    return {"status":"sucess","prompt_list":app.past_prompts }

@app.get("/prompt_history")
async def get_prompt_history():
    return {"status":"sucess","prompt_list":app.past_prompts }

@app.get(
    "/city/",
    tags=["APIs"],
    response_model=str,
    responses={503: {"detail": error503}},
)

def city(prompt: str = Query(..., max_length=8000)):
    """
    update the city map based on the prompt
    """
    return StreamingResponse(get_response_openai(prompt), media_type="text/event-stream")


def clear_past_prompts():
    past_prompts = []
    # clear the current_city to all zeros
    app.current_city= np.zeros((city_width,city_height))
    with open("./data/background.json", "w") as f:
        json.dump(app.past_prompts, f)