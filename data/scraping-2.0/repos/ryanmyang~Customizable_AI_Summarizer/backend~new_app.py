from fastapi import FastAPI, Request, HTTPException
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import openai
import os
import io
import datetime
from firebase_utils import add_data, get_data, set_data
from celery import Celery

# Create a FastAPI instance
app = FastAPI()
client = TestClient(app)

MODELS = ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k']


celery = Celery(
    'tasks',
    broker='redis://localhost:6379/0',  # Replace with your broker URL
    backend='rpc'  # You can change this backend based on your needs
)

# Input
def read_ref(file):
    filepath = os.path.join("refs", file)
    with open(filepath, "r") as f:
        output = f.read()
    return output

def read_message(file):
    filepath = os.path.join("messages", file)
    with open(filepath, "r") as f:
        output = f.read()
    return output

def write_gpt_log(filename, response, instructions, transcript):
    filepath = os.path.join("logs", filename)
    instructions = "" if instructions is None else instructions
    with open(filepath, "w") as f:
        f.write(response)
        f.write("\n\n\n\n\n\n-------SYSTEM--------\n")
        f.write(instructions)
        f.write("\n\n\n\n\n\n-------USER--------\n")
        f.write(transcript)
    return len(response.splitlines())

def gpt(model_num, sys,usr, log_name):
    print('GPT Called: ' + MODELS[model_num])

    if sys is not None:
        completion = openai.ChatCompletion.create(
            model=MODELS[model_num],  
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr}
            ]
            )
    else: 
        completion = openai.ChatCompletion.create(
            model=MODELS[model_num],  
            messages=[
                {"role": "user", "content": usr}
            ]
            )
    write_gpt_log(log_name, completion_text(completion), sys, usr)
    return completion

def completion_text(c):
    return c["choices"][0]["message"]["content"]


###############################################################################################################
@app.post('/api/start-job')
async def start_job(request: Request):
    uid = request.headers.get('Authorization')[7:]  # Remove 'Bearer ' from the token

    data = await request.json()
    doc_id = data['doc_id']
    extractions = data['extractions']
    combinations = data['combinations']

    firestore_data = {
        'doc_id': doc_id,
        'extractions': extractions,
        'combinations': combinations,
        'body': 'processing'
    }

    path = f'users/{uid}/summaries'
    summary_doc_path = add_data(path, firestore_data)
    transcript = get_data(f'users/{uid}/files', doc_id)['transcript']
    print(f'THIS IS THE TRANSCRIPT: {transcript}')

    task = process.delay(summary_doc_path, transcript, int(extractions),int(combinations))

    response = {'message': 'Job starting', 'transcript': transcript}
    return JSONResponse(content=response)

###############################################################################################################
# Output
@celery.task(name="process")
def process(sum_path, transcript, extraction_count: int, combine_count: int):
    print("\n\n\n STARTING PROCESS \n\n\n")
    print(f"Processing: {sum_path}, {transcript}, {extraction_count}, {combine_count}")

    # Load API Key
    load_dotenv()
    openai.api_key = os.getenv('API_KEY')


    ####################
    ##### CONTROLS #####
    ####################

    # transcript = read_ref("chevron")
    # transcript2 = read_ref("chevron_2")
    system_instructions = read_message("system_extract")
    # system_questions = read_message("system_questions")
    combine_sys = read_message("system_combine")
    sort_sys = read_message("system_sort")

    ####################
    ####################
    ####################


    log_time = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")

    if (transcript == None or system_instructions == None):
        print("Input file error")
        quit()

    ### Chat GPT

    ## First Grab out the points multiple times

    all_responses = ""
    ### Grab out points a few times
    for i in range(int(extraction_count)):

        ## GPT CALL
        all_responses += completion_text(gpt(1,system_instructions, transcript,log_time+"_e_"+str(i))) + '\n'
        # all_responses += completion_text(gpt(1,system_instructions, transcript2,log_time+"_e2_"+str(i))) + '\n'

    # question_answers = completion_text(gpt(0,"", read_message("system_questions2") + transcript,log_time+"_q"))
    # all_responses += question_answers + '\n'
    # QUIT FOR TESTING
    # quit()
    ### Combine

    last_combined = all_responses
    for i in range(combine_count):
        # GPT CALL
        last_combined = completion_text(gpt(0,combine_sys, last_combined,log_time+"_c_"+str(i)))

    ### Sorted
    ## GPT CALL
    sorted = completion_text(gpt(0,sort_sys,last_combined,log_time+"_s"))
    
    data = {
        'body': f'{sorted}'
    }
    set_data(sum_path,data)
    print("Finished Processing")
    return "Task Completed"


@app.route('/')
def index():
    return "Testing"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)