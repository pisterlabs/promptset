import openai
from fastapi import APIRouter

router = APIRouter(
    prefix="/fine-tunning",
    tags=["Fine Tunning"]
)

    
@router.get("/fine-tune/list-jobs/")
def fine_tune_list_jobs():
    return openai.FineTuningJob.list(limit=10)

@router.post("/fine-tune/list-job/")
def fine_tune_list_job(job_id: str):
    return openai.FineTuningJob.retrieve(job_id)

@router.post("/fine-tune/list-events")
def fine_tune_list_events(job_id: str):
    return openai.FineTuningJob.list_events(job_id, limit=10)
    
@router.post("/fine-tune/create-job/")
def fine_tune_create_job():
    file = openai.File.create(
        file=open("./fineTuneData.jsonl","rb"),
        purpose="fine_tune",       
    )
    openai.FineTunningJob.create(training_file=file.id, model="gpt-3.5-turbo")

@router.post("/fine-tune/use-model/")
def fine_tune_use_model(text: str):
    completion = openai.ChatCompletion.create(
        model = "ft:gpt-3.5-turbo-0613:personal::7wGbxpsA",
        messages = [
            {"role": "system", "content": "You are a very sarcastic person. that only says bad things."},
            {"role": "user", "content": f'{text}'}
        ]
    )
    return completion.choices[0].message.content

