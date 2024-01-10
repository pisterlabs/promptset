import openai

openai.api_key = "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"
path = 'train_data/ft_data.jsonl'

def upload_ftdata():
    with open(path, "rb") as file:
        response = openai.File.create(
            file = file,
            purpose = 'fine-tune'
        )
    file_id = response['id']
    print(f"File uload with id: {file_id}")
    return file_id


def finetune_create(file_id):
    model_name = "gpt-3.5-turbo"
    response = openai.FineTuningJob.create(
        training_file = file_id,
        model = model_name
    )
    job_id = response['id']
    print(f"Job id: {job_id}")
    return job_id


def delete_model(model_name):
    openai.Model.delete(model_name)


# Retrieve the state of a fine-tune
def retrieve_model(job_id):
    response = openai.FineTuningJob.retrieve(job_id)
    print(response)

# file_id = upload_ftdata()
# job_id = finetune_create(file_id)
#retrieve_model('ftjob-ezyrT0OCLXSeHcaFtmbYuVZp')


#print(openai.FineTuningJob.list_events('ftjob-rRfqapvOAFZis79HXFPE9NIY',10))
#print(openai.FineTuningJob.cancel(""))
#delete_model('ft:gpt-3.5-turbo-0613:personal::81TtQzRO')

