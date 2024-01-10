import openai
from apikey import OPEN_AI_KEY
from dataset import dataset
from Advantage import services, mission, company_name

# api key from openai
openai.api_key = OPEN_AI_KEY

dataset = [
    {
        "prompt" : "What are the services?",
        "completion" : services
    },
    {
        "prompt" : "Contact",
        "completion" : "Email : contact-us@advantagedigital\nPhone number: 6305574004"
    },
    {
        "prompt" : "What is the company misssion?",
        "completion" : mission
    }
]


default_response = "This information is not available at the moment.For more queries can contact our executive"

dataset.append(
    {"prompt" : "default_message","completion" : default_response}
    )



# function for model prompt and query
def fine_tune_model(dataset):

    formated_dataset = [{"role" : "user", "content" : data["prompt"]} for data in dataset]

    fine_tune_model = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = formated_dataset,
        temperature = 0.7,
        max_tokens = 1000,
        n=1,
        stop = None,
        frequency_penalty = 0,
        presence_penalty = 0
    )

    fine_tune_model_id = fine_tune_model['id']
    return fine_tune_model_id

fine_tune_model_id = fine_tune_model(dataset)

with open("fine_tuned_model_id.txt","w") as f:
    f.write(fine_tune_model_id)

