import openai
import os
import datetime

# set your openai api key using os environment variables
if openai.api_key == "":
        
    # try except block to check if openai key has been set
    try:
        openai.api_key = os.getenv["OPENAI_API_KEY"]
    except:
        print("Please set your openai api key in the environment variables")
        exit()




# functions for interacting with openai api
def upload_to_openai(filename):
      openai.File.create(
    file=open(filename, "rb"),
    purpose='fine-tune'
)


# get all files that you have uploaded to openai
def get_all_files():

      files = openai.File.list()

      for file in files['data']:
            # print(file.id, file.filename)
            print(file)


# get all fine-tuned models that you have created
def get_all_finetunes():
      # get all fine-tuned models
      finetunes = openai.FineTune.list()['data']
      for ft in finetunes:
            print(ft['fine_tuned_model'])


# create a fine-tuned model
def fine_tune_model(training_file, training_model = "davinci", custom_name = "my-custom-model"):
      openai.FineTune.create(training_file = training_file, model = training_model, suffix = custom_name)




def get_finetune_status():
    response = openai.FineTune.list()
    data = response['data']
    for item in data: 
        dt_created = datetime.fromtimestamp(item['created_at'])
        dt_updated = datetime.fromtimestamp(item['updated_at'])
        time_taken = dt_updated - dt_created

        print(item['object'], item['id'], item['status'], item['model'],item['fine_tuned_model'], dt_created,dt_updated, time_taken)

    return