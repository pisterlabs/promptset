from openai import OpenAI
client = OpenAI()

response = client.fine_tuning.jobs.create(
  training_file="file-BeToPjiorIxkIdOwfD9sl6LJ", #! need to use id's instead
  validation_file="file-MOCoL94mxSVVTac8YMAoulN2", 
  model="gpt-3.5-turbo-1106", 
  hyperparameters={
    "n_epochs":5,
    "learning_rate_multiplier":.05 #! not sure what is possible
  },

)
print(response)