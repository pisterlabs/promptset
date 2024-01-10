!pip install simpletransformers
!pip install gTTS
!pip install openai


import json

with open(r"train.json", "r") as read_file:
  train = json.load(read_file)

with open(r"test.json", "r") as read_file:
    test = json.load(read_file)  

import logging

from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs


model_type="bert"
model_name= "bert-base-cased"
if model_type == "bert":
    model_name = "bert-base-cased"

elif model_type == "roberta":
    model_name = "roberta-base"

elif model_type == "distilbert":
    model_name = "distilbert-base-cased"

elif model_type == "distilroberta":
    model_type = "roberta"
    model_name = "distilroberta-base"

elif model_type == "electra-base":
    model_type = "electra"
    model_name = "google/electra-base-discriminator"

elif model_type == "electra-small":
    model_type = "electra"
    model_name = "google/electra-small-discriminator"

elif model_type == "xlnet":
    model_name = "xlnet-base-cased"


# Configure the model
model_args = QuestionAnsweringArgs()
model_args.train_batch_size = 16
model_args.evaluate_during_training = True
model_args.n_best_size=3
model_args.num_train_epochs=5

### Advanced Methodology
train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "use_cached_eval_features": True,
    "output_dir": f"outputs/{model_type}",
    "best_model_dir": f"outputs/{model_type}/best_model",
    "evaluate_during_training": True,
    "max_seq_length": 128,
    "num_train_epochs": 20,
    "evaluate_during_training_steps": 1000,
    "wandb_project": "Question Answer Application",
    "wandb_kwargs": {"name": model_name},
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    "n_best_size":3,
    # "use_early_stopping": True,
    # "early_stopping_metric": "mcc",
    # "n_gpu": 2,
    # "manual_seed": 4,
    # "use_multiprocessing": False,
    "train_batch_size": 128,
    "eval_batch_size": 64,
    # "config": {
    #     "output_hidden_states": True
    # }
}

model = QuestionAnsweringModel(
    model_type,model_name, args=train_args
)

### Remove output folder
!rm -rf outputs

# Train the model
model.train_model(train, eval_data=test)

import openai
openai.api_key= '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'

model_id = "gpt-3.5-turbo"




# Evaluate the model
result, texts = model.eval_model(test)

# Make predictions with the model
to_predict = [
    {
        "context": "The  third party involved is Flipkart",
        "qas": [
            {
                "question": "who is the third party",
                "id": "0",
            }
        ],
    }
]

def find_maximum(lst,start=0,max_word=''):
  if start==len(lst):  #base condition
    return max_word
  if len(lst[start])>len(max_word): 
    max_word=lst[start]
  return find_maximum(lst,start+1,max_word)  #calling recursive function
  

answers, probabilities = model.predict(to_predict)

print(answers)
if(answers[0]["answer"][0] != 'empty'):
  print(answers[0]["answer"])
  resp = find_maximum(answers[0]["answer"])
else:
  chat = openai.ChatCompletion.create(
  model=model_id,
  messages=[
  {"role": "user", "content": to_predict[0]["qas"][0]["question"]}
  ]
  )
  resp = chat.choices[0].message.content
  print(chat.choices[0].message.content)


from gtts import gTTS
#Import Google Text to Speech
from IPython.display import Audio #Import Audio method from IPython's Display Class
tts = gTTS(resp, lang='en', tld='co.uk') #Provide the string to convert to speech
tts.save('1.wav') #save the string converted to speech as a .wav file
sound_file = '1.wav'
Audio(sound_file, autoplay=True)
