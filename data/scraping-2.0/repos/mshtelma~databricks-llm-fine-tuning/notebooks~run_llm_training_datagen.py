# Databricks notebook source
# MAGIC %md
# MAGIC # LLM Trainer data generation solution
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC Training models is hard. You have to collect a dataset, clean it, get it in the right format, select a model, write the training code and train it. And that's the best-case scenario.
# MAGIC
# MAGIC The goal of this solution is to explore a experimental new pipeline to generate training data to train a high-performing task-specific model. We try to abstract away all the complexity, so it's as easy as possible to go from idea -> performant fully-trained model.
# MAGIC
# MAGIC **Simply input a description of your task, and the system will generate a dataset from scratch, parse it into the right format, and have it ready for fine-tuning your LLaMA 2 or GPT-3.5 model.**
# MAGIC
# MAGIC ## Features
# MAGIC
# MAGIC - **Dataset Generation**: Using OpenAI GPT-3.5 Turbo, `llm training datagen` will generate a variety of prompts and responses based on the provided use-case.
# MAGIC
# MAGIC - **System Message Generation**: `llm training datagen` will generate an effective system prompt for your model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Describe your model -> fine-tuned LLaMA 2
# MAGIC
# MAGIC The goal of this notebook is to experiment with a new way to make it very easy to build a task-specific model for your use-case.
# MAGIC
# MAGIC First, use the best GPU available (go to Runtime -> change runtime type)
# MAGIC
# MAGIC To create your model, just go to the first code cell, and describe the model you want to build in the prompt. Be descriptive and clear.
# MAGIC
# MAGIC Select a temperature (high=creative, low=precise), and the number of training examples to generate to train the model. From there, just run all the cells.
# MAGIC
# MAGIC You can change the model you want to fine-tune by changing `model_name` in the `Define Hyperparameters` cell.

# COMMAND ----------

# MAGIC %md
# MAGIC #Data generation step

# COMMAND ----------

# MAGIC %md
# MAGIC Write your prompt here. Make it as descriptive as possible!
# MAGIC
# MAGIC Then, choose the temperature (between 0 and 1) to use when generating data. 
# MAGIC **Lower values are great for precise tasks, like writing code, whereas larger values are better for creative tasks, like writing stories.**
# MAGIC
# MAGIC Finally, choose how many examples you want to generate. The more you generate, a) the longer it takes and b) the more expensive data generation will be. But generally, more examples will lead to a higher-quality model. 100 is usually the minimum to start.

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text(
  "num_examples",
  "200",
  "No. of examples to generate"
)
dbutils.widgets.text(
  "temperature",
  "0.2",
  "Choose Temperature (0 -> 1)"
)
dbutils.widgets.text(
  "model_task_desc",
  "A model that takes in a arithmetic math questions in English, and responds with a well-reasoned response in English.",
  "Description of the model's task"
)
# prompt = "A model that takes in a arithmetic math questions in English, and responds with a well-reasoned response in English." # "A model that takes in a puzzle-like reasoning-heavy question in English, and responds with a well-reasoned, step-by-step thought out response in Spanish."
# temperature = .2
# number_of_examples = 200

# COMMAND ----------

prompt = dbutils.widgets.get("model_task_desc")
temperature = float(dbutils.widgets.get("temperature"))
number_of_examples = int(dbutils.widgets.get("num_examples"))

# COMMAND ----------

# MAGIC %md
# MAGIC Run below segment to generate the dataset.

# COMMAND ----------

import os
import openai
import random

openai.api_key = "API-KEY" 

def generate_example(prompt, prev_examples, temperature=.5):
    messages=[
        {
            "role": "system",
            "content": f"You are generating data which will be used to train a generative AI large language model.\n\nYou will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\nOnly one prompt/response pair should be generated per turn.\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\nHere is the type of model we want to train:\n`{prompt}`"
        }
    ]

    if len(prev_examples) > 0:
        if len(prev_examples) > 10:
            prev_examples = random.sample(prev_examples, 10)
        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example
            })

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        max_tokens=1354,
    )

    return response.choices[0].message['content']

# Generate examples
prev_examples = []
for i in range(number_of_examples):
    print(f'Generating example {i}')
    example = generate_example(prompt, prev_examples, temperature)
    prev_examples.append(example)

print(prev_examples)

# COMMAND ----------

# MAGIC %md
# MAGIC We also need to generate a system message.

# COMMAND ----------

def generate_system_message(prompt):

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
          {
            "role": "system",
            "content": "You will be given a high-level description of the model we are training, and from that, you will generate a simple system prompt for that model to use. Remember, you are not generating the system message for data generation -- you are generating the system message to use for inference. A good format to follow is `Given $INPUT_DATA, you will $WHAT_THE_MODEL_SHOULD_DO.`.\n\nMake it as concise as possible. Include nothing but the system prompt in your response.\n\nFor example, never write: `\"$SYSTEM_PROMPT_HERE\"`.\n\nIt should be like: `$SYSTEM_PROMPT_HERE`."
          },
          {
              "role": "user",
              "content": prompt.strip(),
          }
        ],
        temperature=temperature,
        max_tokens=500,
    )

    return response.choices[0].message['content']

system_message = generate_system_message(prompt)

print(f'The system message is: `{system_message}`. Feel free to re-run this cell if you want a better result.')

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's put our examples into a dataframe and turn them into a final pair of datasets.

# COMMAND ----------

import pandas as pd

# Initialize lists to store prompts and responses
prompts = []
responses = []

# Parse out prompts and responses from examples
for example in prev_examples:
  try:
    split_example = example.split('-----------')
    prompts.append(split_example[1].strip())
    responses.append(split_example[3].strip())
  except:
    pass

# Create a DataFrame
df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
})

# Remove duplicates
df = df.drop_duplicates()

print('There are ' + str(len(df)) + ' successfully-generated examples. Here are the first few:')

display(df)

# COMMAND ----------

spark.createDataFrame(df).write.mode("overwrite").format("csv").coalesce(1).save("/dbfs/FileStore/tables/training_data/")
