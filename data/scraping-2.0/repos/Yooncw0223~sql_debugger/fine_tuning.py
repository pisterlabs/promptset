# this python scripts starts fine-tuning job on OpenAI API

import os

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("openaiAPI"))

client.fine_tuning.jobs.create(
  # training_file="file-7emHlOyaQFFFNByMrbtJRr9M",
  training_file="file-Y6PC7KpdbXKYUzynE8dvM4vR",
  model="gpt-3.5-turbo"
)

client.fine_tuning.jobs.create(
  # training_file="file-7emHlOyaQFFFNByMrbtJRr9M",
  training_file="file-RZZKed8zEYZnBI3RBXwROvfO",
  model="gpt-3.5-turbo"
)

client.fine_tuning.jobs.create(
  # training_file="file-7emHlOyaQFFFNByMrbtJRr9M",
  training_file="file-cm2dGiG4CLtR5Sw7vJzd44I2",
  model="gpt-3.5-turbo"
)

print("Just started fine-tuning; wait for the emails!\n")
