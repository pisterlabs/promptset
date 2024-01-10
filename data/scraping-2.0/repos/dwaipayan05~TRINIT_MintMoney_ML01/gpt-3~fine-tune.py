import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_training_file = os.getenv("FILE_KEY")
# openai.FineTune.create(training_file=openai_training_file,
#                        model="davinci", n_epochs=5)

# print(openai.FineTune.list())
openai_fine_tune_list = openai.FineTune.list()
new_model = openai_fine_tune_list["data"][-1]
print(new_model)

