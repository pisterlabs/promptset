
import pandas as pd
from openai_parallel_toolkit import ParallelToolkit, Prompt, OpenAIModel


df = pd.read_excel('parallel_check.xlsx', sheet_name= "Sheet1")

# model_id = "ft:gpt-3.5-turbo-0613:personal::7tTIG42I"

model_v5 = "ft:gpt-3.5-turbo-0613:personal::7v2YWKhT"

model = OpenAIModel(model_v5, temperature=0.5)
toolkit = ParallelToolkit(config_path="config.json", openai_model=model)

responses = []
for prompt in df['Prompt'].tolist():
    prompt_dict = {0: Prompt(instruction=" ", input=prompt)}
    response = toolkit.parallel_api(data=prompt_dict)
    responses.append(response[0])

df['Responses'] = responses
df.to_excel('llel_check.xlsx', index=False)
