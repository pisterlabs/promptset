from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pandas as pd

model = Ollama(
    model="neural-chat:7b-v3.2-fp16",
    temperature=0.7,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

job_postings = pd.read_csv(
    "../data/jobs_Head_of_Product_-_European_Union_-_Remote_-_LinkedIn.csv"
)
descriptions = job_postings["job_description"].to_list()
test_output = descriptions[0]

desired_format = """
[{ "skills": <keywords>, "reference": <sentence from description>}]
"""
example = """
[{ "skills": "python", "reference": "You will be able to write python code."},
{ "skills": "java", "reference": "You will be able to write java code."}]
"""
prompt = f"""
You are a helpful assisant to extract job skills and references from job descriptions.
Your responses should be in the following format:\n\n
{desired_format}\n\n
Here is an example output:\n\n
{example}\n\n
Extract the Job skills, with the reference text to that skill, from the following description: {test_output}
"""
print(prompt)
print(model(prompt))
