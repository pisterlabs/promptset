import os

# Utils
import time
from typing import List

# Langchain
import langchain
from pydantic import BaseModel
from vertexai.language_models import TextGenerationModel

print(f"LangChain version: {langchain.__version__}")

# Vertex AI
from langchain.llms import VertexAI

from llm_experiments.utils import here

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(here() / 'motorway-genai-ccebd34bd403.json')


generation_model = TextGenerationModel.from_pretrained("text-bison@001")
prompt = "What is a large language model?"
response = generation_model.predict(prompt=prompt)
print(response.text)



# LLM model
llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

def form_assistant_prompt(seller_email):
    return f"""Your task is to assist a customer service agent.
                Step 1: Summarise the following email from a customer who is trying to sell their vehicle and the agent. Use 2-5 bullet points.
                Step 2: Output the recommended action for the agent.
                Step 3: Draft a response for the agent, in a polite but semi-formal and friendly tone (suitable for a start up).
                        The email will be delimited with ####
                        Format your response like this:
                        ****
                        Summary:
                        - <bullet 1>
                        - <bullet 2>
                        - <bullet 3>
                        ****
                        Recommended Action:
                        <Action>
                        ****
                        Draft Response:
                        <Draft response>
                        ****

                        Customer email below:

                        ####
                        {seller_email}
                        ####

                        Response:
                        """



# You'll be working with simple strings (that'll soon grow in complexity!)
seller_email = """Good morning Unfortunately the dealer who came to collect the car decided that they only wanted to offer me ¬£9800 when they arrived to collect the car and i was not willing to let it go for that amount.
There was 3 men who cam to collect it and it felt really intimidating and pressured.The car has not been collected."""

result = llm(form_assistant_prompt(seller_email))
print(result)


####

import streamlit as st



