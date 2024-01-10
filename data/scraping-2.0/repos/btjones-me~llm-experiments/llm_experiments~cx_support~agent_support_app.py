import streamlit as st
import os

# Utils
import time
from typing import List

# Langchain
import langchain
from pydantic import BaseModel
from vertexai.language_models import TextGenerationModel

# Vertex AI
from langchain.llms import VertexAI
from llm_experiments.utils import here

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(here() / 'motorway-genai-ccebd34bd403.json')

# LLM model
llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=1024,
    temperature=0.3,
    top_p=0.8,
    top_k=40,
    verbose=True,
)


def form_assistant_prompt(seller_email):
    return f"""Your task is to assist a customer service agent.
                Step 1: Summarise the following email from a customer who is trying to sell their vehicle and the agent. Use 2-5 bullet points.
                Step 2: Estimate the customer's emotional state in no more than 3 words. This might be distressed, angry, upset, happy, excited etc.
                Step 3: Output the recommended action for the agent.
                Step 3: Draft a response for the agent, in a polite but semi-formal and friendly tone (suitable for a start up).
                        The email will be delimited with ####
                        Format your response like this:
                        ****
                        Summary:
                        - <bullet 1>
                        - <bullet 2>
                        - <bullet 3>
                        ****
                        Customer's emotional state:
                        <examples: distressed, angry, upset>
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


def form_validation_prompt(summary_and_context, proposed_response):
    return f"""Your task is to assist a customer service agent. You will receive an email from a customer
                and a proposed response from a customer support agent. You will also receive a summary, an estimate of the
                customer's emotional state, the recommended action and then the draft response.
                You should conduct the following actions:
                Step 1: Identify if the proposed response meets the needs of the customer and is of the appropriate tone
                to match the customer's emotional state.
                Respond with a "Yes, this meets the customer's needs" or "No, this doesn't meet the customer's needs because <give reason>"
                Step 2: Recommend a better response that meets the customer's needs more closely.

                The information will be delimited with ####

                Format your response like this:
                ****
                Validation:
                <response>
                ****
                Improved response:
                <response>
                ****

                Context:

                ####
                {summary_and_context}
                ####

                Proposed Response:

                ####
                {proposed_response}
                ####

                Response:
                """


# Streamlit code starts here
st.set_page_config(page_title='Customer Email Assistant 📧🤖', layout='wide')

st.title('Customer Email Assistant 📧🤖')

seller_email = st.text_area('Seller Email',
                            """Good morning Unfortunately the dealer who came to collect the car decided that they only wanted to offer me ¬£9800 when they arrived to collect the car and i was not willing to let it go for that amount.
There was 3 men who cam to collect it and it felt really intimidating and pressured.The car has not been collected.""")

if 'result' not in st.session_state:
    st.session_state['result'] = None

if st.button('Generate Response') or st.session_state.result is not None:
    st.session_state.result = llm(form_assistant_prompt(seller_email))

    sections = st.session_state.result.split('****')
    for section in sections[1:]:
        title, _, rest = section.partition(':')
        st.subheader(title)
        st.session_state.text_box = st.text_area('', rest.strip())

st.divider()

st.subheader('Check Response:')
draft_response = st.text_area('Draft your response here')

if st.button('Validate Response'):
    summary_and_context, _, _ = st.session_state.result.rpartition('****')
    response = llm(form_validation_prompt(summary_and_context, draft_response))
    title, _, rest = response.partition(':')
    st.subheader(title)
    st.text_area('', rest)




