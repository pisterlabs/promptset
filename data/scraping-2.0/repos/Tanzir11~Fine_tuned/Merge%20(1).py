import transformers
import torch
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
# from huggingface_hub import login
from transformers import AutoTokenizer

# Hardcode the Hugging Face token in the code
api_token = "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"

# Function to get the response back
def getLLMResponse(input_data, template):
    model = "myparalegal/finetuned-Merger_Agreement"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        # api_key=api_token,
        torch_dtype=torch.float16,
        device=0, # Set device to 0 to run on the GPU
        temperature=0.5,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=1000,  # max number of tokens to generate in the output
        repetition_penalty=1.2
    )
    llm = HuggingFacePipeline(pipeline=pipeline)

    # Creating the final PROMPT
    prompt = PromptTemplate(
        input_variables=["Party A Name", "Party A Address", "Party B Name", "Party B Address", "Merger Detail", "Jurisdiction"],
        template=template,
    )

    # Generating the response using LLM
    response = llm(prompt.format(**input_data))
    return response[0]["generated_text"]

# Streamlit app
st.set_page_config(page_title="Merger Contract Generator",
                    page_icon='🤝',
                    layout='centered',
                    initial_sidebar_state='collapsed')
st.header("Merger Contract Generator 🤝")

# Take input from the user
input_data = {
    "Party A Name": st.text_input("Party A Name"),
    "Party A Address": st.text_input("Party A Address"),
    "Party B Name": st.text_input("Party B Name"),
    "Party B Address": st.text_input("Party B Address"),
    "Details of the merger transaction": st.text_area("Merger Detail", height=100),
    "Jurisdiction": st.text_input("Jurisdiction")

}

submit = st.button("Generate Merger Contract")

# When 'Generate' button is clicked, execute the below code
if submit:
    # Template for building the PROMPT
    template = """MERGER CONTRACT

This Agreement is made and entered into as of [Effective Date], by and between:

Party A: {Party A Name}, Address: {Party A Address}

and

Party B: {Party B Name}, Address: {Party B Address}

The Parties desire to enter into a merger transaction (the "Merger") on the terms and conditions set forth herein.

NOW, THEREFORE, in consideration of the mutual covenants contained herein, the Parties agree as follows:

1. Merger Transaction: {Merger Detail}

2. Governing Law: {Jurisdiction}

IN WITNESS WHEREOF, the Parties have executed this Merger Contract as of the date first above written.
    """

    response = getLLMResponse(input_data, template)
    st.subheader("Generated Merger Contract")
    st.write(response)
