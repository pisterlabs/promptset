import transformers
import torch
from flask import Flask, request, render_template_string

from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer

app = Flask(__name__)

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

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_data = {
            "Party A Name": request.form.get("Party_A_Name"),
            "Party A Address": request.form.get("Party_A_Address"),
            "Party B Name": request.form.get("Party_B_Name"),
            "Party B Address": request.form.get("Party_B_Address"),
            "Details of the merger transaction": request.form.get("Merger_Detail"),
            "Jurisdiction": request.form.get("Jurisdiction")
        }

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
        return render_template_string("<h1>Generated Merger Contract</h1><p>{{ response }}</p>", response=response)

    return render_template_string(
        """
        <h1>Merger Contract Generator 🤝</h1>
        <form method="post">
            <label for="Party_A_Name">Party A Name:</label><br>
            <input type="text" id="Party_A_Name" name="Party_A_Name"><br>
            <label for="Party_A_Address">Party A Address:</label><br>
            <input type="text" id="Party_A_Address" name="Party_A_Address"><br>
            <label for="Party_B_Name">Party B Name:</label><br>
            <input type="text" id="Party_B_Name" name="Party_B_Name"><br>
            <label for="Party_B_Address">Party B Address:</label><br>
            <input type="text" id="Party_B_Address" name="Party_B_Address"><br>
            <label for="Merger_Detail">Details of the merger transaction:</label><br>
            <textarea id="Merger_Detail" name="Merger_Detail" rows="4" cols="50"></textarea><br>
            <label for="Jurisdiction">Jurisdiction:</label><br>
            <input type="text" id="Jurisdiction" name="Jurisdiction"><br><br>
            <input type="submit" value="Generate Merger Contract">
        </form>
        """
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0')
