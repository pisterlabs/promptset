import json 
import os
import sys
import pandas
from datasets import load_dataset, Dataset
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

class Note(BaseModel):
    """Note generated from conversation"""
    note: str = Field(..., description="Clinical note generated from conversation")

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)['data']
    df = pandas.DataFrame(data)
    dataset = Dataset.from_dict(df)
    return dataset


if __name__ == "__main__":
    # Load dataset
    extraction_functions = [convert_pydantic_to_openai_function(Note)]
    dataset = load_json(file_path= "/mnt/c/Users/amangokrani/OneDrive - Microsoft/Personal/evaluations/aci-bench/data/challenge_data_json/valid.json")
    temp_path = "/mnt/c/Users/amangokrani/OneDrive - Microsoft/Personal/evaluations/temp/valid"
    
    model = ChatOpenAI(model = "gpt-4-1106-preview", temperature = 0.0, verbose = True)
    
    PROMPT = ChatPromptTemplate.from_template("""
    summarize the conversation to generate a clinical note with four sections: HISTORY OF PRESENT ILLNESS, PHYSICAL EXAM, RESULTS, ASSESSMENT AND PLAN. 
    The conversation is: {conversation}
    """)
    
    chain = PROMPT | model.bind(functions=extraction_functions) | JsonOutputFunctionsParser()
    
    notes = []
    for example in dataset:
        temp_file = os.path.join(temp_path, f"{example['file']}.json")
    
        # Check if the file already exists
        print("processing file: ", temp_file)
        if not os.path.exists(temp_file):
            print("calling LLM....")
            # Call invoke only if the file does not exist
            response = chain.invoke({"conversation": example["src"]})
        
            # Write the response to a file
            with open(temp_file, "w") as f:
                json.dump(response, f, indent=4)
        
        else: 
            print("Skipping LLM call....")
            # Load the file
            with open(temp_file, "r") as f:
                response = json.load(f)
        
        print("Done processing file: ", temp_file)
        notes.append(response["note"])
        
        
    dataset = dataset.add_column("pred", notes)
    dataset.to_json("/mnt/c/Users/amangokrani/OneDrive - Microsoft/Personal/evaluations/outputs/valid.gpt-4-1106-preview.pred.fullnote.json")
