# Barcode-scanner and Open Food Facts API
from fastapi import FastAPI, UploadFile, File
from pyzbar.pyzbar import decode
from PIL import Image
from io import BytesIO
import requests
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
import os
import uvicorn
import re
import datetime

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "ASK SHERVIN"
api="ASK SHERVIN"

# Custom built LLM 1
llm1 = OpenAI(openai_api_key=api, temperature=0.7)
template1 = """You are a service that offers information about average expiration dates about food items based on these sources: The USDA Food Safety and Inspection Service, StillTasty and EatByDate. 
Question: What is the average expiration of {text}. Give out just a specific range of dates.(Like number of weeks or dates to go)
Answer: Has to be in in the following format and nothing else should pop up: number - number weeks.
"""
prompt_template1 = PromptTemplate(input_variables=["text"], template=template1)
answer_chain1 = LLMChain(llm=llm1, prompt=prompt_template1)

# Custom built LLM 2
llm2 = OpenAI(openai_api_key=api, temperature=0.7)
template2 = """You are a service that says if a fruit is a fresh food(perishable) or non perishable. 
Question: {text}
Answer: Single word answer only, so either fresh or no fresh
"""
prompt_template2 = PromptTemplate(input_variables=["text"], template=template2)
answer_chain2 = LLMChain(llm=llm2, prompt=prompt_template2)

# Custom built LLM 3
llm3 = OpenAI(openai_api_key=api, temperature=0.7)
template3 = """You are a service that gives enviornmental facts using credible information online and open food fact api. Give information like carbon footprint and packaging of the food and if the food has positive or negative impact. 
Question: What is the enviornmental impact of {text} and give me specific facts like carbon footprint(specificemissions stats) and packaging about that food item as well.
Answer: Just 2-3 short sentences. Like in total, there should be around 15-20 words in total.
"""
prompt_template3 = PromptTemplate(input_variables=["text"], template=template3)
answer_chain3 = LLMChain(llm=llm3, prompt=prompt_template3)

app = FastAPI()

# Define the FastAPI endpoint for barcode decoding and data extraction
@app.post("/decode_barcode")
async def decode_barcode(file: UploadFile = File(...)):
    # Open and process the uploaded image
    image = Image.open(BytesIO(await file.read()))
    gray = image.convert("L")

    # Decode barcodes in the image
    barcodes = decode(gray)
    decoded_barcodes = []

    # Iterate through decoded barcodes
    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")

        # Use the LLMs to get product details
        name = get_product_details(barcode_data, 1)
        expiry = get_product_details(barcode_data, 2)
        fresh_status = answer_chain2.run(name)
        facts = answer_chain3.run(name)

        decoded_barcodes.append({
            "name": name,
            "barcode_id": barcode_data,
            "expiry": expiry,
            "fresh or no fresh": fresh_status,
            "facts": facts
        })

    return {"barcodes": decoded_barcodes}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)

#uvicorn app.py --host 0.0.0.0 --port 8000
#python -m uvicorn main.py --host 0.0.0.0 --port 8000
