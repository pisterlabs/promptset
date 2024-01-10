from getReport import getReport
from langchain.llms import Ollama
import ast
import json

MODEL = "openhermes2.5-mistral:7b-fp16"

model = Ollama(
    base_url="http://localhost:11434",
    model=MODEL,
    temperature=0.75,
    stop=["<|im_end|>"],
)


def uniList_Data(uniList):
    university_data = {}
    uniList = ast.literal_eval(uniList)
    for uni in uniList:
        prompt = f"""
        Given the university name, write two lines about the university. 
        University Name = {uni}
        """
        output = model(prompt).strip()
        university_data[uni] = output
    with open("./University_Data.txt", "w") as f:
        for key, value in university_data.items():
            f.write(f"\n{key}:\n{value}\n")
