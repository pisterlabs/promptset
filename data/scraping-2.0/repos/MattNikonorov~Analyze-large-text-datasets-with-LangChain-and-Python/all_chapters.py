from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chains.openai_functions import create_structured_output_chain
import os
import json
import time
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.3)
template = """Use the given format to extract information from the following input: {input}. Make sure to answer in the correct format"""
prompt = PromptTemplate(template=template, input_variables=["input"])

json_schema = {
    "type": "object",
    "properties": {
        "summary": {"title": "Summary", "description": "The chapter summary", "type": "string"},
        "messages": {"title": "Messages", "description": "Philosophical messages", "type": "string"},
        "ethics": {"title": "Ethics", "description": "Ethical theories and moral principles presented in the text", "type": "string"}
    },
    "required": ["summary", "messages", "ethics"]
}

chain = create_structured_output_chain(json_schema, llm, prompt, verbose=False)
f = open("texts/Beyond Good and Evil.txt", "r")
phi_text = str(f.read())
chapters = phi_text.split("CHAPTER")

for chi in range(1, len(chapters), 1):
    with get_openai_callback() as cb:
        ch = chain.run(chapters[chi])
        print(cb)
        print("\n")

    print(ch)
    print("\n\n")
    json_object = json.dumps(ch, indent=4)

    if chi == 1:
        with open("Beyond Good and Evil.json", "w") as outfile:
            outfile.write("[\n"+json_object+",")
    elif chi < len(chapters)-1:
        with open("Beyond Good and Evil.json", "a") as outfile:
            outfile.write(json_object+",")
    else:
        with open("Beyond Good and Evil.json", "a") as outfile:
            outfile.write(json_object+"\n]")
    # time.sleep(20) if you face rate limits
