from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chains.openai_functions import create_structured_output_chain
import os
import json
import time
import glob
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.3)
template = """Use the given format to extract information from the following input: {input}. Make sure to answer in the correct format. If a value is not found, set the value as 'not found'"""
prompt = PromptTemplate(template=template, input_variables=["input"])

json_schema = {
    "type": "object",
    "properties": {
        "age": {"title": "Age", "description": "The age of the player as of 2023", "type": "integer"},
        "height": {"title": "Height", "description": "Height of player in centimeters", "type": "integer"},
        "serve": {"title": "Serve", "description": "The player's fastest serve in kmph. If given in mph, convert to kmph", "type": "integer"}
    },
    "required": ["age", "height", "serve"]
}
chain = create_structured_output_chain(json_schema, llm, prompt, verbose=False)
players = glob.glob("top_10_tennis_players/*") # Reading your document directory

for pi in range(len(players)):
    f = open(f"{players[pi]}", "r")
    player_text = str(f.read())

    # Start of highlighted code

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=16000,
        chunk_overlap=2000,
        length_function=len,
        add_start_index=True,
    )
    sub_texts = text_splitter.create_documents([player_text])
    ch = []
    for ti in range(len(sub_texts)):
        with get_openai_callback() as cb:
            ch.append(chain.run(sub_texts[ti]))
            print(ch[-1])
            print(cb)
            print("\n")
            # time.sleep(10) if you hit rate limits

    for chi in range(1, len(ch), 1):
        if (ch[chi]["age"] > ch[0]["age"]) or (ch[0]["age"] == "not found" and ch[chi]["age"] != "not found"):
            ch[0]["age"] = ch[chi]["age"]
        if (ch[chi]["serve"] > ch[0]["serve"]) or (ch[0]["serve"] == "not found" and ch[chi]["serve"] != "not found"):
            ch[0]["serve"] = ch[chi]["serve"]
        if (ch[0]["height"] == "not found") and (ch[chi]["height"] != "not found"):
            ch[0]["height"] = ch[chi]["height"]
        else:
            continue

    print("\n\n")
    json_object = json.dumps(ch[0], indent=4)

    # End of highlighted code

    with open(f"{players[pi].replace('top_10_tennis_players', 'player_data').replace('.txt', '.json')}", "w") as outfile:
        outfile.write("[\n"+json_object+"\n]")

    # time.sleep(20) if you hit rate limits
