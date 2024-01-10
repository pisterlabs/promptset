""" 
Accesses OpenAI 
"""
from openai import OpenAI
from datastructures import Analysis
def _break_text(full_text, length):
    """ Chunks text into as many pieces as necessary and returns the list """

def analyze_report(text, output="friendly"):
    """ 
    Creates an Analysis object from the given text. 
    If output is "friendly", will just return a list of chioices
    Otherwise, will return the full completion object
    """
    system_prompt = """
    You are an analysis engine for post-mortem events. Your purpose is to evaluate post-mortem 
    reports and provide valuable information to ensure they do not occur again. You respond only 
    in json documents with the following keys:
        "title": This is the title of the document
        "summary": Produce a text summary of the entire document
        "keywords": Produce a list of 10 key words from the document text
        "key_ideas": Produce a list of 3-5 short ideas from the document
        "contributing_factors": Produce a list of prevailing and contributing factors identified in the report
    """
    temperature = 3
    model = "gpt-3.5-turbo-16k"
    #model = "gpt-4-32k"
    #model = "gpt-4-1106-preview"
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"The following is a post-mortem report. Create a json document with the title, summary, keywords, prevailing factors, and key ideas. \n {text}"
        }
    ]
    print(f"Sending messages:")
    print(messages)
    client = OpenAI()
    chat_completion = client.chat.completions.create(
        model = model,
        messages = messages
    )
    
    if output == "friendly":
        return chat_completion.choices[0].message.content
    return chat_completion