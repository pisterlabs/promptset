import openai
import requests
import json
import wolframalpha
import os


BAD_INPUT_MESSAGE = "Incompatible input/input cannot be read"

def imageToAnswer(image):
    imageText = requests.post("https://api.mathpix.com/v3/text",
        files={"file": open(image,"rb")},
        data={
        "options_json": json.dumps({
            "math_inline_delimiters": ["$", "$"],
            "rm_spaces": True
        })
        },
        headers={
            "app_id": os.environ.get("MATHPIX_APP_ID"),
            "app_key": os.environ.get("MATHPIX_API_KEY")
        }
    )
    if ("text" in imageText.json()):
        return json.dumps(imageText.json()["text"], indent=4, sort_keys=True)
    return BAD_INPUT_MESSAGE


def wolframLatexParser(text):
    if (text == BAD_INPUT_MESSAGE):
        return BAD_INPUT_MESSAGE
    
    if (len(text) < 5):
        return BAD_INPUT_MESSAGE
    
    parsedText = ""
    if (text[0:4] == r"$\\"):
        text = text = text[4:]
    
    textPointer = 0
    
    while (textPointer < len(text)):
        if (textPointer != len(text) - 3 and text[textPointer] == '\\' 
            and text[textPointer + 1] == '\\' and text[textPointer + 2] == '\\' and text[textPointer + 3] == '\\'):
            parsedText += "; "
            textPointer += 4
            
        elif (textPointer != len(text) - 1 and text[textPointer] == r"{" and text[textPointer + 1] == r"}"):
            parsedText += "; "
            textPointer += 2
        
        elif (textPointer != len(text) - 1 and text[textPointer] == ", "):
            textPointer += 1

        elif (text[textPointer] == '\\' and ((textPointer != len(text) - 1 and text[textPointer + 1] == '\\') or (textPointer == len(text) - 1)) ):
            textPointer += 2

        elif (textPointer < len(text) - 5 and text[textPointer: textPointer + 5] == "begin"):
            while (text[textPointer] != '}'):
                textPointer += 1
            textPointer += 1
        
        elif (textPointer < len(text) - 3 and text[textPointer: textPointer + 3] == "end"):
            while (text[textPointer] != '}'):
                textPointer += 1
            textPointer += 1
        
        elif (textPointer < len(text) - 3 and text[textPointer: textPointer + 3] == r"{l}"):
            textPointer += 3

        elif (textPointer < len(text) - 3 and text[textPointer: textPointer + 3] == r"{c}"):
            textPointer += 3
            
        elif (text[textPointer] == "$"):
            textPointer += 1

        elif (text[textPointer] == r'"'):
            textPointer += 1

        elif (text[textPointer] == "{"):
            textPointer += 1
        
        elif (text[textPointer] == "}"):
            textPointer += 1

        else:
            parsedText += text[textPointer]
            textPointer += 1

    return parsedText


def getWAanswer(text):
    if text == BAD_INPUT_MESSAGE:
        return BAD_INPUT_MESSAGE

    client = wolframalpha.Client(os.environ.get("WOLFRAMALPHA_APP_ID"))
    result = client.query(text)
    if result.success:
        textOutput = next(result.results).text
        return textOutput
    return "No Wolfram Alpha solution is available."


def getAIAnswer(question, studentAnswer, wolframAnswer, client):
    context = """You are a math teacher. Your job is ONLY to look over a question and the process that has been written so far.  
    It is identical to grading a math test. Be strict. You will be given a step-by-step solution labeled "Solution" as an answer key. 
    Use this answer key to identify where a mistake was made.  If the steps or answer is incorrect point out the step at which a mistake was made.  
    You should say: "Incorrect. Hint: What [operation] must you do with [step] to get [answer]?" 
    If the answer is correct based on the answer key, say "This is correct!". 
    Remember, grade only up to what the student has done based on the solution. 
    Do not give the right answer, simply guide the process. 
    If you are seeing similar mistakes give "better" hints. 
    Be quirky and friendly with your hints. Remember to cross reference with "Solution", using that as the only data base. 
    If the FINAL answer doesn't match "solution", it is incorrect"""

    messages = [ {"role": "system", "content": context} ]

    messages.append(
        {"role": "user", "content": context + "\n Here are the statements: \n\
        Question: " + question + "\n Student answer: " + studentAnswer + "\n Correct Answer: " + wolframAnswer})
    
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages
    )
    
    reply = chat.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    return reply
