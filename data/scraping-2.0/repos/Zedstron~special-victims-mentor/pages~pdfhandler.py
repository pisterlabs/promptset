import re
import json
import openai
import vertexai
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdftypes import resolve1
from vertexai.preview.language_models import ChatModel, InputOutputTextPair

fieldset = [b'Date of assault(s)']
rapedate = "5/1/2023"
stateoccurred = "California"
MODEL_OPEN_AI = "gpt-3.5-turbo"

def ExtractJsonObject(rawcontent):
    pattern = r"\[.*\]"
    matches = re.findall(pattern, rawcontent, re.DOTALL)

    return matches[0] if matches else False

def Handler(model):
    handlers = {
        "VertexAI": GetVertexResult,
        "OpenAI": GetOpenAIResult
    }
    return handlers[model]

def GeneratePrompt(model):
    availablePrompts = {
        "VertexAI": f"""
        An unfortunate incident occurred to a girl on {rapedate}.
        According to the law in {stateoccurred}, what are the time limits for filing charges?
        Generate a list of calendar events that outline the deadlines for each legal action that can be taken, starting from {rapedate}.
        Include the appropriate institutions to contact and instructions on presenting her case to them in the event descriptions,
        and don't forget to Output the information as a List of JSON objects including keys as title, description, institution, deadline, instructions, start and end'.
        """,
        "OpenAI": f"""
        A girl was raped on {rapedate}, 
        when does she have to press charges based on {stateoccurred} law? 
        Export as a list of calendar events spanning the maximum dates of opportunities for each legal action that can be taken starting from {rapedate}. 
        For the description of each event, include the relevant institutions that should be engaged and how to present her case to them. 
        Output as a JSON object. """
    }
    return availablePrompts[model]

def extract_form_fields(pdf_path):
    with open(pdf_path, 'rb') as file:
        parser = PDFParser(file)
        doc = PDFDocument(parser)
        global rapedate
        form_fields = {}

        if 'AcroForm' in doc.catalog:
            fields = resolve1(doc.catalog['AcroForm']).get('Fields', [])
            for field in fields:
                field = resolve1(field)
                name, value = field.get('T'), field.get('V')
                if name in fieldset:
                    form_fields[name] = value
                    if name == b'Date of assault(s)':
                        rapedate = value

        return form_fields

def GetOpenAIResult(prompt):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_OPEN_AI,
            messages=[
                {"role": "system", "content": prompt},
            ],
            temperature=0,
        )

        return json.loads(response["choices"][0]["message"]["content"])
    except Exception as e:
        print(str(e))
        return False

def GetVertexResult(prompt):
    vertexai.init(project="rational-photon-392301", location="us-central1")
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 1010,
        "top_p": 0.8,
        "top_k": 40
    }
    chat = chat_model.start_chat(
        context="""Hello""",
    )
    response = chat.send_message(prompt, **parameters)
    print(response)
    response = ExtractJsonObject(response.text)
    return json.loads(response.strip()) if response else False

def GetResult(filename, model):
    try:
        extract_form_fields(filename)
        prompt = GeneratePrompt(model)
        return Handler(model)(prompt)
    except Exception as e:
        print("Exception generated", e)
        return False
