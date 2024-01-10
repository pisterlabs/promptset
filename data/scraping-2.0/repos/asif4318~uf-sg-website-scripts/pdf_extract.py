import os
import pdfplumber
import json
import os
import openai
import re
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_bill_number(title):
    # Regular expression pattern to match the bill number format
    pattern = re.compile(r"\d\d\d\d-\d\d\d\d", re.IGNORECASE)
    match = re.search(pattern, title)
    if match:
        return "SSB " + match.group()
    else:
        return title


def extract_beginning(pdf_path: str):
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[0].extract_text()
        if len(pdf.pages) > 1:
            text = text + "\n" + pdf.pages[1].extract_text()
        truncated = (text[:800]) if len(text) > 800 else text
        return truncated


def generate_message(content: str):
    return [
        {
            "role": "system",
            "content": '''You are a helpful assistant that returns to me properly formatted json objects in the format \n{\"id\": \"\", \"title\":  \"\",  \"author\": \"\", \"sponsor\": \"\", \"summary\": \"\",  \"status\": \"\"} extracted from the text I provide. Id is at the beginning of the text in the format \"XXXX-XXXX\" where X is an integer. Summary is a 100 word max summary that does not include authors or sponsors in the summary. Do not include any special escaping characters such as line breaks.
            If the data includes 3000 J. Wayne Reitz Union PO ... or similar, ensure the "status" is "PASSED". Else the "status" property is "TBD".
            This summary is bad: Resolution Celebrating 50 Years of Women’s Athletics at the University of Florida. Sponsored by Senator Jonathan Stephens, Senator Oscar Santiago, Senator Raj Mia, Senator Catherine Gomez, Senator Taylor Hoerle, Senator Isabelle Gerzenshtein, Senator Hana Ali, Senator Savanah Partridge, Deputy Minority Party Leader Mohammed Faisal, Member-at-Large Jacey Cable, Judiciary Vice-Chair Mason Solomon, Senator Bronson Allemand, Senator Saketh Damera, Senator Jacob Ka.
            This summary is good: The University of Florida Student Senate acknowledges the remarkable achievements of the Women's Athletics program, which has produced 92 Olympians earning a total of 64 Olympic medals. In recognition of the program's 50th anniversary and the pioneering efforts of Dr. Ruth Alexander, Donna Deutsch, Linda Hall Thornton, and Mimi Ryan in advocating for Women's Athletics in 1972, the Student Senate honors their contributions. Additionally, the Senate expresses admiration for the female athletes representing the Florida Gators, applauding their dedication, perseverance, and commitment to the university. Lastly, the University of Florida Student Senate celebrates the 50th anniversary of the Women's Athletics program at the university. 
            This summary is bad: This bill, authored by Judiciary Chairman John Brinkman, aims to modernize and reform Senate meetings. It has several sponsors, including Judiciary Vice-Chairman Mason Solomon, Senator Mara Vaknin, Senator Julia Haley, Senator Taylor Soukup, Member-at-Large Jacey Cable, Senator Jagger Leach, and Senator Sidney Ruedas. The bill proposes amendments to Rule I, which governs the officers of the Senate. One of the key changes is the process for electing the Senate President, which would occur at the first meeting following the validation of Senate election results. The bill seeks to bring efficiency and transparency to Senate meetings.
            This summary is good: This bill, authored by Judiciary Chairman John Brinkman, aims to modernize and reform Senate meetings by proposing amendments to Rule I, which governs the officers of the Senate. The key change includes a revised process for electing the Senate President immediately after validating Senate election results. The bill's objective is to enhance efficiency and transparency in Senate meetings.
            This is a bad summary: "Each February commemorates Black History Month, a period which honors and appreciates the rich culture, history, and contributions of Black and African Americans throughout their continuous str", it is not fully complete
            Summaries must be full senetences
            '''
        },
        {
            "role": "assistant",
            "content": content,
        }
    ]


def generate_message_second(content: str):
    return [
        {
            "role": "system",
            "content": '''Please return a PROPERLY formatted JSON string, your last response was not properly formatted. The json should be parseable by python. You are a helpful assistant that returns to me properly formatted json objects in the format \n{\"id\": \"\", \"title\":  \"\",  \"author\": \"\", \"sponsor\": \"\", \"summary\": \"\",  \"status\": \"TBD\"} extracted from the text I provide. Id is at the beginning of the text in the format \"XXXX-XXXX\" where X is an integer. Summary is a 100 word max summary that does not include authors or sponsors in the summary. Do not include any special escaping characters such as line breaks.  
            This summary is bad: Resolution Celebrating 50 Years of Women’s Athletics at the University of Florida. Sponsored by Senator Jonathan Stephens, Senator Oscar Santiago, Senator Raj Mia, Senator Catherine Gomez, Senator Taylor Hoerle, Senator Isabelle Gerzenshtein, Senator Hana Ali, Senator Savanah Partridge, Deputy Minority Party Leader Mohammed Faisal, Member-at-Large Jacey Cable, Judiciary Vice-Chair Mason Solomon, Senator Bronson Allemand, Senator Saketh Damera, Senator Jacob Ka.
            This summary is good: The University of Florida Student Senate acknowledges the remarkable achievements of the Women's Athletics program, which has produced 92 Olympians earning a total of 64 Olympic medals. In recognition of the program's 50th anniversary and the pioneering efforts of Dr. Ruth Alexander, Donna Deutsch, Linda Hall Thornton, and Mimi Ryan in advocating for Women's Athletics in 1972, the Student Senate honors their contributions. Additionally, the Senate expresses admiration for the female athletes representing the Florida Gators, applauding their dedication, perseverance, and commitment to the university. Lastly, the University of Florida Student Senate celebrates the 50th anniversary of the Women's Athletics program at the university. 
            This summary is bad: This bill, authored by Judiciary Chairman John Brinkman, aims to modernize and reform Senate meetings. It has several sponsors, including Judiciary Vice-Chairman Mason Solomon, Senator Mara Vaknin, Senator Julia Haley, Senator Taylor Soukup, Member-at-Large Jacey Cable, Senator Jagger Leach, and Senator Sidney Ruedas. The bill proposes amendments to Rule I, which governs the officers of the Senate. One of the key changes is the process for electing the Senate President, which would occur at the first meeting following the validation of Senate election results. The bill seeks to bring efficiency and transparency to Senate meetings.
            This summary is good: This bill, authored by Judiciary Chairman John Brinkman, aims to modernize and reform Senate meetings by proposing amendments to Rule I, which governs the officers of the Senate. The key change includes a revised process for electing the Senate President immediately after validating Senate election results. The bill's objective is to enhance efficiency and transparency in Senate meetings.
            This is a bad summary: "Each February commemorates Black History Month, a period which honors and appreciates the rich culture, history, and contributions of Black and African Americans throughout their continuous str", it is not fully complete
            Summaries must be full senetences
            '''
        },
        {
            "role": "assistant",
            "content": content,
        }
    ]


def get_gpt_info(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message,
        temperature=1,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response["choices"][0]["message"]["content"]

####### Driver Code ##############


# pdf_folder = 'bills-converted'
pdf_folder = 'test'

results = []
error_paths = []

for filename in os.listdir(pdf_folder):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, filename)
        message = generate_message(extract_beginning(pdf_path))
        print("Extraction complete: " + filename)
        print("="*40)

        bill_info = get_gpt_info(message)
        try:
            bill_as_json = json.loads(bill_info)
            bill_as_json["id"] = extract_bill_number(filename)
            results.append(bill_as_json)
        except:
            print("Error " + filename)
            error_paths.append(filename)
        print(len(results))

for filename in error_paths:
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, filename)
        message = generate_message_second(extract_beginning(pdf_path))
        print("Extraction complete: " + filename)
        print("="*40)

        bill_info = get_gpt_info(message)
        try:
            bill_as_json = json.loads(bill_info)
            bill_as_json["id"] = extract_bill_number(filename)
            results.append(bill_as_json)
        except:
            print("Error " + filename)
            error_paths.append(filename)
        print(len(results))

# Save results to a JSON file

with open('bill_results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Results saved to bill_results.json")
