import json
import requests
import PyPDF2
from transformers import AutoTokenizer
import json
import requests
import re
from langchain import HuggingFaceHub,LLMChain
from langchain.prompts import PromptTemplate

command = "You are an API that converts bodies of text into a single question and answer into a JSON format. Each JSON " \
          "contains a single question with a single answer. Only respond with the JSON and no additional text. \n"
hub_llm = HuggingFaceHub(repo_id='bigscience/bloom')
prompt = PromptTemplate(
    input_variables=['question'],
    template= command+" {question}"
)
hub_chain = LLMChain(prompt=prompt,llm=hub_llm,verbose=True)


API_URL = "https://api-inference.huggingface.co/models/gpt2"
API_TOKEN ='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'
headers = {"Authorization": f"Bearer {API_TOKEN}"}

tokenizer = AutoTokenizer.from_pretrained("nomic-ai/gpt4all-falcon")
history = {'internal': [], 'visible': []}

def run(user_input, history):
    print("initialising request......")
    request = {
        'user_input': user_input,
        'history': history,
    }
    print("sending request....")
    hub_chain.run(user_input)
    ##response = requests.post(API_URL, data=user_input,headers=headers)
    ##response=requests.request("POST", API_URL, headers=headers, data=request)
    print("request sent....")
    print(hub_chain.run(user_input))
    ##result = response.json()
    ##print(json.loads(response.json()['body']))
    ##print("response received....")
    ##print(result)
    ##return result

def extract_text_from_pdf(file_path):
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page_obj = pdf_reader.pages[page_num]
        text += page_obj.extract_text()
    pdf_file_obj.close()
    return text

def tokenize(text):
    enc = tokenizer.encode(text)
    return enc

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def is_json(data):
    try:
        json.loads(data)
        return True
    except ValueError:
        return False

def submit_to_api(chunk, retries=3):
    for i in range(retries):
        try:
            print(i)
            print("th try to communicate to llm")
            response = run(chunk.strip(), history)
            # Extract JSON string from between back-ticks
            if is_json(response):
                print(response)
                return json.loads(response)
            else:
                match = re.search(r'`(.*?)`', response, re.S)
                if match and is_json(match.group(1)):
                    print(f"Attempt {i + 1} failed. Retrying...")
                    return json.loads(match.group(1))  # assuming you want to return the JSON data
                else:
                    print("Request failed:")
                    print(response)
        except requests.exceptions.RequestException as e:
            continue
    print("Max retries exceeded. Skipping this chunk.")
    return None

print("Extracting Texts From PDF........")
text = extract_text_from_pdf('/home/jehu/Desktop/projs/power/read/(Synthese Library 220) Gordon Pask (auth.), Gertrudis van de Vijver (eds.) - New Perspectives on Cybernetics_ Self-Organization, Autonomy and Connectionism-Springer Netherlands (1992).pdf')
tokens = tokenize(text)

token_chunks = list(chunks(tokens, 256))
print("Done Tokenizing........")
responses = []
q=0
for chunk in token_chunks:
    q=q+1
    print(q)
    response = submit_to_api(tokenizer.decode(chunk))
    if response is not None:
        responses.append(response)
    else:
        print("Response is NON")

# Write responses to a JSON file
with open('responses.json', 'w') as f:
    json.dump(responses, f)