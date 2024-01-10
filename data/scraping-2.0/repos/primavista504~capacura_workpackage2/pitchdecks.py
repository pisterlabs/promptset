from langdetect import detect
import openai
import json
import os
import pytesseract
from tqdm import tqdm
import os
try:
 from PIL import Image
except ImportError:
 import Image

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
openai.api_key = "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"
pdf_path ='C:/Users/joevf/Documents/Capacura/Pitchdecks/Archiv/'

def pdf2im(images, n_images):
    for i in range(n_images):
      
          # Save pages as images in the pdf
        if i < 10:
            images[i].save('pdf_pages/page0'+ str(i) +'.png', 'PNG')
        else:
            images[i].save('pdf_pages/page'+ str(i) +'.png', 'PNG')

def read_pdf_img(rel_pdf_path):
    pitchdeck = []
    for i in range(len(os.listdir(rel_pdf_path))):
        if i < 10:
            image_path= rel_pdf_path+"page0"+str(i)+".png"
        else:
            image_path= rel_pdf_path+"page"+str(i)+".png"
        extractedInformation = pytesseract.image_to_string(Image.open(image_path))
        pitchdeck.append(extractedInformation)
    pitchdeck = '\n___________________\n'.join(pitchdeck)
    return pitchdeck

def extract_information(pitchdeck,company_name,error_message=''):
  output_format = """ 
  {
    "company_name": "",
    "company_city": "",
    "company_country": "",
    "TAM": "",
    "SAM": "",
    "SOM": "",
    }"""

  query_en = """
  Above there is a PDF pitchdeck for a company, each page is separated by underscores.
  Filter out the company name, city and country where their company is located.
  Also return the TAM (Total Addressable Market), SAM (Serviceable Available Market), SOM (Serviceable Obtainable Market).
  Fill entries which you don't find with "Unknown".
  Return everything in a JSON Dictionary in the format as shown below.
  """

  query_de = """
  Oben sehen Sie ein PDF-Pitchdeck für ein Unternehmen, jede Seite ist durch Unterstriche getrennt.
   Filtern Sie den Firmennamen, die Stadt und das Land heraus, in dem sich das Unternehmen befindet.
   Geben Sie außerdem den TAM (Total Addressable Market), den SAM (Serviceable Available Market) und den SOM (Serviceable Obtainable Market) zurück.
   Füllen Sie Einträge, die Sie nicht finden, mit „Unbekannt“ aus.
   Geben Sie alles in einem JSON-Wörterbuch im unten gezeigten Format zurück.
  """

  prompt = pitchdeck
  
  language = detect(prompt)
  if language == 'de':
    print('Detected text language: German')
    query = query_de
  elif language == 'en':
    print('Detected text language: English')
    query = query_en
  else:
    print('Language not found...')
    language = 'en'
    print('...Continuing in English')
  try:
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      #model = "gpt-4",
      messages=[
            {"role": "system", "content": f"You are a helpful business analyst."},
            #{"role": "assistant", "content": f"{assistance}"},
            {"role": "user", "content": f"\n{prompt}\n{query}\n\n{output_format}"}
        ],
        presence_penalty=0,
        #max_tokens = 4000,
        frequency_penalty = 0,
        top_p = 1,
        temperature = 0.3,
    )
  
  except openai.error.InvalidRequestError as e:
    try:
      print(e)
      response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        #model = "gpt-4",
        messages=[
              {"role": "system", "content": f"You are a helpful business analyst. With many years of business analysis experience, you can analyze useful information from any business files."},
              #{"role": "assistant", "content": f"{assistance}"},
              {"role": "user", "content": f"\n{prompt}\n{query}\n{output_format}"}
          ],
          presence_penalty=0,
          #max_tokens = len(query.split()),
          frequency_penalty = 0,
          top_p = 1,
          temperature = 0.3,
      )
    except:
      print(e)
      print(len(prompt))
      prompt = prompt[:(3*14000)-len(query)-len(output_format)]
      print(len(prompt))
      response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        #model = "gpt-4",
        messages=[
              {"role": "system", "content": f"You are a helpful business analyst. With many years of business analysis experience, you can analyze useful information from any business files."},
              #{"role": "assistant", "content": f"{assistance}"},
              {"role": "user", "content": f"\n{prompt}\n{query}\n\n{output_format}\n{error_message}"}
          ],
          presence_penalty=0,
          #max_tokens = len(query.split()),
          frequency_penalty = 0,
          top_p = 1,
          temperature = 0.3,
      )
  except Exception as e:
     print(e)
     return ''
     
  output_dict = response.choices[0].message.content.strip()

  print(output_dict)
  return output_dict,language
  
  
def complete_dictionary(error_dict):

    error_prompt = "Above you see a dictionary that is incomplete. Leave all information as it is and return a complete and correct JSON dictionary."
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      #model = "gpt-4",
      messages=[
            {"role": "user", "content": f"{error_dict}\n{error_prompt}"}
        ],
        presence_penalty=0,
        #max_tokens = len(query.split()),
        frequency_penalty = 0,
        top_p = 1,
        temperature = 0.3,
    )
    output_dict = response.choices[0].message.content.strip()
    return output_dict