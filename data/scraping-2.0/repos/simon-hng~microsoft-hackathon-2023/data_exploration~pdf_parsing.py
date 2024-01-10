from langchain.document_loaders import PDFMinerPDFasHTMLLoader
from bs4 import BeautifulSoup
import json
from pathlib import Path
categories = ['Module Level:', 'Language:', 'Duration:', 
              'Frequency:', 'Credits:', 'Total Hours:', 
              'Self-study Hours:', 'Contact Hours:', 'Description of Examination Method:',
              'Repeat Examination:', '(Recommended) Prerequisites:', 'Content:', 'Intended Learning Outcomes:', 
              'Teaching and Learning Methods:', 'Media:', 'Reading List:', 'Responsible for Module:', 
              'Courses (Type of course, Weekly hours per semester), Instructor:']
pdf_path = pdf_path = Path("/Users/pauldelseith/Documents/microsoft-hackathon-2023/Data/Modulhandbuecher/Modulehandbook_MBA Executive Master of Business Administration in Business & IT.pdf")
loader = PDFMinerPDFasHTMLLoader(str(pdf_path))
data = loader.load()[0]   # entire PDF is loaded as a single Document
page_count = 32

# extract the content of the page by getting the outer divs
soup = BeautifulSoup(data.page_content,'html.parser')
content = soup.find_all('div')

# filter out Alphabetical Index
for idx, element in enumerate(content):
    if 'Alphabetical Index' in str(element):
        content = content[:idx]
        break

# extract the module descriptions
module_descriptions = {}
current_module_name = ""
for idx, element in enumerate(content):
    if 'Module Description' in str(element):
        module_name = content[idx+1].text
        module_name = module_name.replace('\n', '')
        module_descriptions[module_name] = []
        current_module_name = module_name
    elif current_module_name != "" and element.text.replace('\n', '') != current_module_name:
        module_descriptions[current_module_name].append(element)

# extract the categories in every module description
module_descr_classified = {}
page_break_detected = False
page_break_detected_idx = 0
for key, value in module_descriptions.items():
    module_descr_classified[key] = {}
    extr_categorie = ""
    for idx, val in enumerate(value):
        if any(categorie in val.text for categorie in categories):
            txt_split = val.text.split(":")  
            extr_categorie = txt_split[0]
            if extr_categorie == "Credits":
                # remove weird "*" for Credits
                extr_content = txt_split[1].replace("*","")
            else:
                extr_content = ''.join(txt_split[1:])
            module_descr_classified[key][extr_categorie] = extr_content.replace('\n', '')
            # check if page break is detected
            if idx + 2 < len(value) and f" of {page_count}" in value[idx+2].text:
                page_break_detected = True
                page_break_detected_idx = idx
            else:
                page_break_detected = False
        elif idx == page_break_detected_idx + 4 and page_break_detected:
            # add text after page break to the last category
            module_descr_classified[key][extr_categorie] = ''.join([module_descr_classified[key][extr_categorie]," ", val.text.replace('\n', '')])
        elif idx < (len(value) - 4) and extr_categorie == "Courses (Type of course, Weekly hours per semester), Instructor":
            # add text after Courses to the last category since there are weird breaks here
            module_descr_classified[key][extr_categorie] = ''.join([module_descr_classified[key][extr_categorie],", ", val.text.replace('\n', '')])
    

with open(f"{pdf_path.stem}.json", 'w', encoding="utf-8") as json_file:
    json.dump(module_descr_classified, json_file, ensure_ascii=False)
