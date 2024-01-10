import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import PyPDF2
import openai
import json
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StdOutCallbackHandler
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options


# Constants and Global Configurations
FIREFOX_BINARY_PATH = '/opt/firefox/firefox'
GECKODRIVER_PATH = '/usr/bin/geckodriver'
GECKODRIVER_LOG_PATH = '/geckodriver.log'
DRUCKSACHEN_DIR = 'Drucksachen'
MODEL_NAME = 'gpt-4-0314'
FRAGENKATALOG_FILE = 'fragenkatalog.json'
RESULTS_FILE = 'results.txt'


# Setup Firefox configurations
def get_firefox_configuration():
    options = Options()
    
    # Set binary location for Firefox
    #options.binary_location = FIREFOX_BINARY_PATH
    
    # Set command line arguments, e.g. for headless mode
    options.add_argument("-headless")
    
    return options

# Set OpenAI configuration
def set_openai_config():
    openai.organization = os.getenv("OPENAI_ORGANIZATION")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    check_model_availability()

# Check if the desired model is available
def check_model_availability():
    model_list = openai.Model.list()['data']
    model_ids = [x['id'] for x in model_list]
    if MODEL_NAME not in model_ids:
        print(f'Model {MODEL_NAME} is not available.')
        exit()

# Scrape and download documents
def extract_info(driver):
    # Extract information from the Übersicht section
    uebersicht = driver.find_element(By.ID, 'content-übersicht')
    initiative = uebersicht.find_element(By.XPATH, '//label[text()="Initiative:"]/following-sibling::span').text
    beratungsstand = uebersicht.find_element(By.XPATH, '//label[text()="Beratungsstand:"]/following-sibling::span').text
    
    # Extract information from the Wichtige Drucksachen and Plenum sections
    wichtige_drucksachen = []
    plenum = []
    documents = driver.find_elements(By.XPATH, '//label[text()="Wichtige Drucksachen"]/following-sibling::ul/li')

    for doc in documents:
        date = doc.find_element(By.XPATH,'./div/div').text
        title = doc.find_element(By.XPATH,'./div/div/a').text
        link = doc.find_element(By.XPATH,'./div/div/a').get_attribute('href')
        if 'BT-Drucksache' in title:
            wichtige_drucksachen.append({'date': date, 'title': title, 'link': link})
        elif 'BT-Plenarprotokoll' in title:
            plenum.append({'date': date, 'title': title, 'link': link})
    
    return {
        'initiative': initiative,
        'beratungsstand': beratungsstand,
        'wichtige_drucksachen': wichtige_drucksachen,
        'plenum': plenum
    }

def download_file(url, date):
    # if folder not empty, delete all files
    if os.listdir('Drucksachen'):
        for file in os.listdir('Drucksachen'):
            os.remove(os.path.join('Drucksachen', file))

    
    doc_type = date.split('(')[1].split()[0]
    local_filename = f'Drucksachen/{doc_type}.pdf'
    
    # Create the Drucksachen folder if it doesn't exist
    if not os.path.exists('Drucksachen'):
        os.makedirs('Drucksachen')
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None
    
    return local_filename

# Process each document file
def process_documents():
    with open('fragenkatalog.json', 'r', encoding='utf-8') as file:
        fragenkatalog = json.load(file)

    document_files = [f for f in os.listdir('Drucksachen') if f.endswith('.pdf')]

    handler = StdOutCallbackHandler()
    llm = ChatOpenAI(temperature=0, model='gpt-4-0314', streaming=True)

    template = ChatPromptTemplate.from_messages([
        ("system", "Du bist juristischer Referent des Bundestages."),
        ("human", "Bitte beantworte diesen Fragenkatalog zu dem angehängten Dokument in angemessener Knappheit. Um die Fragen zu beantworten arbeite bitte in Stichpunkten."),
        ("ai", "Alles klar, was sind die Fragen?"),
        ("human", "Die Fragen: {questions}. \n\nSei bitte so konkret wie möglich. Bei der Kritischen Perspektive zu Rhetorik und Stilmitteln bitte die Begriffe und die Kritikpunkte daran kurz aufschreiben. "),
        ("ai", "Okay, was ist das Dokument?"),
        ("human", "Das Dokument: {document}")
        ,
    ])

    chain = LLMChain(llm=llm, prompt=template, callbacks=[handler])

    all_results = ""
    for document_file in document_files:
        document_type, _ = os.path.splitext(document_file)
        questions = fragenkatalog['DokumentTypen'].get(document_type)
        if questions is None:
            print(f'No questions found for document type: {document_type}')
            continue
        questions_str = '\n'.join(questions)

        document_path = os.path.join('Drucksachen', document_file)
        with open(document_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            document_text = ''
            for page_num in range(len(list(reader.pages))):
                page = reader.pages[page_num]
                document_text += page.extract_text()

        result = chain.run({
            'document': document_text,
            'questions': questions_str
        })
        print(result)
        print("**********************")

        
        result_text = '******NEUES DOKUMENT*******************************************************+\n'
        result_text += f'Document: {document_file}\n'
        result_text += f'Fragenkatalog für: {document_type}\n'
        result_text += 'Fragen:\n'
        result_text += questions_str
        result_text += '\n\LLM:\n'
        result_text += str(result)
        all_results += result_text + '\n\n'
    with open('results.txt', 'w') as f:
        f.write(all_results)
    
    os.remove(document_path)

    return all_results

# Main function
def main():
    url = input('Enter the URL of the document: ')
    options = get_firefox_configuration()
    service = FirefoxService(executable_path=GECKODRIVER_PATH, log_output=GECKODRIVER_LOG_PATH)
    driver = webdriver.Firefox(service=service, options=options)


    try:
        driver.get(url)
        driver.implicitly_wait(10)
        info = extract_info(driver)
        for doc in info['wichtige_drucksachen']:
            url = doc['link']
            date = doc['date']
            local_filename = download_file(url, date)
            print(f'Downloaded {local_filename}')
        process_documents()
    finally:
        driver.quit()

# Flask API function
def process_url(url):
    # You may want to return some meaningful results to the Flask API
    result_data = {}
    
    options = get_firefox_configuration()
    service = FirefoxService(executable_path=GECKODRIVER_PATH, log_path=GECKODRIVER_LOG_PATH)
    
    driver = webdriver.Firefox(service=service, options=options)
    
    try:
        driver.get(url)
        driver.implicitly_wait(10)
        info = extract_info(driver)
        
        for doc in info['wichtige_drucksachen']:
            url = doc['link']
            date = doc['date']
            local_filename = download_file(url, date)
            # You might want to include these in the result data to return to Flask
            result_data[local_filename] = f'Downloaded {local_filename}'
        
        processed_data = process_documents()
        result_data['processed_data'] = processed_data
    finally:
        driver.quit()
    
    return result_data


if __name__ == "__main__":
    set_openai_config()
    main()




# https://dip.bundestag.de/vorgang/verbot-von-%C3%B6l-und-gasheizungen-verhindern-priorisierung-der-w%C3%A4rmepumpen/298662