# setup
import os
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver import FirefoxOptions
from pypdf import PdfReader
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
username = os.getenv("username")
password = os.getenv("password")
question = os.getenv("question")
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from IPython.display import Markdown, display
import base64
import time
import glob2

# PyPDF for backup

#def get_pdf_text(pdf_docs):
#    text = ""
#    for pdf in pdf_docs:
#        pdf_reader = PdfReader(pdf)
#        for page in pdf_reader.pages:
#            text += page.extract_text()
#    return text

opts = FirefoxOptions()
opts.add_argument("--headless")
#profs = FirefoxProfile()
opts.set_preference("browser.download.useDownloadDir", True)
__cwd__ = os.getcwd()
download_dir = __cwd__ + '/data'
opts.set_preference("browser.download.folderList", 2)
opts.set_preference("browser.download.dir", download_dir)

#opts.set_profile = profs

driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=opts)

# code
driver.get('https://emas2.ui.ac.id/login/index.php')
time.sleep(1)
#driver.find_element("xpath", "//body/div[3]/div[1]/div[2]/section/div/div/div/div[3]/div/div[1]/form/button").click(

driver.find_element("id", "username").send_keys(username)
driver.find_element("id", "password").send_keys(password)
driver.find_element("id", "loginbtn").click()
time.sleep(10)
driver.find_element("xpath", "/html/body/div[5]/div[1]/div[2]/div/div/section/div/aside/section[2]/div/div/div[1]/div[2]/div/div/div[1]/div/div/div[2]/div/div[1]/div/div/ul[1]/li/div/div[2]/h6/a").click()
time.sleep(1)
driver.find_element("xpath", "/html/body/div[4]/div[1]/div[2]/div/div/section/div[1]/div[1]/div/div/div/div/table/tbody/tr/td[2]/div/div[1]/a").click()
time.sleep(1)

#Fitur eksperimental - input modul
driver.find_element("xpath", "/html/body/div[4]/div[1]/div[1]/div/div/nav/ul/li[7]/span[1]/a/span").click()
time.sleep(1)
driver.find_element("xpath", "/html/body/div[4]/div[1]/div[2]/div/div/section/div/div/ul/li[7]/div[3]/ul/li[2]/div/div/div[2]/div[1]/a/span").click()
time.sleep(1)
#driver.find_element("xpath", "//*[@id='secondaryToolbarToggle']").click()
time.sleep(1)
#Save as pdf
pdf = base64.b64decode(driver.print_page())
with open("data/print_page.pdf", 'wb') as f:
    f.write(pdf)
#print(driver.find_element("xpath", "/html/body").text)

#Manual PDF scan for testing
#pdf_docs = glob2.iglob("pdf/print_page.pdf")
#raw_text = get_pdf_text(pdf_docs)

#Untuk testing
#print(raw_text)

documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query(question)

#Storing and Loading the Index
index.storage_context.persist()

from llama_index import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context=storage_context)

#Custom
from llama_index import ServiceContext, set_global_service_context

#define LLM:
llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)
#configure service context
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1000, chunk_overlap=20)
#set_global_service_context(service_context)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

files = glob2.iglob('data/*')
for f in files:
    os.remove(f)

#display(Markdown(f"<b>{response}</b>"))
print(response)

#time.sleep(5)
driver.quit()
