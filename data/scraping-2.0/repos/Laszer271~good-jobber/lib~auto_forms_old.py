from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import AIMessagePromptTemplate

from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By

import json
import os

with open('./credentials/openai.txt', 'r') as f:
    openai_api_key = f.read()
    os.environ['OPENAI_API_KEY'] = openai_api_key


# def do_work(form, user_info):
#     # llm = OpenAI(openai_api_key=openai_api_key, temperature=0.1)

#     # print(form)

#     form_template = ""
#     form_path = './data/form_template.html'
#     with open(form_path, 'w') as f:
#         f.write(form)

#     review_template = """\
#     You are provided with personal information witin file: {user_info}.
#     You are also provided with the following HTML form: {form_path}
#     Your task is to load data from both those files and write a Selenium script that will fill the form with the data as if the form was a real website.
#     You should output only the Selenium script and not the data loading part. 
#     IMPORTANT: Especially you can't output the data from the personal information file.

#     You should first load the data from both files and print to the console the data that you loaded.
#     For that you can use print(...)
#     Next, after you already know the html code of the form from the previous step,
#     you should write the Selenium script that will fill the form with the data.
#     You should use the following Selenium methods:

#     '''
#     from selenium import webdriver
#     from selenium.webdriver import FirefoxOptions

#     opts = FirefoxOptions()
#     opts.add_argument("--headless")
#     browser = webdriver.Firefox(options=opts)

#     ==== HERE GOES YOUR CODE ====
#     # You should use: find_element("xpath", xpath), find_element_by_xpath and equivalent methods don't work in the current version of SeleniumBase
#     '''
#     """

#     agent_executor = create_python_agent(
#         llm=OpenAI(temperature=0, max_tokens=1000),
#         tool=PythonREPLTool(),
#         verbose=True,
#         agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     )

#     prompt_template = SystemMessagePromptTemplate.from_template(review_template)
#     print(prompt_template)
#     messages = prompt_template.format(form_path=form_template, user_info=user_info)
#     response = agent_executor.run(messages)
#     print(response)
#     # print(response.content)


def do_work(form, user_info):
    # llm = OpenAI(openai_api_key=openai_api_key, temperature=0.1)

    # print(form)

    form_template = ""
    # form_path = './data/form_template.html'
    # with open(form_path, 'w') as f:
    #     f.write(form)

    review_template = """\
    There is a json file containing personal information. To keep the information private, you are only provided with the keys of this json file.
    You are also provided with the following HTML form that needs to be filled with the personal data.
    The form is a form that normally candidates fill when applying for a job.

    Your task is to output a json file that does mapping from the XPATH of the form fields to the keys of the json file.
    In the form there are also fields that need files to be uploaded. You should treat them the same way as the other fields.
    If you don't have the keys needed to fill some of the fields, you should output null for those fields.
    If some of the fields require a creative answer, you should output "CREATIVE" for those fields.
    Example:
    Personal information keys:
    ["Name", "Surname", "Email", "CV", "Phone number"]

    Form to be filled:



    {
        "//*[@id="name"]": "Name",
        "//*[@id="apply-modal"]/section/common-material-modal/div/section/nfj-apply-internal-step-application/form/nfj-form-field[2]/div[1]/div/input": "Email",
        "/html/body/div[4]/div[2]/div/mat-dialog-container/nfj-posting-apply-internal-modal/section/common-material-modal/div/section/nfj-apply-internal-step-application/form/div[2]/nfj-user-files/nfj-apply-attachment-btn/span[2]/label": "CV",
        "//*[@id="your_mothers_maiden_name"]": null,
        "//*[@id="cover_letter"]": "CREATIVE",
    }
    """

    agent_executor = create_python_agent(
        llm=OpenAI(temperature=0, max_tokens=1000),
        tool=PythonREPLTool(),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    prompt_template = SystemMessagePromptTemplate.from_template(review_template)
    print(prompt_template)
    messages = prompt_template.format(form_path=form_template, user_info=user_info)
    response = agent_executor.run(messages)
    print(response)
    # print(response.content)


def scrape_web_page(url):
    # Fetch the web page content
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        html_content = response.text
    else:
        print("Failed to fetch the web page.")
        return None

    # Parse the HTML with Beautiful Soup
    soup = BeautifulSoup(html_content, "html.parser")

    return soup

if __name__ == '__main__':
    url_to_scrape = "https://nofluffjobs.com/pl/job/programista-web-regular-ework-group-warszawa"

    driver = webdriver.Chrome()

    # Open the web page
    driver.get(url_to_scrape)

    # Find the form element by its tag name "form"
    id_button = '//*[@id="applyButton"]'
    form_tag = '//form'
    aplikuj_button = WebDriverWait(driver, 10).until(lambda x: x.find_element('xpath', id_button))
    driver.execute_script("arguments[0].scrollIntoView();", aplikuj_button)
    aplikuj_button.click()
    form = WebDriverWait(driver, 10).until(lambda x: x.find_element(By.TAG_NAME, 'form'))

    myForm = driver.find_element(By.TAG_NAME, 'form')
    # content_form = driver.find_element("xpath", form_tag)
    
    user_info_path = './data/example_personal_info.json'
    with open(user_info_path, 'r') as f:
        user_info = json.load(f)

    content = myForm.get_attribute("outerHTML")
    print('='*100)
    print(content)
    print('='*100)
    raise
    # print(content)
    do_work(content, user_info=user_info_path)