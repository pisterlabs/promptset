from time import sleep
import json
import os
import re
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pandas as pd
import numpy as np
import openai
from urllib.parse import urlsplit, urlunsplit
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.Model.list())
# initialize the driver
langs = ["c++", "java", "python3", "c", "c#", "javascript", "ruby", "swift", "go", "scala", "kotlin", "rust", "php", "typescript", "racket", "erlang", "elixir", "dart"]
decl_langs = langs + ["python"]
gpt_models = ["gpt-4", "gpt-3.5-turbo"]


def login(driver):
    driver.get("https://leetcode.com/accounts/login/")
    wait = WebDriverWait(driver, 10)
    sleep(5)
    email = wait.until(EC.presence_of_element_located((By.NAME, "login")))
    email.send_keys("anandj@uchicago.edu")
    password = wait.until(EC.presence_of_element_located((By.NAME, "password")))
    password.send_keys("$(5GwewCf%~QRLN")
    password.send_keys(Keys.RETURN)
    sleep(5)
    wait = WebDriverWait(driver, 10)
    
def startup():
    driver = webdriver.Firefox()
    login(driver)
    return driver

def only(iterable):
    """Return the one and only element of the iterable, or raise an exception if the iterable is empty or has multiple elements."""
    it = iter(iterable)
    result = next(it)
    try:
        next(it)
    except StopIteration:
        return result
    else:
        raise ValueError("Expected only one element in iterable, but found more.")

def get_prob_prompt(driver):
    return driver.find_element('xpath', "/html/head/meta[@name='description']").get_attribute("content")

def get_btns(driver):
    return driver.find_elements('tag name', 'button')

def get_btn(langs):
    return only([btn for btn in get_btns(driver) if btn.text.lower() in langs])

def get_submit_button(langs):
    return only([btn for btn in get_btns(driver) if btn.text.lower() == "submit"])

def get_lang(driver):
    return get_btn(langs).text

def starter_code(driver):
    return '\n'.join([line.text for line in driver.find_elements('css selector', "div.view-line")])

def switch_lang(lang):
    """Switch the webpage language to the given language."""
    btn = get_btn(langs)
    btn.click()
    
    langbtns = btn.find_elements("xpath", "//li")
    lb = only([btn for btn in langbtns if btn.text.lower() == lang])
    lb.click()

def make_prompt(driver):
    return get_prob_prompt(driver) + f'\n\nHere is the starter code in {get_lang(driver)}:\n\n```\n' + starter_code(driver) + '\n```\n\n' + f'Do not provide an explanation, just code in {get_lang(driver)}. Be sure to annotate all code blocks with triple backticks (```).'

# def make_prompt2(driver):
#     return get_prob_prompt(driver) + f'\n\nHere is the starter code:\n\n```{get_lang(driver)}\n' + starter_code(driver) + '\n```\n\n' + f'Do not provide an explanation, just code in {get_lang(driver)}. Be sure to annotate all code blocks with triple backticks (```).'

def msg(x):
    return x.choices[0].message

def content(x):
    return msg(x).content

def get_code(s):
      """Extract code blocks from the given string and return a list of them."""
      code_re = re.compile(r'```([\s\S]*?)```')
      return code_re.findall(s)

def get_code_w_lang(s, lang):
      """Extract code blocks from the given string and return a list of them."""
      code_re = re.compile(s, fr'```{re.escape(lang)}([\s\S]*?)```')
      return code_re.findall(s)

def solve_problem(p):
    completion = openai.ChatCompletion.create(
  model=gpt_models[0],
  messages=[
    {"role": "user", "content": p},
  ]
)
    m = msg(completion)
    c = content(m)
    xs = get_code(c)
    assert len(xs) == 1
    x = only(xs)
    return x


def find_btn(driver, button_text="Submit"):
    return driver.find_element('xpath', f'//button[text()="{button_text}"]')
    
def click_button(driver, button_text="Submit"):
    submit_button = find_btn(driver, button_text)
    submit_button.click()

def submit_prob(driver, soln):
    driver.find_element(By.CSS_SELECTOR, ".inputarea").send_keys(Keys.COMMAND + 'a' + Keys.DELETE)
    driver.find_element(By.CSS_SELECTOR, ".inputarea").send_keys(soln)
    lang = get_lang(driver)
    if lang == "Rust":
        driver.find_element(By.CSS_SELECTOR, ".inputarea").send_keys(Keys.BACKSPACE) # lol hack for rust
    elif lang == "C++":
        print("HIIII C++")
        # driver.find_element(By.CSS_SELECTOR, ".inputarea").send_keys(Keys.BACKSPACE)
        # driver.find_element(By.CSS_SELECTOR, ".inputarea").send_keys(Keys.BACKSPACE)
        driver.find_element(By.CSS_SELECTOR, ".inputarea").send_keys(Keys.DOWN + Keys.BACKSPACE)

    click_button(driver)


def remove_declaration(s: str, langs:list =langs + ["python"]) -> str:
    # Split the input string into lines
    lines = s.split('\n')
    
    # Check if the first line is in langs
    if len(lines) > 0 and lines[0].strip() in langs:
        # Remove the first line and rejoin the remaining lines
        return '\n'.join(lines[1:])
    
    # If the first line is not in langs, return the original string
    return s

def remove_last_line(s: str) -> str:
    # Split the string into lines
    lines = s.split('\n')
    
    # Remove the last line and rejoin the remaining lines
    new_lines = lines[:-1]
    return '\n'.join(new_lines)

def remove_url_suffix(url):
    # Split the URL into its components
    scheme, netloc, path, query, fragment = urlsplit(url)

    # Strip the last part of the path
    path_parts = path.split('/')
    if path_parts[-1] == '':
        path_parts = path_parts[:-1]  # Remove last empty part if present
    path_parts = path_parts[:-1]  # Remove last non-empty part
    path = '/'.join(path_parts) + '/'

    # Rebuild the URL and return it
    return urlunsplit((scheme, netloc, path, query, fragment))

df = pd.read_csv("more_probs.csv")
urls = list(map(remove_url_suffix, df.prob_url.tolist()))
driver = startup()

# index = urls.index("https://leetcode.com/problems/trim-a-binary-search-tree/")
# index = urls.index("https://leetcode.com/problems/maximum-profit-in-job-scheduling/")
# for (i, url) in enumerate(urls[index:]):
for (i, url) in enumerate(urls):
    print(f'{i}: {url}')

    try:
        driver.get(url)
        sleep(3)
        switch_lang('rust')

        p = make_prompt(driver)
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "user", "content": p},
        ]
        )
        m = msg(completion)
        c = content(completion)
        xs = get_code(c)
        x = only(xs)
        full_sol = remove_last_line(remove_declaration(x, decl_langs))

        submit_prob(driver, full_sol)
        sleep(10)
    except Exception as e:
        print(url)
        print(e)
        print(f"Exception caught: {e}")
        # raise e

# todo
# implement logging to save the prompts, but better than i had previously. and get all of them 
# we also want to either 1) sandbox and run locally to capture error, or use the error result from leetcode (which i think may be abbreviated)
# the task is to be able to make a graph about how much feedback improves the ability for chatgpt to solve problems
# we may want to experiment with giving it the ability to search for the docs, find the repo and copy the latest doc page or function signatures

