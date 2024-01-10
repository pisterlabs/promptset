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

# initialize the driver
langs = ["c++", "java", "python3", "c", "c#", "javascript", "ruby", "swift", "go", "scala", "kotlin", "rust", "php", "typescript", "racket", "erlang", "elixir", "dart"]

# options = webdriver.FirefoxOptions()
# options.add_argument("start-maximized")
# driver = webdriver.Firefox(options=options)

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

def getc(driver):
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
    return getc(driver) + f'\n\nHere is the starter code in {get_lang(driver)}:\n\n```\n' + starter_code(driver) + '\n```\n\n' + f'Do not provide an explanation, just code in {get_lang(driver)}. Be sure to annotate all code blocks with triple backticks (```).'

# def make_prompt2(driver):
#     return getc(driver) + f'\n\nHere is the starter code:\n\n```{get_lang(driver)}\n' + starter_code(driver) + '\n```\n\n' + f'Do not provide an explanation, just code in {get_lang(driver)}. Be sure to annotate all code blocks with triple backticks (```).'

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

# openai.api_key = os.getenv("OPENAI_API_KEY"); 
# hist.append(msg(completion))
# hist = []

def solve_problem(p):
    completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
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

driver = startup()
# Wait for the page to load and verify login success
assert "LeetCode" in driver.title
langs = ['c++', 'python3', 'typescript', 'rust']
df = pd.read_csv('probs.csv')
df.info()

data_dir = "data"
starter_dir = os.path.join(data_dir, "starter")

# get a list of all the text files in data_dir
text_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and f.endswith(".txt")]

df = pd.read_csv("probs.csv") # replace with the actual file path of your dataframe CSV
problem_data = []

# iterate over the text files and find the matching json file in starter_dir
for text_file in text_files:
    problem_id = os.path.splitext(text_file)[0]
    json_file = os.path.join(starter_dir, problem_id + ".json")
    
    if os.path.isfile(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        
        # find the corresponding row in the dataframe
        row = df.loc[df['prob_id'] == int(problem_id)]
        if not row.empty:
            problem_data.append((text_file, data, row))

print(problem_data)

prob_url = problem_data[0][2]['prob_url'].iloc[0]
driver.get(prob_url)

x = solve_problem(driver)
print(x)

switch_lang('typescript')
switch_lang('python3')
p = make_prompt(driver)
sol = solve_problem(p)
solb = remove_declaration(sol)
submit_prob(driver, solb)
click_button(driver, "Close")

rows = list(df.iterrows())
ts = two_sum_row = rows[0]

for (i, row) in rows[4:]:
    print(row)

    try:
        driver.get(row.prob_url)
        sleep(3)
        switch_lang('c')

        p = make_prompt(driver)
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "user", "content": p},
        ]
        )
        m = msg(completion)
        c = content(m)
        xs = get_code(c)
        x = only(xs)
        full_sol = remove_last_line(remove_declaration(x, decl_langs))

        submit_prob(driver, full_sol)
        sleep(3)
    except Exception as e:
        print(row)
        print(e)

driver = startup()
i, row = rows[0]
driver.get(row.prob_url)
sleep(3)
switch_lang('c')

p = make_prompt(driver)
completion = openai.ChatCompletion.create(
model="gpt-3.5-turbo",
messages=[
{"role": "user", "content": p},
]
)
m = msg(completion)
c = content(m)
xs = get_code(c)
x = only(xs)
full_sol = remove_last_line(remove_declaration(x, decl_langs))

submit_prob(driver, full_sol)
sleep(3)

driver = startup()

url = "https://leetcode.com/problemset/all/?sorting=W3sic29ydE9yZGVyIjoiQVNDRU5ESU5HIiwib3JkZXJCeSI6IkRJRkZJQ1VMVFkifV0%3D"
driver.get(url)
sleep(3)

table = driver.find_element(By.CSS_SELECTOR, '[role="table"]')
header = table.find_elements(By.CSS_SELECTOR, '[role="columnheader"]')
header = list(map(lambda x: x.text, header))
rows = table.find_elements(By.CSS_SELECTOR, '[role="row"]')
row = rows[1]
cols = row.find_elements(By.CSS_SELECTOR, '[role="cell"]')
# Extract data from each row and store in list of dictionaries
data = []

for i in range(51):
    print(i)
    table = driver.find_element(By.CSS_SELECTOR, '[role="table"]')
    # Find all rows within the table
    rows = table.find_elements(By.CSS_SELECTOR, '[role="row"]')
    print(f'nrows: {len(rows)}')
    for row in rows[1:]:
        cols = row.find_elements(By.CSS_SELECTOR, '[role="cell"]')
        print(f'ncells: {len(cols)}')
        prob_url = cols[2].find_elements("tag name", "a")
        print(len(prob_url))
        if len(prob_url) == 0:
            continue
        prob_url = prob_url[0].get_attribute("href")
        cols = [col.text for col in cols] + [prob_url]
        data.append(cols)
    next_button = driver.find_element("css selector", 'button[aria-label="next"]')
    next_button.click()
    sleep(3)


index = urls.index("https://leetcode.com/problems/linked-list-cycle/")
index = urls.index("https://leetcode.com/problems/shortest-word-distance/")


df = pd.read_csv("more_probs.csv")
urls = list(map(remove_url_suffix, df.prob_url.tolist()))
driver = startup()

index = urls.index("https://leetcode.com/problems/trim-a-binary-search-tree/")
for (i, url) in enumerate(urls[index:]):
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
