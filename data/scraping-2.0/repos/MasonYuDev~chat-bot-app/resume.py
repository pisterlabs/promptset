from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from openai import OpenAI


def scrape_dynamic_page(url):

  chrome_options = Options()
  chrome_options.add_argument('--headless')
  chrome_options.add_argument('--disable-gpu')
  chrome_options.add_argument('--no-sandbox')
  chrome_options.add_argument('--disable-dev-shm-usage')
  chrome_options.add_argument('--no-sandbox')

  # Set up the webdriver
  driver = webdriver.Chrome(options=chrome_options)
  job_description_url = url
  driver.get(job_description_url)
  content = driver.page_source

  soup = BeautifulSoup(content, 'html.parser')

# Remove hreflang tags
  for tag in soup.find_all('link', hreflang=True):
    tag.extract()

# Remove cookiereport tags
  for tag in soup.find_all(class_=lambda x: x and 'CookieReports' in x):
    tag.extract()
# Remove country options
  for tag in soup.find_all('option', value=True):
    tag.extract()

# Remove CSS stylings
  for tag in soup.find_all(True):
    tag.attrs = {}

  for style_tag in soup.find_all('style'):
    style_tag.clear()

# Get the modified HTML content
  modified_html = str(soup)
  driver.quit()

  return modified_html

def process_web_content(html_content):
  system = "You are a helpful product manager who will clean dirty html content scraped from a job description, and return only the exact job description and title. Please return a JSON object with these 2 parameters."
  user = html_content
  model = "gpt-3.5-turbo-16k"
  max_tokens = 1500
  temperature = 0.1
  return run_open_ai(model, system, user, max_tokens, 1, None, temperature)

def extract_job_requirements(job_description):
  system = "You are a HR automatic resume processor. You have a given job description and will need to return a given set of criteria inferred by this job description. Please output a simple list of top 15 skills implied by this job description as a JSON object."
  user = job_description
  model = "gpt-4"
  max_tokens = 1500
  temperature = 1
  return run_open_ai(model, system, user, max_tokens, 1, None, temperature)

def generate_cover_letter(job_description, matching_skills):
  system = "Given the json object of job description and title, plus a JSON object of the applicant's experiences, write a cohesive and succinct cover letter convincing hiring manager to hire this applicant. Minimize focus on skills that are missing."
  user = job_description + "/n/n" + matching_skills + "/n/n/n This is the end of the JSON object even if it's cut off."
  model = "gpt-4"
  max_tokens = 1300
  temperature = 1
  return run_open_ai(model, system, user, max_tokens, 1, None, temperature)

def run_open_ai(model, system, user, tokens, n, stop, temp):
  """
  Base OpenAI Function.

  Args:
    model (str): OpenAI model name
    system (str): System prompt
    user (str): User prompt
    tokens (int): Max tokens
    n (int): The number of completion choices to generate. Usually 1
    stop (str): optional setting that tells the API when to stop generating tokens. Usually None
    temp (float): Set temperature

  Returns:
    prompt response (str)

  """
  client = OpenAI()

  response = client.chat.completions.create(
    model=model,
    messages=[
      {"role": "system", "content": system},
      {"role": "user", "content": user},
    ],
    max_tokens=tokens,
    n=n,
    stop=stop,
    temperature=temp
  )

  return response.choices[0].message.content

