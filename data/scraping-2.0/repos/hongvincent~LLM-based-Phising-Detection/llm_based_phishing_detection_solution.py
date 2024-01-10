import os
import openai
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Union
import re
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException


_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
app = FastAPI()

API_KEY = "###API_KEY###"  # Replace with your API key


def extract_urls(text: str) -> List[str]:
    # url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # url_pattern = r'(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+'
    url_pattern = r'(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+(?:/[^\s]*)?'
    return re.findall(url_pattern, text)


def check_url(url: str) -> dict:
    api_url = f"https://api.criminalip.io/v1/quick/hash/view/domain?domain={url}"
    headers = {
        "x-api-key": API_KEY
    }
    payload = {}

    response = requests.request("GET", api_url, headers=headers, data=payload)
    if response.status_code == 200:
        data = response.json()

        # Check if the expected keys exist
        if "data" in data and "result" in data["data"]:
            result = data["data"]["result"]
            _type = data["data"].get("type", "N/A")

            if result == "malicious":
                return {"url": url, "status": result, "type": _type}
            else:
                return {"url": url, "status": result}
        else:
            # Print the entire response for debugging purposes
            print(f"Unexpected API response for URL {url}: {data}")
            return {"url": url, "error": "Unexpected API response"}
    else:
        return {"url": url, "error": f"{response.status_code}, {response.text}"}


def parsing_web_source(web_url) -> str:
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)

    url = str(web_url)

    try:
        # Attempt to load the website
        driver.get(url)
        html = driver.page_source
    except WebDriverException as e:
        # If there's an error (e.g., website not accessible), print an error message and return
        print(f"Error accessing {url}: {str(e)}")
        driver.quit()
        return "Error accessing website"

    soup = BeautifulSoup(html, "html.parser")

    login_form = soup.find("form", class_="login-form")
    login_button = soup.find("button", class_="login-button")

    inline_scripts = soup.findAll("script", {"type": "text/javascript"})
    suspicious_inline_scripts = [script for script in inline_scripts if "window.location" in script.text or "ajax" in script.text or "xhr" in script.text]

    links = soup.findAll("a", href=True)
    suspicious_links = [link for link in links if "http" in link["href"] and not link["href"].startswith(url)]
    iframes = soup.findAll("iframe")
    suspicious_iframes = [iframe for iframe in iframes if not iframe["src"].startswith(url)]

    result_sc = f"{login_form}{login_button}{suspicious_inline_scripts}{suspicious_links}{suspicious_iframes}"
    print(result_sc)
    driver.quit()
    return result_sc


def base_prompt_form(url) -> str:
    source_code = parsing_web_source(url)

    # Prioritize certain parts for analysis
    scripts_content = re.findall(r'<script.*?>(.*?)</script>', source_code, re.DOTALL)
    scripts_content_str = " ".join(scripts_content)

    # Summarize the source code to fit within the model's token limit
    max_length = 4000 - len(url) - len(scripts_content_str)
    source_code_summary = scripts_content_str + " " + source_code[:max_length] + (source_code[max_length:] and '...')
    print(source_code_summary)
    system_message = f"""
    I want you to act as a cyber security specialist.
    """

    user_message = f"""
    Analyze the following web page source code to determine if it is a phishing site. Use advanced analysis methods to minimize false positives. Identify if there are any decisive indicators in the code and consider the following main features:

    1. Suspicious HTML Tags: <input>, <a>, <iframe> 
    2. Domain and URL structure: {url}
    3. External resources: Existence of external resources connecting to the attacker's server
    4. JavaScript code: Suspicious redirect and data transmission related scripts observed

    - Source code:
      {source_code_summary}

    Carefully review the source code of the web page and identify the following key elements:
    - Having Sub Domain, Prefix_Suffix, URL_of_Anchor
    - The <a>tag and the website's domain name are different
    - Whether external objects embedded within a web page, such as media files are loaded from another domain
    - Suspicious form creation and data transmission structures
    - Visible or hidden redirects leading users to malicious sites
    - Pathways through which data is sent to the attacker
    - Elements exposing sensitive information such as user authentication details

    Finally, after carefully reviewing the four items above, please score each of them as follows and add them up. 25 points if you suspect phishing, 0 points otherwise. If you suspect all of the items are phishing, you have a total of 100 points.
    Never give me a side note about your grading. Give me only the total score, in the following format
    [totalscore: <totalscore>]
    Again, you can only respond to me in the above format.
    """
    messages = [
        {'role': 'system',
         'content': system_message},
        {'role': 'user',
         'content': f"{user_message}"},
    ]
    llm_response = get_completion_from_messages(messages)
    return llm_response


def get_completion_from_messages(messages,
                                 model="gpt-3.5-turbo",
                                 temperature=0,
                                 max_tokens=500) -> Union[str, dict]:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"Error encountered: {str(e)}")
        return {"result": "error"}

def determine_response(score: Union[int, str]) -> str:
    """Maps the extracted score to the desired response."""
    if score == "No score found":
        return score
    elif score >= 75:
        return "phishing"
    elif score == 50:
        return "caution"
    elif score == 25:
        return "attention"
    else:
        return "safe"

class SMS(BaseModel):
    sms_text: str


@app.post('/check-sms')
async def check_sms(payload: dict):
    sms_text = payload.get("sms_text", "")
    urls = extract_urls(sms_text)
    results = []

    for url in urls:
        result = check_url(url)

        # Printing each item of the JSON to the server terminal
        for key, value in result.items():
            print(f"{key}: {value}")

        # If the status is 'malicious' or 'unknown', call llm_call function
        if result.get('status') in ['unknown']:
            sms_new = SMS(sms_text=url)
            llm_result = llm_call(sms_new)

            # Handle if llm_result is an error
            if isinstance(llm_result, dict) and llm_result.get("result") == "error":
                print("Error processing LLM response")
                result['llm_response'] = "Error processing LLM response"
                results.append(result)
                continue

            # Print the full llm_call() result on the server terminal
            print(f"LLM Call Result: {llm_result.get('result')}")

            # Add the LLM response to the result
            result['llm_response'] = llm_result
        results.append(result)

    if not results:
        return {"result": "No URLs found in the SMS"}

    # Extracting the score from the llm_response
    score_pattern = r"\[totalscore: (\d+)\]"

    # Ensure the result is a string before applying regex
    llm_response_str = str(results[0].get("llm_response", {}).get("result", ""))
    match = re.search(score_pattern, llm_response_str)

    score = int(match.group(1)) if match else "No score found"
    response = determine_response(score)  # Use a function to determine the response

    return {"result": response}

@app.post("/llm")
def llm_call(sms: SMS) -> dict:
    urls = extract_urls(sms.sms_text)
    print(urls)
    # Check if the url starts with http or https
    for i in range(len(urls)):
        if not urls[i].startswith('http'):
            urls[i] = 'http://' + urls[i]

    llm_response = base_prompt_form(urls[0])
    return {"result": llm_response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)