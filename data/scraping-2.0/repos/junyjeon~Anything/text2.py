import openai
import requests
from bs4 import BeautifulSoup
from gpt4free import GPT

# GPT-4 모델 로드
gpt = GPT()


def is_crawling_allowed(url):
    robots_url = url + "/robots.txt"
    response = requests.get(robots_url)
    if response.status_code == 200:
        if "Disallow: /" in response.text:
            return False  # 크롤링이 금지된 사이트
    return True  # 크롤링이 허용되거나 robots.txt 파일이 없는 사이트


def crawl_website(url, element_id):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        element = soup.find(id=element_id)
        return element.get_text() if element else ""
    return ""


def extract_keywords(text):
    response = gpt.generate(text)
    keywords = response.split(',')
    return keywords


def create_finetuning_data(selected_keywords):
    finetuning_data = []
    for keyword in selected_keywords:
        # 키워드를 사용하여 파인튜닝 데이터 생성
        prompt = f"Please summarize information about {keyword}."
        response = gpt.generate(prompt)
        completion = response
        finetuning_data.append({"prompt": prompt, "completion": completion})
    return finetuning_data


# 사용자가 웹사이트와 element_id를 지정합니다.
website_url = "http://example.com"
element_id = "content"

if is_crawling_allowed(website_url):
    text = crawl_website(website_url, element_id)
    keywords = extract_keywords(text)

    # 여기서 사용자가 keywords 리스트에서 키워드를 선별합니다.
    # 이 예시에서는 모든 키워드를 사용한다고 가정합니다.
    selected_keywords = keywords  # 실제로는 사용자가 선별한 키워드 리스트를 사용해야 합니다.

    finetuning_data = create_finetuning_data(selected_keywords)

    # 여기서 finetuning_data를 필요에 따라 저장하거나 활용합니다.
    # 예를 들어, 이 데이터를 파일에 저장하거나 API를 통해 전송할 수 있습니다.
else:
    print("Crawling is not allowed for this website.")


# Path: text2.py

# OpenAI API 키 설정
openai.api_key = 'your-api-key'


def is_crawling_allowed(url):
    robots_url = url + "/robots.txt"
    response = requests.get(robots_url)
    if response.status_code == 200:
        if "Disallow: /" in response.text:
            return False  # 크롤링이 금지된 사이트
    return True  # 크롤링이 허용되거나 robots.txt 파일이 없는 사이트


def crawl_website(url, element_id):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        element = soup.find(id=element_id)
        return element.get_text() if element else ""
    return ""


def extract_keywords(text):
    response = openai.Completion.create(
        model="gpt-4.0-turbo",
        prompt=f"Extract keywords from this text: \"{text}\"",
        max_tokens=60
    )
    # 추출된 키워드를 파싱하여 리스트로 반환
    keywords = response.choices[0].text.split(',')
    return [keyword.strip() for keyword in keywords]


def create_finetuning_data(selected_keywords):
    finetuning_data = []
    for keyword in selected_keywords:
        # 키워드를 사용하여 파인튜닝 데이터 생성
        prompt = f"Please summarize information about {keyword}."
        response = openai.Completion.create(
            model="gpt-4.0-turbo",
            prompt=prompt,
            max_tokens=60
        )
        completion = response.choices[0].text.strip()
        finetuning_data.append({"prompt": prompt, "completion": completion})
    return finetuning_data


# 사용자가 웹사이트와 element_id를 지정합니다.
website_url = "http://example.com"
element_id = "content"

if is_crawling_allowed(website_url):
    text = crawl_website(website_url, element_id)
    keywords = extract_keywords(text)

    # 여기서 사용자가 keywords 리스트에서 키워드를 선별합니다.
    # 이 예시에서는 모든 키워드를 사용한다고 가정합니다.
    selected_keywords = keywords  # 실제로는 사용자가 선별한 키워드 리스트를 사용해야 합니다.

    finetuning_data = create_finetuning_data(selected_keywords)

    # 여기서 finetuning_data를 필요에 따라 저장하거나 활용합니다.
    # 예를 들어, 이 데이터를 파일에 저장하거나 API를 통해 전송할 수 있습니다.
else:
    print("Crawling is not allowed for this website.")
