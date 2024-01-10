import requests
import openai
from bs4 import BeautifulSoup

# 사용자로부터 검색어를 입력받습니다.
query = input("검색어를 입력하세요: ")

# 네이버 검색 URL을 구성합니다.
url = f"https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query={query}"

# requests를 이용해 URL의 HTML 페이지를 가져옵니다.
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
response = requests.get(url, headers=headers)

# 서버로부터 올바른 응답을 받았는지 확인합니다.
if response.status_code == 200:
    # BeautifulSoup을 이용해 파싱합니다.
    soup = BeautifulSoup(response.text, 'html.parser')

    # 요약, 뉴스, 지식iN, 지식백과, view의 각 항목을 추출합니다.
    detail = soup.find_all('div', class_='detail')
    news_title = soup.find_all('a', class_='news_tit')
    news_detail = soup.find_all('a', class_='api_txt_lines dsc_txt_wrap')
    descriptions = soup.find_all('div', class_='api_txt_lines desc')
    view_title = soup.find_all('a', class_='title_link _cross_trigger')
    question = soup.find_all('a', class_='api_txt_lines question_text')
    answer = soup.find_all('a', class_='api_txt_lines answer_text')
    
    print("\n[요약]")
    for det in detail:
        print("\n",det.get_text().strip())
        
    print("\n[뉴스]")
    # 뉴스 제목과 상세 설명을 출력합니다.
    for newst, newsd in zip(news_title, news_detail):
        print("\n",newst.get_text().strip())
        print(newsd['href'].strip())  # 상세 설명이 아니라 링크를 추출합니다.
        print(newsd.get_text().strip())

        
    print("\n[지식iN]")
    # 질문과 답변을 출력합니다.
    for ques, ans in zip(question, answer):
        print("\n",ques.get_text().strip())
        print(ans.get_text().strip())
        
    print("\n[view]")
    for title in view_title:
        print("\n",title.get_text().strip())
        
    print("\n[지식백과]")
    # 각 설명의 텍스트를 출력합니다.
    for desc in descriptions:
        print("\n",desc.get_text().strip())
        
else:
    print("Error: Unable to fetch the web page.")


