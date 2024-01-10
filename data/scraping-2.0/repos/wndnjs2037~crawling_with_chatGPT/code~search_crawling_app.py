import streamlit as st
import pandas as pd
from newspaper import Config, Article
import googletrans
import openai
from langdetect import detect
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import re
from selenium import webdriver
from youtube_transcript_api import YouTubeTranscriptApi
from konlpy.tag import Kkma
from pykospacing import Spacing
from playwright.sync_api import sync_playwright
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import urllib.request
import json
import urllib
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# 필요한 전역변수 선언
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
config = Config()
translator = googletrans.Translator()
input_prompt = ""
first_url = ""
second_url = ""
third_url = ""

# openAPI 개인용 시크릿키
YOUR_API_KEY = '보안상의 이유로 깃허브에는 올리지 않습니다'

#  chatGPT에 명령어를 날리기 위한 함수
def chatGPT(prompt, API_KEY=YOUR_API_KEY):

    # set api key
    openai.api_key = API_KEY

    # Call the chat GPT API
    completion = openai.Completion.create(
        # 'text-curie-001'  # 'text-babbage-001' #'text-ada-001'  #모델 종류별 기능 확인해보기
        engine='text-davinci-003', prompt=prompt, temperature=0.5, max_tokens=1024, top_p=1, frequency_penalty=0, presence_penalty=0)

    return completion['choices'][0]['text']


# chatGPT가 정상실행될 수 있도록 페이지에 처음 접속시 자동으로 실행되는 실행 테스트 코드 - 안쓸거임
def chatGPT_execution_confirmation():
  st.write("chatGPT 실행 가능 여부 확인중 ...")
  prompt = f"this is running test."
  answer = chatGPT(prompt).strip()
  print("answer : ", answer)

  if answer is not None:
    st.write("실행 가능 여부 확인완료.")

  else:
    st.write("비정상 작동")

# 네이버 검색용) 포스트 크롤러
def naver_post_crawler_for_naver_search(url):
    st.write("네이버포스트 링크입니다.")

    playwright = sync_playwright().start() # 크롤링용 플레이라이트 시작
    browser = playwright.chromium.launch(headless=True) 
    page = browser.new_page() # 웹브라우저 열기
    page.goto(url) # 파라미터로 받은 url 주소로 이동
    ctx = page.locator('xpath=/html/body') # 오픈한 페이지에서 가져올 html code path 지정(body로 지정해서 전체 가져옴)
    ctx_inner_html = ctx.inner_html() # 조회한 페이지에서 html 코드 가져오기

    # 포스트의 제목을 가져오기 위함
    # 제목 태그명을 선택한 뒤 class 이름 붙여서 접근
    title_tag = page.locator('h3.se_textarea')
    title_text = title_tag.inner_html().replace("  ","")
    st.write("게시글 제목:", title_text)

    # 앞을 잘라내기 위해 title 태그 지정
    # 해당 태그 다음부터 본문의 내용이 시작된다(제목, 본문)
    under_title_tag_location = ctx_inner_html.find("<h3 class=\"se_textarea")
    

    # 뒷부분을 잘라내기 위해 end 태그 지정
    # 해당 태그가 위치한 곳부터 내용을 잘라낸다(본문이 끝난 이후 바로 다음에 등장하는 태그로 지정)
    over_end_tag_location = ctx_inner_html.find("<div class=\"state_line") 

    # 조회한 태그의 위치만큼 html 코드를 잘라내기
    tag_data = ctx_inner_html[under_title_tag_location:over_end_tag_location]

    # 잘라낸 html 코드에서 tag 데이터를 지우고 순수 텍스트만 가져오기
    cleantext = BeautifulSoup(tag_data, "lxml").text
    naver_post_cleantext = cleantext.replace("\n\n", "")
    # print(cleantext.replace("\n\n", "")) # 불필요한 공백(엔터) 제거 후 출력확인
    st.write("본문:")
    st.write(naver_post_cleantext)

    browser.close()
    print("-------------------완료------------------")

    return naver_post_cleantext


# 구글 검색용) 네이버 포스트 크롤러
def naver_post_crawler_for_google_search(url, playwright, browser):
    st.write("네이버포스트 링크입니다.")

    # playwright = sync_playwright().start() # 크롤링용 플레이라이트 시작 - 중복선언 불가
    browser = playwright.chromium.launch(headless=True) 
    page = browser.new_page() # 웹브라우저 열기
    page.goto(url) # 파라미터로 받은 url 주소로 이동
    ctx = page.locator('xpath=/html/body') # 오픈한 페이지에서 가져올 html code path 지정(body로 지정해서 전체 가져옴)
    ctx_inner_html = ctx.inner_html() # 조회한 페이지에서 html 코드 가져오기

    # 포스트의 제목을 가져오기 위함
    # 제목 태그명을 선택한 뒤 class 이름 붙여서 접근
    title_tag = page.locator('h3.se_textarea')
    title_text = title_tag.inner_html().replace("  ","")
    st.write("게시글 제목:", title_text)

    # 앞을 잘라내기 위해 title 태그 지정
    # 해당 태그 다음부터 본문의 내용이 시작된다(제목, 본문)
    under_title_tag_location = ctx_inner_html.find("<h3 class=\"se_textarea")

    # 뒷부분을 잘라내기 위해 end 태그 지정
    # 해당 태그가 위치한 곳부터 내용을 잘라낸다(본문이 끝난 이후 바로 다음에 등장하는 태그로 지정)
    over_end_tag_location = ctx_inner_html.find("<div class=\"state_line") 

    # 조회한 태그의 위치만큼 html 코드를 잘라내기
    tag_data = ctx_inner_html[under_title_tag_location:over_end_tag_location]

    # 잘라낸 html 코드에서 tag 데이터를 지우고 순수 텍스트만 가져오기
    cleantext = BeautifulSoup(tag_data, "lxml").text
    naver_post_cleantext = cleantext.replace("\n\n", "")
    # print(cleantext.replace("\n\n", "")) # 불필요한 공백(엔터) 제거 후 출력확인
    st.write("본문:")
    st.write(naver_post_cleantext)

    browser.close()
    print("-------------------완료------------------")

    return naver_post_cleantext
  
# 블로그 본문은 가져오기 위해 iframe 제거 후 blog.naver.com 붙이기
def delete_iframe(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"}
    res = requests.get(url, headers=headers)
    res.raise_for_status()  # 문제시 프로그램 종료
    soup = BeautifulSoup(res.text, "lxml")

    src_url = "https://blog.naver.com/" + soup.iframe["src"]

    return src_url

# 네이버블로그 본문 스크래핑 - 스마트 에디터 2.0, ONE 포함
def text_scraping(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"}
    res = requests.get(url, headers=headers)
    res.raise_for_status()  # 문제시 프로그램 종료
    soup = BeautifulSoup(res.text, "lxml")

    if soup.find("div", attrs={"class": "se-main-container"}): # 본문 전체 태그
        text = soup.find(
            "div", attrs={"class": "se-main-container"}).get_text()
        text = text.replace("\n", "") 
        return text

    elif soup.find("div", attrs={"id": "postViewArea"}):  # 스마트 에디터 2.0도 크롤링 해오기 위함
        text = soup.find("div", attrs={"id": "postViewArea"}).get_text()
        text = text.replace("\n", "")
        return text

    else:
        return "네이버 블로그는 맞지만, 확인이 불가한 데이터입니다."  # 스마트 에디터로 작성되지 않은 글은 조회하지 않음 ..


# 네이버 검색용)
# 네이버에 검색어를 입력하여 View 탭으로 이동, li의 link를 가져오는 함수
def naver_crawling(input_data):

  crawling_done_count = 0
  original_crawling_result =""

  # 검색할 키워드 입력
  query = input_data
  url = "https://search.naver.com/search.naver?where=view&sm=tab_jum&query=" + \
      quote(query)  # 아스키코드형식으로 변환, 네이버 View 탭 기준으로 긁어오기

  # 내 헤더 : Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36
  headers = {
      "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"}
  res = requests.get(url, headers=headers)
  res.raise_for_status()  # 문제시 프로그램 종료
  soup = BeautifulSoup(res.text, "lxml")

  # 네이버에서 정의해놓은 게시글들의 리스트 태그 (limit으로 제한, 최대 30개 가능)
  # limit은 단순히 반복하는 횟수 제한임
  posts = soup.find_all("li", attrs={"class": "bx _svp_item"}, limit=10) # li 태그의 bx _svp_item이라는 클래스를 가진 값들을 soup으로 찾아서 저장, limit 10 제한
  
  
  for post in posts:  # posts 입력시 30개 가져옴
    
    if crawling_done_count < 3:

        # 해당 게시글의 링크 가져오기
        # 가져올 태그와 태그에 사용된 class명 입력해줌
        post_link = post.find("a", attrs={"class": "api_txt_lines total_tit _cross_trigger"})[
            'href']
        st.write("검색된 link : ", post_link)

        # 해당 게시글의 제목 가져오기
        post_title = post.find("a", attrs={
                                "class": "api_txt_lines total_tit _cross_trigger"}).get_text()  # 해당 게시글의 제목 text 값 가져오기
        st.write("게시글 제목 : ", post_title)

        
        # 도메인별 분리
        blog_p = re.compile("blog.naver.com")  # compile() : 문자열을 컴파일해서 파이썬 코드로 반환
        blog_m = blog_p.search(post_link)
        cafe_p = re.compile("cafe.naver.com") # 카페 게시글 거르기 위함
        cafe_m = cafe_p.search(post_link)
        post_p = re.compile("post.naver.com")
        post_m = post_p.search(post_link)
        
        # url에 blog가 포함된 경우
        if blog_m:
            st.write("네이버 블로그입니다.")
            blog_text = text_scraping(delete_iframe(post_link))
            st.write("크롤링된 네이버 블로그 본문 : ", blog_text)

            # 크롤링한 데이터를 번역(en 이외의 언어인 경우)
            for_prompt_data_one = google_translator(blog_text)

            st.write("chatGPT 실행중...")

            # 사용자가 입력한 프롬프트와 en으로 번역된 크롤링 데이터를 chatGPT 프롬프트에 전송
            prompt_command = input_prompt + " " + for_prompt_data_one # 직접 input 된 데이터가 들어가는지 확인 필요
            st.write(input_prompt)
            # st.write("prompt_command:",prompt_command)
            result = chatGPT(prompt_command).strip()  # 왼쪽, 오른쪽 공백 제거
            st.write("결과 원문:", result)
            st.write("결과 길이:", len(result))

            # chatGPT 결과의 길이가 0이면 3회 재요청
            retry_prompt(result, prompt_command)

            # chatGPT를 통해 받은 결과를 한국어로 다시 번역
            result_to_kr = translator.translate(result, dest='ko')
            st.write("chatGPT 요청 결과(한국어로 번역):", result_to_kr.text)

            # 네이버 블로그 크롤링 완료시 count 증가
            crawling_done_count = crawling_done_count + 1
            st.write("crawling_done_count :",crawling_done_count)

            st.write("-"*50)

        # 네이버 포스트인 경우
        elif post_m:
            # 포스트 전용 크롤러 실행
            original_crawling_result = naver_post_crawler_for_naver_search(post_link)

            # 크롤링한 데이터를 번역(en 이외의 언어인 경우)
            for_prompt_data_one = google_translator(original_crawling_result)

            st.write("chatGPT 실행중...")

            # 사용자가 입력한 프롬프트와 en으로 번역된 크롤링 데이터를 chatGPT 프롬프트에 전송
            prompt_command = input_prompt + " " + for_prompt_data_one
            # st.write("prompt_command:",prompt_command)
            result = chatGPT(prompt_command).strip()  # 왼쪽, 오른쪽 공백 제거
            st.write("결과 원문:", result)
            st.write("결과 길이:", len(result))

            # chatGPT 결과의 길이가 0이면 3회 재요청
            retry_prompt(result, prompt_command)

            # chatGPT를 통해 받은 결과를 한국어로 다시 번역
            result_to_kr = translator.translate(result, dest='ko')
            st.write("chatGPT 요청 결과(한국어로 번역):", result_to_kr.text)

            # 네이버 포스트 크롤링 완료시 count 증가
            crawling_done_count = crawling_done_count + 1
            st.write("crawling_done_count :",crawling_done_count)

            st.write("-"*50)


        # 네이버 카페인 경우
        elif cafe_m:
            st.write("네이버 카페는 크롤링이 불가능합니다.") # 접근 권한 문제로 인해 크롤링 불가

        else:
            st.write("이 프로그램에서 크롤링 가능한 글이 아닙니다.")
    else:
        
        st.write("정상 크롤링 3회 완료.")
        break



# 브런치 링크 크롤러
# 구글 검색결과를 크롤링해오는 과정에서 선언한 browser 그대로 사용
def brunch_crawler(url, playwright, browser):
    st.write("브런치 링크입니다.")

    # playwright = sync_playwright().start() # 크롤링용 플레이라이트 시작 - 중복 선언 불가
    browser = playwright.chromium.launch(headless=True) 
    page = browser.new_page() # 웹브라우저 열기
    page.goto(url) # 파라미터로 받은 url 주소로 이동
    ctx = page.locator('xpath=/html/body') # 오픈한 페이지에서 가져올 html code path 지정(body로 지정해서 전체 가져옴)
    ctx_inner_html = ctx.inner_html() # 조회한 페이지에서 html 코드 가져오기
    # print(ctx_inner_html)

    # 브런치 게시글의 제목을 가져오기 위함
    # 제목 태그명을 선택한 뒤 class 이름 붙여서 접근
    title_tag = page.locator('h1.cover_title')
    title_text = title_tag.inner_html().replace("  ","")
    st.write("제목 : ", title_text)

    # 앞을 잘라내기 위해 브런치 title 태그 지정
    # 해당 태그 다음부터 본문의 내용이 시작된다(제목, 본문)
    under_title_tag_location = ctx_inner_html.find("<h1 class=\"cover_title")

    # 뒷부분을 잘라내기 위해 브런치 end 태그 지정
    # 해당 태그가 위치한 곳부터 내용을 잘라낸다(본문이 끝난 이후 바로 다음에 등장하는 태그로 지정)
    over_end_tag_location = ctx_inner_html.find("<div class=\"wrap_body_info") 

    # 조회한 태그의 위치만큼 html 코드를 잘라내기
    tag_data = ctx_inner_html[under_title_tag_location:over_end_tag_location]

    # 잘라낸 html 코드에서 tag 데이터를 지우고 순수 텍스트만 가져오기
    cleantext = BeautifulSoup(tag_data, "lxml").text
    brunch_cleantext = cleantext.replace("\n\n", "")# 불필요한 공백(엔터) 제거 후 출력확인
    st.write("본문 : ")
    st.write(brunch_cleantext)

    browser.close()
    print("-------------------완료------------------")

    return brunch_cleantext

# 구글검색용) 네이버 블로그 - iframe 제거 후 blog.naver.com 붙이기
# def naver_blog_delete_iframe(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"}
    res = requests.get(url, headers=headers)
    res.raise_for_status()  # 문제시 프로그램 종료
    soup = BeautifulSoup(res.text, "lxml")

    src_url = "https://blog.naver.com/" + soup.iframe["src"]

    return src_url

# 구글검색용) 네이버블로그 본문 스크래핑 - 스마트 에디터 2.0, ONE 포함
# def naver_blog_text_scraping(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"}
    res = requests.get(url, headers=headers)
    res.raise_for_status()  # 문제시 프로그램 종료
    soup = BeautifulSoup(res.text, "lxml")

    if soup.find("div", attrs={"class": "se-main-container"}): # 본문 전체 태그
        text = soup.find(
            "div", attrs={"class": "se-main-container"}).get_text()
        text = text.replace("\n", "") 
        return text

    elif soup.find("div", attrs={"id": "postViewArea"}):  # 스마트 에디터 2.0도 크롤링 해오기 위함
        text = soup.find("div", attrs={"id": "postViewArea"}).get_text()
        text = text.replace("\n", "")
        return text


    else:
        return "네이버 블로그는 맞지만, 확인이 불가한 데이터입니다."  # 크롤링 안되는 글 처리

# 구글검색용) 네이버 블로그 크롤링해서 출력
def naver_blog_crawler(url):
    st.write("네이버블로그 링크입니다.")

    blog_text = text_scraping(delete_iframe(url))
    print("네이버 블로그 본문 : ", blog_text)
    st.write(blog_text)

    print("-------------------완료------------------")

    return blog_text


# newspaper 라이브러리를 사용한 크롤러
def newspaper_crawler(url):
    # 입력받은 url을 통해 크롤링 해오기 위한 코드 - newspaper Article 사용
    url_str = str(url)
    url_strip = url_str.strip()
    page = Article(url_strip, config=config) # 언어설정을 하지 않으면 자동감지된다.
    page.download()
    page.parse() # 웹페이지 parse 진행

    # 크롤링 된 결과의 본문(text)을 변수에 따로 저장
    url_crawling_text = page.text
    url_crawling_title = page.title

    # 크롤링 원본 결과 출력
    st.write("게시글 제목 : ", url_crawling_title)
    st.write("크롤링 결과 : ")
    st.write(url_crawling_text)

    return url_crawling_text


# 유튜브 링크인 경우 크롤링 해주는 함수
def youtube_script_crawling(id):
    st.write("유튜브 스크립트 크롤링 실행중...")

    # url과 video id를 통해 제목을 크롤링해올 동영상 주소에 접근
    params = {"format": "json", "url": "https://www.youtube.com/watch?v=%s" % id}
    url = "https://www.youtube.com/oembed"
    query_string = urllib.parse.urlencode(params)
    url = url + "?" + query_string

    # 동영상 title을 가져오기 위함
    with urllib.request.urlopen(url) as response:
        response_text = response.read()
        data = json.loads(response_text.decode())
        video_title = data['title']
        # 응답받은 값에서 제목 데이터를 가져옴
        st.write("영상 제목 : ", video_title)

    # 프로그램 시작
    srt1 = YouTubeTranscriptApi.get_transcript(id, languages=['ko'])

    text = ''

    for i in range(len(srt1)):
        text += srt1[i]['text'] + ' '  # text 부분만 가져옴

    # 공백 제거
    text_ = text.replace(' ', '')

    kkma = Kkma()
    text_sentences = kkma.sentences(text_)

    # 문장이 끝나는 어절을 임의로 지정해서 분리할 용도로 사용
    lst = ['죠', '다', '요', '시오', '습니까', '십니까', '됩니까', '옵니까', '뭡니까',]

    # 제외하고자 하는 단어 파일을 읽어들여서 for문에 사용
    df = pd.read_csv('not2.csv', encoding='utf-8')
    not_verb = df.loc[:, 'col1'].to_list()

    text_all = ' '.join(text_sentences).split(' ')

    for n in range(len(text_all)):
        i = text_all[n]
        if len(i) == 1:  # 한글자일 경우 추가로 작업하지 않음
            continue

        else:
            for j in lst:  # 종결 단어
                # 질문형
                if j in lst[4:]:
                    i += '?'

                # 명령형
                elif j == '시오':
                    i += '!'

                # 마침표
                else:
                    if i in not_verb:  # 특정 단어 제외
                        continue

                    else:
                        if j == i[len(i)-1]:  # 종결
                            text_all[n] += '.'
                            # print("\n", text_all[n], end='/ ')

    # 나뉜 문장을 다시 조인
    spacing = Spacing()
    text_all_in_one = ' '.join(text_all)
    result = spacing(text_all_in_one.replace(' ', ''))
    st.write(result) # 원본 크롤링 데이터 출력

    return result


# 크롤링된 결과의 언어를 자동 감지하고 번역해주는 함수
def google_translator(original_crawling_data):
    # detect 라이브러리를 통해 텍스트의 언어 자동감지
    language_sensing = detect(original_crawling_data)
    st.write("언어감지 : ", language_sensing)

    # 크롤링한 언어가 한국어, 일본어이면 구글 번역기 실행(한글,일본어만 -> 영어)
    if language_sensing == 'ko' or language_sensing == 'ja':
        url_crawling_slice = original_crawling_data[0:5000] # 구글번역기 최대 제한수로 인한 슬라이스

        # 영어로 번역
        crawling_data_translate = translator.translate(url_crawling_slice, dest='en')

        # 프롬프트에 날리기 위해 정리된 크롤링 데이터 변수에 저장
        for_prompt_data = crawling_data_translate.text

        # 번역된 결과 출력
        st.write("----- url1 크롤링 결과 번역(en) -----")
        st.write(for_prompt_data)

    # 크롤링한 언어가 영어면 번역 없이 그대로 진행 - 긴 경우 슬라이스만
    elif language_sensing == 'en':
        for_prompt_data = original_crawling_data[0:10000] # chatgpt 최대 제한수로 인한 슬라이스
    
    # 크롤링 되지 않은 경우 == 데이터가 없는 경우 -> 에러처리
    else:
        st.error("크롤링된 값이 너무 적거나 모호해서 언어를 감지할 수 없습니다. 다른 url을 입력해주세요.", icon="🚨")


    return for_prompt_data

def retry_prompt(answer, prompt):
    # 응답된 결과의 길이가 0이면 (없으면) 3회까지 다시 시도
    if len(answer) == 0:
        st.write("[Error] chatGPT의 답변이 정상적으로 생성되지 않았습니다.")
        st.write("5초 후 자동으로 재요청됩니다. (1/3).")
        time.sleep(5)

        first_re_answer = chatGPT(prompt).strip()
        st.write("chatGPT 1차 재요청 결과의 길이 : ", len(first_re_answer))
        st.write("chatGPT 1차 재요청 결과: ", first_re_answer)

        if len(first_re_answer) == 0:
            st.write("[Error] chatGPT의 답변이 정상적으로 생성되지 않았습니다.")
            st.write("10초 후 자동으로 재요청됩니다. (2/3).")
            time.sleep(10)

            second_re_answer = chatGPT(prompt).strip()
            st.write("chatGPT 2차 재요청 결과의 길이 : ", len(second_re_answer))
            st.write("chatGPT 2차 재요청 결과: ", second_re_answer)

            if len(second_re_answer) == 0:
                st.write("[Error] chatGPT의 답변이 정상적으로 생성되지 않았습니다.")
                st.write("15초 후 자동으로 재요청됩니다. (3/3).")
                time.sleep(15)

                third_re_answer = chatGPT(prompt).strip()
                st.write("chatGPT 3차 재요청 결과의 길이 : ", len(third_re_answer))
                st.write("chatGPT 3차 재요청 결과: ", third_re_answer)

                if len(third_re_answer) == 0:
                    st.write("[Error] 잠시 후 재요청 해주시기 바랍니다.")


# 구글 검색시 실행되는 함수
def google_search_crawling(input_data):
  crawling_done_count = 0
  original_crawling_result =""
  st.write("구글 크롤링 실행중...")

  # 구글 검색시의 게시글들의 url 값 가져오기
  baseUrl = 'https://www.google.com/search?q='
  # plusUrl = input_data
  # url = baseUrl + quote(plusUrl)
  # 검색어와 base url 합치기
  url = baseUrl + input_data 
  # 한글은 인터넷에서 바로 사용하는 방식이 아니라, quote_plus가 변환해줌 -> URL에 %CE%GD%EC 이런 거 만들어줌

  #  플레이라이트로 변경
  playwright = sync_playwright().start()
  browser = playwright.chromium.launch(headless=True, channel="chrome") 
  context = browser.new_context() # 시크릿
  page = context.new_page() # 브라우저 실행하기
  page.goto(url) # 입력한 검색어의 검색결과 탭으로 이동
  print("\n------------------------------------------------\n")
  print(page.content())
  html = page.content() 
  soup = BeautifulSoup(html, "lxml") # 구글 검색결과 화면의 html 코드 가져와서 저장

  # 셀렉트할 class명 지정
  # 검색결과 리스트를 지칭하는 class : MjjYud -> 광고태그는 제외한 모든 게시글을 긁어옴 (이미지, 영상 포함인듯) - 확인 필요
  google_search_result_list = soup.select('.MjjYud')[:10] # 상위 3개 가져옴 - limit과 비슷 ..


  # 동영상 -> X5OiLe 안의 href값 가져와야함
  # 기존과 동일하게 MjjYud로 가져오면 #fpstate=ive&vld=cid:~~형태의 값이 href에 들어있음
  # 임시방편으로 ... #fpstate=ive&vld=cid:라는 글자가 있으면 다시 셀렉트해서 href를 가져오도록 수정
    # browser.close() # 종료 후 close 필요

  # 게시글의 list를 반복 순회하며 제목 태그의 텍스트 값과 url 값을 출력함
  for i in google_search_result_list:


    # 정상적으로 크롤링 된 횟수가 3회까지 가능하도록 조건문 설정
    if crawling_done_count < 3:

        # url 값만 가져오기
        print(" 구글 url : ", i.a.attrs['href'])
        st.write("구글 검색 결과 url : ", i.a.attrs['href'])
        link = i.a.attrs['href']

        if "https" not in link:
            st.write("크롤링 가능한 링크가 아닙니다. (게시글이 아님)")
            continue
        
        # 동영상 링크 인식하기
        # 에러가 났다가 말았다가 함
        elif '#fpstate=ive&vld=cid:' in link:
            video_list = soup.select('.X5OiLe')[:1] # 일단 한개만 가져오게 설정

            for i in video_list:
                link = i.a.attrs['href']
                st.write("구글 url(video) : ", i.a.attrs['href'])

        # url domain 값으로 어떤 사이트인지 판단
        # 1. 유튜브 url인 경우
        elif 'youtube' in link:
            # 확인용 안내문구
            st.write("크롤링 작업 진행중...")
            st.write("유튜브 링크입니다.")
            youtube_url = link
            search_value = "v="
            index_no = youtube_url.find(search_value)
            video_id = youtube_url[index_no+2:]

            print("video_id:",video_id)
            st.write("video_id:",video_id)

            # 유튜브 스크립트 크롤링 후 원본 결과를 변수에 저장
            original_crawling_result = youtube_script_crawling(video_id)

        # 2. 브런치인 경우
        elif 'brunch' in link:
            # 확인용 안내문구
            st.write("크롤링 작업 진행중...")
            # 브런치 글 크롤링 후 원본 결과를 변수에 저장
            original_crawling_result = brunch_crawler(link, playwright, browser)

        # 3. 네이버 블로그인 경우
        elif 'blog.naver.com' in link:
            # 확인용 안내문구
            st.write("크롤링 작업 진행중...")
            # 네이버블로그 글을 크롤링 후 원본 결과를 변수에 저장
            original_crawling_result = naver_blog_crawler(link)

        # 4. 네이버 포스트인 경우
        elif 'post.naver.com' in link:
            # 확인용 안내문구
            st.write("크롤링 작업 진행중...")
            # 네이버 포스트 글을 크롤링 후 원본 결과를 변수에 저장
            original_crawling_result = naver_post_crawler_for_google_search(link, playwright, browser)

        # 5. 수동으로 제외할 사이트 명시하기 - 현재는 나무위키만 지정
        elif 'namu' in link:
            st.write("크롤링이 불가능한 링크입니다. (나무위키)")
            continue

        # 6. 이외의 경우
        else:

            # 확인용 안내문구
            st.write("기타 링크입니다.")
            st.write("크롤링 작업 진행중...")

            original_crawling_result = newspaper_crawler(link)


        if len(original_crawling_result) == 0:
            st.write("크롤링이 정상적으로 동작하지 않았습니다. 크롤링이 불가능한 주소입니다.") # 여기다가 그냥 순수 텍스트 모두 긁어오는 코드 추가하기
            continue
            
        # 크롤링한 데이터를 번역(en 이외의 언어인 경우)
        for_prompt_data_one = google_translator(original_crawling_result)

        st.write("chatGPT 실행중...")

        # 사용자가 입력한 명령어(prompt)와 slice한 string 값을 붙여서 chatGPT에 날리는 작업
        prompt_command = input_prompt + " " + for_prompt_data_one
        # st.write(input_prompt)
        # st.write("prompt_command:",prompt_command)
        result = chatGPT(prompt_command).strip()  # 왼쪽, 오른쪽 공백 제거
        st.write("결과 원문:", result)
        st.write("결과 길이:", len(result))

        # chatGPT 결과의 길이가 0이면 3회 재요청
        retry_prompt(result, prompt_command)

        # chatGPT를 통해 받은 결과를 한국어로 다시 번역
        result_to_kr = translator.translate(result, dest='ko')
        st.write("chatGPT 요청 결과(한국어로 번역):", result_to_kr.text)
        
        crawling_done_count = crawling_done_count + 1
        st.write("crawling_done_count :",crawling_done_count)

        st.write("-"*50)

    else:
        st.write("크롤링 3회 완료.")
        break

# chatGPT에 요청을 다시 날리는 재요청 버튼 - 안씀
def chatGPT_retry_button(prompt):

  disabled_flag = True
  time.sleep(5)
  disabled_flag = False

  if st.button("재요청", disabled=disabled_flag):
    
    re_answer = chatGPT(prompt).strip()
    st.write("chatGPT 재요청 결과 : ", re_answer)
  
  else:
      st.write(" ")


# 네이버 크롤링 실행용 함수 - 버튼 클릭시 작동
def start_naver_search_button(input_data):
    # 입력값이 아무것도 없으면 버튼 비활성화
    disabled_flag = True

    # 입력값이 생기면 버튼 활성화
    if len(input_data) != 0:
        disabled_flag = False

    # 버튼 클릭시 실행
    if st.button("네이버 크롤링 시작", disabled=disabled_flag):
      st.write("네이버 시작")

      naver_crawling(input_data)

      st.write("완료")

    else:
      st.write(" ")

# 구글 크롤링 실행용 함수 - 버튼 클릭시 작동
def start_google_search_button(input_data):
    # 입력값이 아무것도 없으면 버튼 비활성화
    disabled_flag = True

    # 입력값이 생기면 버튼 활성화
    if len(input_data) != 0:
        disabled_flag = False

    # 버튼 클릭시 실행
    if st.button("구글 크롤링 시작", disabled=disabled_flag):
      st.write("구글 시작")

      google_search_crawling(input_data)

      
      st.write("완료")

    else:
      st.write(" ")



# 제목
st.title('검색엔진에 검색한 결과를 크롤링해서 chatGPT에 명령하기')

# 안내 문구
# chatGPT_execution_confirmation()
# st.info('크롤링 가능 언어 : 한국어(ko), 영어(en), 일본어(ja) / 자동인식됨', icon="ℹ️")

# 텍스트
st.write("기능1) 네이버에 검색할 내용을 입력해주세요.")
input_naver = st.text_input(label='네이버 검색')
st.write("입력확인 : ", input_naver)
st.write("")
st.write("기능2) 구글에 검색할 내용을 입력해주세요.")
input_google = st.text_input(label='구글 검색')
st.write("입력확인 : ", input_google)
st.write("")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
st.write("ChatGPT에 요청할 prompt를 입력하세요. (기본값 : Summarize the following)")
input_prompt = st.text_input(label='ex) Summarize the following ...', value="Summarize the following ")
st.write("입력확인 : ", input_prompt)

# 크롤링 실행 버튼
start_naver_search_button(input_naver)
start_google_search_button(input_google)


