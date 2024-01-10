import streamlit as st
from bs4 import BeautifulSoup
from utils import *
from keybert import KeyBERT
from transformers import BertModel
import openai


def generate_response(prompt):
    messages = []
    messages.append({"role": "user", "content": prompt})
    print("\ngenerate start.")
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
    )
    print("finished.")
    response = completion.choices[0].message.content
    return response


def transfer_text_style(texts, target_style, API, headers, max_length):
    flag = True
    print("\n문체를 변환할 텍스트:", texts)
    while flag:
        msg = []
        for text in texts:
            inputs = f"{target_style} 말투로 변환:{text[0]}"
            msg.append(query(API, headers, {"inputs": inputs}, max_length))
        print("\n변환 완료: ", msg)
        try:
            if "error" in msg[0][0].keys():
                print("변환 오류. 다시 변환")
            else:
                flag = False
        except:
            if "error" in msg[0].keys():
                print("변환 오류. 다시 변환")
            else:
                flag = False
    return msg


# side bar
st.sidebar.title("고급 기능🌸")
select_species = st.sidebar.selectbox("어떤 말투로 설명해드릴까요?", ["원문", "구어체", "나루토", "enfp"])
st.sidebar.markdown("---")
slider_range = st.sidebar.slider("추가 질문은 얼마나 할까요?", 0.0, 1.0, (0.5))  # 시작 값  # 끝 값
st.sidebar.markdown("---")

st.header("말랑뉴스 🧠")
st.markdown("뉴스를 풀어서 설명해드려요.")

# 사용자로부터 URL 입력 받기
with st.form("form", clear_on_submit=False):
    print("start")
    url = st.text_input("URL: ", "", key="input_url")
    submitted = st.form_submit_button("Submit")

if submitted:
    try:
        with st.spinner("### API를 호출하는 중... 잠시만 기다려주세요... 🤔"):
            API_TOKEN = "INSERT YOUR HUGGINGFACE API KEY"
            API_KEY = "INSERT YOUR OPENAI AIP KEY"
            headers = {"Authorization": f"Bearer {API_TOKEN}"}
            # kobart
            API_URL_kobart = (
                "https://api-inference.huggingface.co/models/ainize/kobart-news"
            )
            # koalpaca
            API_URL_koalpaca = "https://api-inference.huggingface.co/models/beomi/KoAlpaca-Polyglot-12.8B"
            # keybert
            model = BertModel.from_pretrained("skt/kobert-base-v1")
            kw_model = KeyBERT(model)
            # korean smilestyle dataset
            API_URL_korean = "https://api-inference.huggingface.co/models/heegyu/kobart-text-style-transfer"
            # gpt
            openai.api_key = API_KEY
        with st.spinner("### 뉴스를 가져오는 중... 잠시만 기다려주세요... 🤔"):
            # beutifulsoup4로 url 접속 및 텍스트 크롤링
            response = requests.get(url)
            html_content = response.text
            soup = BeautifulSoup(html_content, "html.parser")
            title = soup.find("h2", class_="media_end_head_headline")  # 제목
            writer = soup.find("em", class_="media_end_head_journalist_name")  # 기자
            datestamp = soup.find(
                "div", class_="media_end_head_info_datestamp"
            )  # 입력 수정
            article = soup.find("article", class_="go_trans _article_content")  # 기사 본문
            for br in article.find_all("br"):
                br.replace_with("\n")
            article_text = article.get_text()
            title = title.get_text()
            # <article> 태그 내의 내용 중 <em> 태그(주석) 제외
            exclude_em = article.find_all("em")
            for em_tag in exclude_em:
                article_text = article_text.replace(em_tag.get_text(), " ")

            # <article> 태그 내의 내용 중 <strong> 태그(강조 반복) 제외
            exclude_strong = article.find_all("strong")
            for strong_tag in exclude_strong:
                article_text = article_text.replace(strong_tag.get_text(), " ")

            text = preprocessing_text(article_text)  # 모델 학습과 동일한 전처리
            text = list2str(text)  # string으로 변환
        try:
            with st.spinner("### 뉴스를 요약하는 중... 잠시만 기다려주세요... 🤔"):
                output = query(API_URL_kobart, headers, {"inputs": text})  # 추출 요약
                summary = output[0]["summary_text"]
            with st.spinner("### 질문을 생성하는 중... 잠시만 기다려주세요... 🤔"):
                if not select_species == "원문":
                    ms = transfer_text_style(
                        [[summary]],
                        select_species,
                        API_URL_korean,
                        headers,
                        max_length=1024,
                    )
                    summary = ms[0][0]["generated_text"]
                nouns = noun_extractor(summary)  # 명사 추출
                preprocessed_text = " ".join(nouns)
                keywords_items = kw_model.extract_keywords(
                    preprocessed_text,
                    keyphrase_ngram_range=(1, 1),
                    stop_words=None,
                    top_n=10,
                )  # 키워드 추출
                question_word = []
                for i, keyword in enumerate(keywords_items):
                    if i == 0:
                        keywords = keyword[0]
                    else:
                        keywords += ", " + keyword[0]
                    if keyword[1] > 1 - slider_range:
                        question_word.append(keyword[0])
                que, real_que = question_query(question_word)
            with st.spinner("### 답변을 생성하는 중... 잠시만 기다려주세요... 🤔"):
                print("\n답변을 생성할 키워드:", question_word)
                real_answer = generate_response(text + real_que)
                print("GPT가 생성한 답변:", real_answer)
                real_answers = []
                if "(Q)" in real_answer:
                    for qna in real_answer.split("(Q)")[1 : 1 + len(question_word)]:
                        real_answers.append(qna.split("(A) ")[1:])
                else:
                    for qna in real_answer.split("(A) ")[1 : 1 + len(question_word)]:
                        real_answers.append([qna])
                print("\nSplit을 적용한 답변:", real_answers)
                if not select_species == "원문":
                    msg = transfer_text_style(
                        real_answers,
                        select_species,
                        API_URL_korean,
                        headers,
                        max_length=1024,
                    )
                    real_answers = []
                    print(msg)
                    for ms in msg:
                        real_answers.append(ms[0]["generated_text"])
                    real_answers = [real_answers]
                print("문체 변환 완료: ", real_answers)
                datetime = ""
                p_datetime = preprocessing_text(datestamp.get_text())  # datestamp도 전처리
                for pdt in p_datetime:
                    if "입력" in pdt or "수정" in pdt:
                        datetime += pdt + " "
                wrt = " " if not writer else writer.get_text()
                st.write(f"#### {title}")
                st.markdown(
                    f'<p style="color: grey; font-size: 12px;">원문: {datetime} {wrt} <br>말랑뉴스에 의해 요약된 뉴스입니다.</p>',
                    unsafe_allow_html=True,
                )

                st.write(summary)
                print("제목 및 요약문 출력 완료")
                for i in range(len(que)):
                    st.write(f"##### {que[i]}")
                    if not select_species == "원문":
                        st.write(f"🧠 {postprocessing_text(real_answers[0][i])}")
                    else:
                        st.write(f"🧠 {postprocessing_text(real_answers[i][0])}")
                print("질문과 답변 출력 완료")
                # st.write(keywords_items)
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown(
                    f'<p style="color: grey; font-size: 12px;">문서에서 추출된 키워드: {keywords}<br>추출된 키워드에 대해 설명이 필요하다면 추가 질문 정도를 높여보세요!</p>',
                    unsafe_allow_html=True,
                )
                st.balloons()
        except:
            st.write("#### 아직 모델을 불러오는 중이에요.")
            st.write("#### 잠시 뒤에 URL을 다시 입력해주세요. 😭")
    except:
        st.write("#### 뉴스를 찾을 수 없어요.")
        st.write("#### URL을 다시 입력해주세요. 🙄")
else:
    pass
