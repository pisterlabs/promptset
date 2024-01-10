from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
from deepl import Translator
import os
import requests
from bs4 import BeautifulSoup as bs
from data import SecondhandPost

load_dotenv()


## 중고나라 스크래핑 클래스
class Joongna:
    ## https://web.joongna.com/product/게시글번호
    #  중고나라 검색 url
    #  ?category=카테고리번호&page=페이지번호
    def __init__(self, cursor):
        # self.category_nums = [1150, 1151, 1152, 1153, 1154, 1155, 1156, 1193, 1194, 1195, 1196]
        self.category_nums = [1150]
        self.name = 'joongna'

        self.JOONGNA_URL = "https://web.joongna.com"
        self.JOONGNA_SEARCH_URL = self.JOONGNA_URL + "/search"
        self.JOONGNA_PRODUCT_URL = self.JOONGNA_URL + "/product"
        self.SORT_RECENT = 'RECENT_SORT'

        self.member_id = 1


        # self.category_names = {
        #     '스마트폰': {'삼성': 1150, '애플': 1151, 'LG': 1152, '기타': 1153},
        #     '태블릿': {'삼성': 1154, '애플': 1155, '기타': 1156}
        #     # ,'노트북': {'삼성': 1193, '애플': 1194, 'LG': 1195, '기타': 1196}
        # }

        ## DB에 저장된 IT 기기명 리스트
        sql_query = 'select device_name from device'
        cursor.execute(sql_query)
        _device_list = cursor.fetchall()

        device_list_ko = []
        ## 단어 리스트를 한 문장으로 변환
        text = ''
        for device in _device_list:
            device_list_ko.append(device[0])
            text += device[0] + ', '
        text = text[:-2]

        ## 기기명을 한글 -> 영어로 번역
        deepl_auth_key = os.environ.get('DEEPL_API_KEY')
        trans = Translator(deepl_auth_key)
        trans_res = trans.translate_text(text, target_lang='EN-US').text

        ## 영어 대문자로 띄워쓰기 없이
        ## 게시글 제목에서 추출한 기기명과 비교하기 위함
        self.device_list_en = "".join(trans_res.split()).upper().split(',')

        ## 추출한 기기명을 다시 DB에 저장된 기기명으로 바꿔주기 위한 딕셔너리
        self.device_list_ko_DB = {}
        for device_en, device_ko in zip(self.device_list_en, device_list_ko):
            self.device_list_ko_DB[device_en] = device_ko

        print(self.device_list_ko_DB)

    ## ChatGPT에 쿼리를 보내는 함수
    def _ask_gpt(self, query):

        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        model = 'gpt-4-1106-preview'
        # GPT에 질문하기
        completion = client.chat.completions.create(
            model=model,
            messages=[
                # {"role":"system", "content":""},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )

        # 생성된 응답 출력
        answer = completion.choices[0].message.content

        return answer

    ## GPT를 활용해 제목에서 기기명을 추출하는 함수
    ### 현재 테스트를 위해 기기명은 '기타'만 사용
    def extract_device_name(self, text):
        print('extract_device_name\n')

        ## GPT에 질문
        ## "다음 백틱 한 개로 구분된 글에서 백틱 세 개로 구분된 단어들 중에서 관련도가 가장 높은 하나를 표기하여 주세요. "
        ## "해당하는 단어가 없으면 '기타'를 표기해주세요. 다른 답변을 포함하지 말아주세요.\n"
        # "```"
        # +str(device_list) +
        # "```"
        gpt_query = ("당신의 역할은 주어진 백틱 세개로 구분된 문장 속에서 IT 기기의 이름을 찾아내는 것입니다.\n숫자로 주어지는 요구사항에 맞춰서 대답해주세요.\n"
                     "1. 기기명에는 용량과 통신사의 이름은 포함되지 않습니다.\n"
                     "2. 발견한 IT 기기명은 영어로 번역합니다.\n"
                     "3. 각 단어의 가장 앞 글자만 대문자로 표기합니다.\n"
                     "4. 다른 대답은 필요 없습니다. 기기명만을 큰 따옴표 안에 넣어 대답해주세요.\n\n"
                     "```\n"
                     +text+
                     "\n```\n")

        device_name = "".join(self._ask_gpt(gpt_query).split()).replace('"', '').strip().upper()
        # print(device_name)
        if device_name not in self.device_list_en:
            device_name = 'OTHER'
        # print(device_name)

        return device_name

    ## 게시된 시간을 알기 위해 변환하는 함수
    def parse_upload_time(self, time_text):
        if '초 전' in time_text:
            seconds_ago = int(time_text.split('초 전')[0])
            return datetime.now() - timedelta(seconds=seconds_ago)
        elif '분 전' in time_text:
            minutes_ago = int(time_text.split('분 전')[0])
            return datetime.now() - timedelta(minutes=minutes_ago)
        elif '시간 전' in time_text:
            hours_ago = int(time_text.split('시간 전')[0])
            return datetime.now() - timedelta(hours=hours_ago)
        elif '일 전' in time_text:
            days_ago = int(time_text.split('일 전')[0])
            return datetime.now() - timedelta(days=days_ago)
        else:
            # 처리할 수 없는 다른 형식의 경우 예외 처리
            raise ValueError("Unsupported time format")



    ########### 공통 함수

    ## 게시글 리스트에서 제품의 링크를 추출하는 함수
    def extract_link(self,category_num, page_num):
        print('extract_link\n')
        # 제품 링크 배열
        links = []
        link_location = {}

        # url 세팅
        category = "category=" + str(category_num)
        page = "page=" + str(page_num)
        sort = "sort="+ self.SORT_RECENT

        scraping_url = self.JOONGNA_SEARCH_URL + "?" + category + "&" + page + "&" + sort

        #  웹 페이지의 내용을 가져옴
        response = requests.get(scraping_url)
        soup = bs(response.text, 'html.parser')
        tags = soup.find_all('a', 'group box-border overflow-hidden flex rounded-md cursor-pointer pe-0 pb-2 lg:pb-3 flex-col items-start transition duration-200 ease-in-out transform hover:-translate-y-1 md:hover:-translate-y-1.5 hover:shadow-product bg-white')

        for tag in tags:
            link = tag['href']
            location_span = tag.find('span','text-gray-400')
            location = location_span.text.strip()
            product_url = self.JOONGNA_URL + link
            links.append(product_url)
            link_location[product_url] = location

        ## 중고나라는 게시글 리스트에서만 위치 정보를 추출할 수 있으므로 이 함수에서 위치데이터 또한 반환
        return links, link_location


    ## 게시글의 정보를 추출하여 데이터 클래스로 반환하는 함수
    def extract_text(self, post_url, location):
        print('extract_text\n')
        # 게시글 웹 페이지 스크래핑
        response = requests.get(post_url)
        soup = bs(response.text, 'html.parser')

        ### 스크래핑된 데이터를 데이터 클래스에 넣기 위해 변환
        post_id = 0

        title = soup.find('h1').text.strip()
        content = ''
        try:
            content = soup.find('article').text.strip()
        except Exception as e:
            print(f'no content {e}')

        ## time
        span_tag = soup.find('div', 'flex justify-between text-body').find('span')
        time = datetime.now()
        try:
            time = self.parse_upload_time(span_tag.text.strip())
        except Exception as e:
            print(f'no content {e}')

        ## img_address
        img_address = []
        img_tags = soup.find_all('img', 'object-cover w-full h-full rounded-lg top-1/2 left-1/2')
        try:
            for img_tag in img_tags:
                img_address.append(img_tag['src'])
        except Exception as e:
            print(f'no image {e}')

        ## price
        price_ = soup.find('div','text-heading font-bold text-[40px] pe-2 md:pe-0 lg:pe-2 2xl:pe-0 mr-2')
        price = 0
        try:
            price_text = price_.text.strip()
            price = int(''.join(filter(str.isdigit, price_text)))
        except Exception as e:
            print(f'no image {e}')


        device_name_en = self.extract_device_name(title)
        device_name = self.device_list_ko_DB[device_name_en]

        print('device name is '+device_name)

        post_like_count = 0
        post_view_count = 0
        url = os.environ.get('LOCATION_API')+'?keyword='+location
        res = requests.get(url)
        city = ''
        try:
            city = res.json()[0]['location']['city']
        except Exception as e:
            print(f'city를 입력하는 과정에서의 오류 - {e}')

        street = location
        zipcode = ''


        # member_id, device_id // post_update_time => DB 쿼리로 입력 // city, zipcode => 미사용
        data_class = SecondhandPost(post_id=post_id,post_title=title,post_content=content,post_time=time,
                       post_like_count=post_like_count,post_view_count=post_view_count,
                       device_name=device_name, member_id=self.member_id,
                       post_update_time=datetime.now(),
                       img_folder_address=str(img_address),
                       secondhand_price=price,post_url=post_url,
                       city=city, street=street, zipcode=zipcode, device_id=0)
        print(data_class)
        return data_class

# res = extract_text(post_url='https://web.joongna.com/product/144992312')
# print(res)


#### TEST

## 제품 링크 추출 확인
# links = extract_link(1150,1)
# print(links)

# res = _extract_device_name('이번에 산 갤럭시 a 13 팔아요')
# print("ans: "+res)

    ## db 세팅값
# load_dotenv()
# host = os.environ.get('DB_HOST')
# user = os.environ.get('DB_USER')
# password = os.environ.get('DB_PASSWORD')
# db_name = os.environ.get('DB_NAME')
# charset = os.environ.get('DB_CHARSET')
# port = int(os.environ.get('DB_PORT'))
#
# with pymysql.connect(host=host, user=user, password=password,
#                      database=db_name, charset=charset, port=port) as conn:
#     # Create a cursor object to interact with the database
#     with conn.cursor() as _cursor:
#         _post_url = 'https://web.joongna.com/product/144992312'
#         _location = ''
#         res = extract_text(post_url=_post_url, location=_location, cursor=_cursor)
#         print(res)