from selenium.common import NoSuchElementException, TimeoutException

from app.config.DriverConfig import DriverConfig
from app.config.DatabaseConfig import DatabaseConfig
import re

import html

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

import openai


class AnswerScrapper:
    def __init__(self):
        self.driver = DriverConfig().getDriver()
        self.wait = WebDriverWait(self.driver, 10)
        self.platform = ""
        self.problem_number = ""

    def __del__(self):
        self.driver.quit()

    def tag_validation(self, element):
        # 해당 element의 태그가 코드 관련인지 확인(코드 전체 내용을 긁어오기 위해)
        return element.tag_name == "code" or \
               element.tag_name == "pre" or \
               "code" in element.get_attribute('class') or \
               "code" in element.get_attribute('id')

    def start_answer_scrap(self, platform, problem_number):
        self.platform = platform
        self.problem_number = problem_number
        search_key_word = f"{platform} C++ {problem_number} 정답 코드"
        scrap_codes = self.get_scrap_codes(search_key_word)
        return scrap_codes

    def get_scrap_codes(self, search_key_word):
        self.driver.get("https://www.google.com/")
        search_box = self.driver.find_element(By.NAME, "q")
        search_box.send_keys(search_key_word)
        search_box.send_keys(Keys.RETURN)
        self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "h3")))
        scrap_codes = []
        success_count = 0
        for i in range(1, 11):
            scrap_code = self.scrap_try(i, self.wait)
            if scrap_code:
                scrap_codes.append(scrap_code)
                success_count += 1
            if success_count == 3:
                break
        return scrap_codes

    def scrap_try(self, page_number, wait):
        is_code = False
        answer_code = None
        try:
            link = self.wait.until(EC.element_to_be_clickable(
                (By.XPATH, f'//*[@id="rso"]/div[{page_number}]/div/div/div[1]/div/div/span/a/h3')))
        except NoSuchElementException:
            # 요소가 존재하지 않으면 여기서 함수를 종료합니다.
            return answer_code
        except TimeoutException:
            return answer_code
        if self.problem_number not in link.text:
            # 제목에 문제 번호가 포함되어 있지 않는 경우 더이상 진행하지 않음.
            return answer_code
        link.click()
        # page_number 번째 게시글 클릭
        elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'include')]")
        for el in elements:
            parent_el = el.find_element(By.XPATH, '..')
            # 부모 element 찾음.
            while parent_el.tag_name != "html":
                # 최상위 태그 도달할 때까지 반복
                if self.tag_validation(parent_el):
                    soup = BeautifulSoup(parent_el.get_attribute('outerHTML'), 'html.parser')
                    answer_code = html.unescape(soup.text.replace(" ", "").replace("\n", ""))
                    answer_code = re.sub(r'//.*?\n|/\*.*?\*/', '', answer_code, flags=re.DOTALL)
                    is_code = True
                    break
                parent_el = parent_el.find_element(By.XPATH, '..')
            if is_code:
                break
        self.driver.back()
        return answer_code

    def make_message(self, scrap_codes):
        request_message = ""
        num = 1
        for message in scrap_codes:
            request_message = request_message + f"{num} : {message}\n"
            num += 1
        request_message = request_message + f"위 {len(scrap_codes)}개의 코드를 참고해서 {self.platform} C++ {self.problem_number}번 정답 코드를 코드 형태로 보여줘"
        return request_message

class NetworkCommunication:
    #GPT와 통신하는 역할
    _isKey = False
    def __new__(cls):
        if not cls._isKey:
            openai.api_key = "YOUR_KEY"   # Open ai api key 입력
            _isKey = True

    @classmethod
    def request_gpt(cls, message):
        if not cls._isKey:
            cls()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": message}
            ]
        )
        reponse_message = response.choices[0].message['content']

        # 정규표현식을 사용하여 #include로 시작하여 return 0;으로 끝나는 부분을 찾습니다. - 백준 문제 코드 추출용
        match = re.search(r'#include.*?int main.*?return 0;.*?}', reponse_message, re.DOTALL)
        if match:
            extracted_code = match.group(0)
            return extracted_code
        # 프로그래머스 문제 코드 추출용
        match2 = re.search(r'#include.*?solution.*?return ans.*?}', reponse_message, re.DOTALL)
        if match2:
            return match2.group(0)
        return reponse_message


def main():
    korean_name = {
        "baekjoon": "백준",
        "programmers": "프로그래머스",
        "codeforce": "코드포스"
    }
    answer_scrapper = AnswerScrapper()
    db_conn = DatabaseConfig().getConnection()
    cursor = db_conn.cursor()

    while True:

        # 이미 해당 값이 존재하는지 확인
        select_sql = """
        SELECT 
            problem.id AS problem_id,
            problem.code AS problem_code, 
            platform.name AS platform_name,
            problem.name AS problem_name
        FROM problem
        JOIN platform ON problem.platform_id = platform.id
        LEFT JOIN solution ON problem.id = solution.problem_id
        WHERE platform_id = 1 AND problem.solved_count >= 1000 AND solution.problem_id IS NULL
        ORDER BY problem.solved_count DESC
        """

        cursor.execute(select_sql)

        # 결과 가져오기 및 출력
        problem_id, problem_code, platform_name, problem_name = cursor.fetchone()

        platform_name = korean_name[platform_name]
        print(f"problem_id : {problem_id}\n{platform_name}의 {problem_code}번 문제 정답 코드 크롤링 시작합니다.")

        try:
            if platform_name == "프로그래머스":
                problem_code = problem_name
            scrap_codes = answer_scrapper.start_answer_scrap(platform_name, problem_code)
            if len(scrap_codes) == 0:
                raise Exception
            message = answer_scrapper.make_message(scrap_codes)
            print(message)
            new_code = NetworkCommunication.request_gpt(message)
            if not new_code:
                raise Exception
            print(new_code)

        except Exception as e:
            # 실패 시 로깅
            print(f"problem_id : {problem_id}\n{platform_name}의 {problem_code}번 문제 정답 코드 크롤링에 실패했습니다.")
            new_code = "error"
            # 여기서 your_problem_number는 실패한 문제의 번호입니다.

        # INSERT 쿼리 실행
        insert_sql = "INSERT INTO solution (problem_id, programming_language_id, code) VALUES (%s, 1, %s)"
        cursor.execute(insert_sql, (problem_id, new_code))
        db_conn.commit()  # 중요: 데이터베이스에 변경 사항을 저장합니다.


    # DB 연결 끊기
    DatabaseConfig().getConnection().close()

    # 셀레니움 웹 드라이버 종료
    answer_scrapper.driver.quit()


if __name__ == "__main__":
    main()
