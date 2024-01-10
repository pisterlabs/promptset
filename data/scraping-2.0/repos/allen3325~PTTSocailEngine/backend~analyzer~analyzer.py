from entity.resultDTO import ResultDTO
import openai
import os
from dotenv import load_dotenv
from article_fetcher.article_fetcher import Article_Fetcher
from text_cleaner.text_cleaner import Text_Cleaner
from db.db_connect import result_collection

class Analyzer:
    # This global variable is to save GPT's response. Use UUID check response,
    response_db = {} # {UUID : Response}
    
    def __init__(self) -> None:
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.current_directory = os.getcwd()
        self.AF = Article_Fetcher()
        self.TC = Text_Cleaner()
        self.check_folder()
        self.delete_txt()

    def delete_txt(self):
        with open("./result/prompt_summary.txt", "w") as file:
            file.write("")
        with open("./result/GPT_report.txt", "w") as file:
            file.write("")

    def check_folder(self):
        # 檢查 result 資料夾是否存在
        result_folder = "result"
        if not os.path.exists(result_folder):
            # 如果不存在，則創建資料夾
            os.makedirs(result_folder)
            print(f"已創建 {result_folder} 資料夾")
        else:
            print(f"{result_folder} 資料夾已存在")

    # Generate response and save to text file
    def prompt_input(self, system: str, prompt: str):
        print("=============== Call ChatGPT ===============")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "system",
                    "content": system,
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            max_tokens=2048,
        )
        print(response["choices"][0]["message"]["content"])

        with open("./result/prompt_summary.txt", "a") as file:
            file.write(response["choices"][0]["message"]["content"] + "\n")

        return response["choices"][0]["message"]["content"]

    def prompt_report(self, system: str, prompt: str):
        print("=============== Call ChatGPT ===============")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "system",
                    "content": system,
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=1,
        )
        print(response["choices"][0]["message"]["content"])

        with open("./result/GPT_report.txt", "a") as file:
            response["choices"][0]["message"]["content"] = response["choices"][0][
                "message"
            ]["content"].replace("/\n/g", "<br>")
            file.write(response["choices"][0]["message"]["content"] + "\n")

        return response["choices"][0]["message"]["content"]

    """
    --------------- OLD
    frontend -> call API -> get response
    api -> keyword -> Article -> call GPT(100 s ) -> return GPT response
    --------------- NEW
    frontend -> generate UUID -> call generate_report API without return -> wait 90s in javascript -> polling check_API per second
    check_API -> UUID -> find result in response_dictionary[UUID] -> return response_dictionary[UUID]
    
    DOC
    Get UUID from frontend and save response in dictionary by UUID.
    Avoid cloudflare 100 seconds timeout.
    """

    def prompt_analyzer(
        self, keyword: str, tag: str, K: int, size: int, start: int, end: int, uuid: str
    ):
        print(f"------------------- start of {uuid} session.")
        contents = []
        comments = []
        # Read article's summary to list
        content_and_comment = ""
        content_and_comment_list = []

        # Create result model
        result = ResultDTO(
            result="initial_result",
            keyword=keyword,
            date=str(start) + '-' + str(end),
            session_id=uuid
        )

        # Retrieve related article
        prompt_AF = self.AF.get_article_by_keyword(
            keyword=keyword, tag=tag, K=K, size=size, start=start, end=end
        )

        # Extract all content and content
        for article in prompt_AF:
            article.content = self.TC.clean_text(article.content)
            contents.append(article.content)
            comments.append(article.get_all_comment_list())

        # Generate article's summary to text file
        for i in range(len(contents)):
            # new prompt
            # prompt = """現在給你[文章]以及[留言]，請對文章做200字以內總結\n並且條列式列出你覺得跟這篇文章內容有高度相關的留言(最多100則留言)。回復格式為\n(文章):\n(留言):\n[文章]\n"""+str(contents[i])+"\n[留言]\n"+str(comments[i])
            # old prompt
            prompt = (
                """現在給你[文章]以及[留言]，請對文章做200字以內總結\n並且條列式列出你覺得跟這篇文章內容有高度相關的留言(最多30則留言)。回復格式為\n(文章):\n(留言):\n[文章]\n"""
                + str(contents[i])
                + "\n[留言]\n"
                + str(comments[i])
            )
            content_and_comment_list.append(self.prompt_input("你是一位在臺灣的資深時事分析專家", prompt))

        # with open("./result/prompt_summary.txt", "r") as file:
        #     content_and_comment_list = file.readlines()
        for content in content_and_comment_list:
            content_and_comment += content

        # Generate report
        # new prompt
#         prompt ="""你是一位時事分析專家，我會給你幾篇(文章)以及(留言)，請綜合分析這些留言對事件的風向看法，以及留言對事件的觀點為何?\n給出一個對事件總結的標題，以及做一個[表格]分析，[表格]以markdown language呈現，
#         (1)列出事件的觀點\n
#         (2)對此觀點的詳細描述或是依據\n
#         (3)留言對此觀點的看法(每個觀點最多10則留言)\n""" + content_and_comment
        # old prompt
        prompt = ("""你是一位時事分析專家，我會給你幾篇(文章)以及(留言)，請綜合分析這些留言對事件的風向看法，以及留言對事件的觀點為何?\n給出一個對事件總結的標題，以及做一個[表格]分析，
        [表格]以markdown language呈現，[表格]需列出{必要欄位}如下\n
        {欄位1}事件的觀點
        {欄位2}對此觀點的詳細描述或是依據
        {欄位3}留言對此觀點的看法(每個觀點最多10則留言)\n
        {欄位4}留言對觀點的情緒分析
        表格範例:\n
        | {事件觀點} | {詳細描述} | {留言} | {情緒} |
        |----|----|----|----|
        |....|....|....|....|
        |....|....|....|....|
        |....|....|....|....|
        |....|....|....|....|
        |....|....|....|....|
        表格結束\n
        做完表格後，重新以不同角度分析做一份表格，[新表格]目的是對整張表格思考出一個特定分析主題(舉例說明:政傾向、信任度...等)。
        [新表格]格式為，複製原始表格內容，然後改變{欄位4}的標籤，標籤為分析得到的主題(舉例說明:政黨傾向、信任度...等)，
        以及改變每一列的{欄位4}:以主題分析得到的結論(舉例說明:國民黨、民進黨、藍、綠...等)\n
        新表格範例:\n
        標題:表格與分析角度
        | {事件觀點} | {詳細描述} | {留言} | {欄位4} |
        |----|----|----|----|
        |....|....|....|....|
        |....|....|....|....|
        |....|....|....|....|
        |....|....|....|....|
        |....|....|....|....|
        表格結束\n""" + content_and_comment
        )
        res = self.prompt_report("你是一位在台灣的資深時事分析專家", prompt)
        
        print(f"------------------- report generate done.")
        result.result = res

        result_collection.insert_one(
            result.dict(by_alias=True, exclude=["id"])
        )
        print(f"------------------- save {result.session_id} to db done.")

    # def check_response(self, uuid: str):
    #     print(f"Analyzer.response_db is {Analyzer.response_db}")
    #     if uuid == "":
    #         return "Empty UUID."
    #     # Get report and detect dictionary should be clear from ram
    #     elif uuid in Analyzer.response_db:
    #         report = Analyzer.response_db[uuid]
    #         if len(Analyzer.response_db) > 100:
    #             Analyzer.response_db.clear()
    #         return report
    #     # Used to return 404
    #     else:
    #         return 'not ready'
