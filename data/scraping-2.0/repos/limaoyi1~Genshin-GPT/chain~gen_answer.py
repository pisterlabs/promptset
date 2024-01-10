import uuid

from langchain.chat_models import ChatOpenAI

from readconfig.myconfig import MyConfig
from chain.gpt_memory import GptChain
from matchquery.match import MatchAnswer

config = MyConfig()


# 抽象父类
class Gen:
    config: MyConfig = None
    GptChain: GptChain = None

    def __init__(self, session_id, npc_name):
        self.config = MyConfig()
        print(f"open ai key:{self.config.OPENAI_API_KEY}")
        self.GptChain = GptChain(openai_api_key=self.config.OPENAI_API_KEY, openai_base_url=self.config.OPENAI_BASE_URL,
                                 session_id=session_id,
                                 redis_url=self.config.REDIS_URL, npc_name=npc_name)


# ----------------------------------------------------------------
# 生成标题
class GenAnswerOfRole(Gen):
    role_name: str = None
    material: str = None
    query: str = None
    match_answers: [] = None
    match_query: [] = None
    match_wiki: [] = None

    def __init__(self, session_id, role_name):
        super().__init__(session_id, role_name)
        self.match_db = None
        self.match_query = None
        self.role_name = role_name

    def query_to_role(self, query):
        self.query = query
        self.predict_answer_init()
        self.get_match_answer()
        return self.get_role_answer()

    def predict_answer_init(self):
        # 真的有必要在问一遍LLM吗？
        # text = f"""请你作为{self.role_name}(游戏角色) 简要的回答以下问题:
        # Question: {self.query}
        # """
        # llm1 = ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY)
        # self.material =llm1.predict(text)
        pass

    def get_match_answer(self):
        answer = MatchAnswer(self.role_name)
        # self.match_answers = answer.match(self.material)
        self.match_answers = answer.match(self.query)
        print(self.match_answers)
        # self.match_wiki = answer.matchWiki(self.query)
        # print(self.match_wiki)
        self.match_db = answer.matchTools(self.query)
        print(self.match_db)

    def get_role_answer(self):
        text = ""
        for answer1 in self.match_answers:
            text = text + answer1 + "\n" + "        "
        # wiki = ""
        # for wiki_text in self.match_wiki:
        #     wiki += wiki_text + "\n" + "        "
        db = self.match_db.replace("\n", "\n        ")
        # ----------
        # wiki:
        # {wiki}

        template = f"""this is my (旅行者的) new question :{self.query}

Provide you with possible relevant wiki text from the vector database:
====
        db:
        {db}
====

Provide you with possible relevant words that {self.role_name} has said from the vector database:
====
        {text}
====
You will use the tone of {self. role_name} to talk to me.
Imitate {self.role_name}'s linguistic style and sentence structures.Keep the conversation simple.
question :{self.query}
{self.role_name}:"""
        return self.GptChain.predict(template)


if __name__ == '__main__':
    # session_id = 0011225

    session_id = "1234578913161"
    print(session_id)
    query = "如何看待剑术"
    role = "神里绫华"
    title = GenAnswerOfRole(session_id, role)
    answer = title.query_to_role(query)
    print("\n\n\n")
    print("======================")
    print("你\t:" + query)
    print(role + "\t:" + answer)
    print("======================")

# 将一个问题拆分成多个子问题解决,可以大大提高AI对问题的理解,从而提高程序的速度和准确性
