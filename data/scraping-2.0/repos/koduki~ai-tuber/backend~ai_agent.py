import time
import os

class AIAgent:
    talks = [
        {"url":"https://gigazine.net/news/20240105-niklaus-wirth-passed-away/?s=09","data":""},
        {"url":"https://ja.wikipedia.org/wiki/Stable_Diffusion","data":""},
        {"url":"https://ja.wikipedia.org/wiki/ChatGPT","data":""},
        {"url":"https://www.4gamer.net/games/338/G033856/20231220044/","data":""},
        {"url":"https://www.moguravr.com/snapdragon-xr2-plus-gen-2-revealed/","data":""},
        {"url":"https://www.moguravr.com/vtuber-contents-2023/","data":""},
        {"url":"https://note.com/shi3zblog/n/nf657d6105bd9","data":""},
        {"url":"https://ymmt.hatenablog.com/entry/2024/01/05/165100","data":""},
        {"url":"https://collabo-cafe.com/events/collabo/frieren-anime2023-add-info-2nd-cours/","data":""},
        {"url":"https://gamewith.jp/fgo/article/show/432003","data":""},   
    ]

    def __init__(self, llm_model) -> None:
        self.chLLM(llm_model)
    
    def chLLM(self, llm_model):
        self.llm_model = llm_model
        print("use model: " + llm_model)

        # 出力フォーマットを定義
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.pydantic_v1 import BaseModel, Field

        class Reply(BaseModel):
            current_emotion: str = Field(description="maxe")
            character_reply: str = Field(description="れん's reply to User")

        parser = JsonOutputParser(pydantic_object=Reply)

        # テンプレートとプロンプトエンジニアリング
        from langchain.prompts import (
            ChatPromptTemplate,
            HumanMessagePromptTemplate,
            SystemMessagePromptTemplate,
            MessagesPlaceholder,
        )

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'prompt_system.txt')
        prompt_system = open(file_path, "r", encoding='utf-8').read()

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(prompt_system),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"), 
        ]).partial(format_instructions=parser.get_format_instructions())

        # モデルの準備
        from langchain.chat_models import ChatOpenAI
        from langchain_google_genai import ChatGoogleGenerativeAI

        if llm_model == 'gpt4':
            llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        elif llm_model == 'gpt3':
            llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
        elif llm_model == 'gemini':
            llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
        else:
            llm = None

        # チェインを作成
        from langchain.chains import LLMChain
        from langchain.memory import ConversationBufferWindowMemory
        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
        self.chain = LLMChain(llm=llm,prompt=prompt,verbose=False,memory=memory)

    #
    # methods
    #
    def _say(self, text):
        import json

        ls = time.perf_counter()
        res = self.chain.invoke({"input": text})
        le = time.perf_counter()
        print("llm response(sec): " + str(le - ls))
        print("res: " + str(res))

        text = str(res['text']).replace("'", "\"")
        data = json.loads(text)
        data["current_emotion"] = data["current_emotion"].split(":")[0]
        print("parsed: " + str(data))

        return data

    def say_short_talk(self):
        import random

        index = random.randrange(len(self.talks))
        if self.talks[index]["data"] == "":
            url = self.talks[index]["url"]
            msg = f"""以下のページを参考にリスナーに「ちょっと小話でもしようかの」と言って、600文字程度の雑談をしてください。その際にキャラクターらしさを含めた内容になるようにしてください。
            {url}
            """
            
            self.talks[index]["data"] = self._say(msg)

        return self.talks[index]["data"]

    def say_chat(self, comment):
        return self._say(comment)
