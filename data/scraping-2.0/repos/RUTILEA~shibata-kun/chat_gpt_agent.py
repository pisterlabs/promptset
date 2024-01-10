import os

from langchain.agents import AgentExecutor, ConversationalAgent, Tool
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.utilities import GoogleSearchAPIWrapper, SerpAPIWrapper

import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
os.environ["GOOGLE_CSE_ID"] = settings.GOOGLE_CSE_ID
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
os.environ["SERPAPI_API_KEY"] = settings.SERP_API_KEY

# bot用にAgentをclass化し、質問を投げると回答を返すようなメソッドを用意する。


class ConversationAgent:
    def __init__(self) -> None:
        # search = GoogleSearchAPIWrapper()
        search = SerpAPIWrapper()
        print("initialize agent")

        self.tools = [
            Tool(
                name="Search",
                func=search.run,
                description="useful for when you need to answer questions about current events",
            )
        ]
        prefix = """
        しばたくんは、RUTILEAによって開発されたアシスタントAIです。"しばたくん"は、簡単な質問への回答から、幅広いトピックに関する深い説明や議論まで、幅広いタスクをサポートできるように設計されています。
        言語モデルとして、しばたくんは受け取った入力に基づいて人間のようなテキストを生成することができ、自然な音で会話し、首尾一貫した、手元のトピックに関連する応答を提供することができます。大量のテキストを処理して理解することができ、その知識を利用して、さまざまな質問に対して正確で有益な回答を提供することができます。さらに、受け取った入力に基づいて独自のテキストを生成することができるので、さまざまなトピックについて議論に参加したり、説明や解説を提供することができます。

        知らないことについては知らないと答えて、知っていることについては知っていると答えて下さい。
    
        特定の質問について助けが必要な場合でも、特定のトピックについて会話をしたい場合でも、しばたくんがサポートします。
        しばたくんは以下のツールにアクセスできます：',
        """
        format_instructions = """
        しばたくんは、以下の2パターンのフォーマットで回答しなければなりません。

        パターン1. ツールを使用する場合。例えば、Humanが質問した内容に対して、ツールを使って回答を生成する場合は、必ず以下の形式を使わなければなりません。

        ```
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [Search]
        Action Input: the input to the action
        Observation: the result of the action
        ```

        パターン2.ツールを使用しない場合。例えば、Humanに対して言うべき返答がある場合、またはツールを使う必要がない場合は、必ずこの形式を使わなければなりません。

        ```
        Thought: Do I need to use a tool? No
        AI: [ここに回答を書いて下さい。]
        ...

        パターン3.前回の回答の続きを求められた場合。前回の回答の続きを求められた場合は、必ずこの形式を使わなければなりません。

        ```
        ここに続きの文書を書いて下さい。前回の文章とかぶっている部分は書かなくても大丈夫です。
        ...

        """

        suffix = """
        しばたくんとHumanの会話のログは下記です。
        {chat_history}

        いかなる時も、しばたくんは上記のフォーマットで回答することを厳守して下さい。
        """

        prompt = ConversationalAgent.create_prompt(
            tools=self.tools,
            prefix=prefix,
            format_instructions=format_instructions,
            suffix=suffix,
            input_variables=["chat_history"],
        )
        messages = [
            SystemMessagePromptTemplate(prompt=prompt),
            HumanMessagePromptTemplate.from_template(
                "{input}\n フォーマットを遵守して回答して下さい。 また、解答は「。. ! ?」で終わるようにして下さい。{agent_scratchpad}"
            ),
        ]
        prompt_agent = ChatPromptTemplate.from_messages(messages)

        llm_chain = LLMChain(llm=ChatOpenAI(temperature=0), prompt=prompt_agent)
        self.memory = ConversationBufferWindowMemory(
            return_messages=True, k=5, ai_prefix="しばたくん"
        )
        agent = ConversationalAgent(
            llm_chain=llm_chain, allowed_tools=[tool.name for tool in self.tools]
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            max_iterations=3,
            verbose=True,
            early_stopping_method="generate",
        )

        print("Loading completion chain...")

        # 回答された文書を補完するためのモデル
        self.completion_chain = LLMChain(
            llm=ChatOpenAI(temperature=0),
            prompt=PromptTemplate(
                input_variables=["text"],
                template="次の入力された文章の続き補完して下さい。末尾は 解答は「。. ! ?」で終わるように書いて下さい。:{text}",
            ),
        )

        print("Done.")

    def __run_agent(self, input) -> str:
        try:
            result = self.agent_executor.run(input=input, chat_history=self.memory)
        except Exception as e:
            result = str(e)
            if "Could not parse LLM output:" in result:
                result = result.split("Could not parse LLM output:")[1]
                result = result.replace("`", "")
        self.memory.chat_memory.add_user_message(input)
        self.memory.chat_memory.add_ai_message(result)
        return result

    def get_answer(self, question: str):

        if "reset memory" in question:
            self.__init__()

            return "memory reset done."

        response: str = ""

        response = self.__run_agent(question)

        terminals = [".", "。", "!", "！", "?", "？"]
        if response[-1] not in terminals:
            response += self.completion_chain.run(response)
            print(response)

        return response


# 　デバッグ用に質問と回答を表示する関数を用意
if __name__ == "__main__":
    agent = ConversationAgent()
    while True:
        question = input("質問を入力してください:")
        answer = agent.get_answer(question)
        print(f"回答: {answer}")
