# Commented out IPython magic to ensure Python compatibility.
# always needed
import os
import faiss
#from torch.utils.tensorboard import SummaryWriter
# log and save
from pathlib import Path
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.agents import initialize_agent

import nltk
nltk.download('punkt')
from langchain.vectorstores import FAISS
from typing import List, Union
import re
from langchain.agents import Tool, AgentOutputParser
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory ,ReadOnlySharedMemory

from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from pathlib import Path

def initialize():
    
    os.environ["OPENAI_API_TYPE"] = "OPENAI_API_TYPE"
    os.environ["OPENAI_API_VERSION"] = "OPENAI_API_VERSION"
    os.environ["OPENAI_API_BASE"] = "OPENAI_API_BASE"
    os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

    global llmAzure 
    
    llmAzure = AzureChatOpenAI(
        openai_api_version="openai_api_version",
        deployment_name="deployment_name",
        model_name="model_name"
    )
    #OpenAI類默認對應 「text-davinci-003」版本：
    #ChatOpenAI類默認是 "gpt-3.5-turbo"版本
    #OpenAI是即將被棄用的方法，最好是用ChatOpenAI
    # 檢查是否有可用的GPU
    #EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    #llm_chat = ChatOpenAI(temperature=0) #GPT-3.5-turbo
    embeddings = OpenAIEmbeddings(deployment_name='deployment_name', chunk_size=1)

    # 獲取當前 Python 腳本的絕對路徑
    current_script_path = Path(__file__).resolve().parent

    # 在當前目錄下檢查 "faiss_index" 資料夾是否存在
    faiss_index_path = current_script_path / 'faiss_index'
    faiss_index_path.mkdir(parents=True, exist_ok=True)

    # 檢查 "index.faiss" 文件是否存在
    index_path = faiss_index_path / 'index.faiss'
    if not index_path.exists():
        texts = ['這是一個測試文本', '這是另一個測試文本']
        db = FAISS.from_texts(texts, embeddings)
        db.save_local(folder_path=str(faiss_index_path))
    # 讀取已經創建的向量數據庫
    index = faiss.read_index(str(index_path))
    # 獲取向量數據庫中的向量數量
    num_vectors = index.ntotal
    print("向量數據庫中的向量數量：", num_vectors)
    return embeddings
    
"""## 模板 （Agent, tool, chain)

## 定義Tools的集合
"""



def get_my_agent_1():

    def voice_message(query: str):
        global llmAzure 
        embeddings = OpenAIEmbeddings(deployment_name='deployment_name', chunk_size=1)

        # 獲取當前 Python 腳本的絕對路徑
        current_script_path = Path(__file__).resolve().parent
        # 在當前目錄下找 "faiss_index" 資料夾
        faiss_index_path = current_script_path / 'faiss_index'

        prompt_template = """你現在是一個文本過濾器，你的工作不是回答問題，首先，請先抓出問句內是否有錯字，如果有，請幫我修正為正確的字，並確保語句通順，再
        根據提供的文本幫我把問題裡出現的所有同義詞(或與其相似的詞)替換成與其相似的所有集合詞，並輸出替換後的問題(保留原本問題的架構及意思)
        ============
        問題:{question}
        文本:
        {context}
        =============
        只需要回傳問題本身
        """
        question_prompt = PromptTemplate(
                template=prompt_template, input_variables=["question","context"]
        )
        docsearch0 = FAISS.load_local(str(faiss_index_path), embeddings, index_name="index_Synonym")
        qa = RetrievalQA.from_chain_type(ChatOpenAI(engine="gpt-4-32k"), chain_type="stuff", retriever=docsearch0.as_retriever(),
                                        chain_type_kwargs = {"verbose": True,
                                                            "prompt": question_prompt}) 
        result = qa.run(query)


        fubon_question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
        Return any relevant text. 
        {context}
        Question: {question}
        Relevant text, if any:
        """
        FUBON_QUESTION_PROMPT = PromptTemplate(
            template=fubon_question_prompt_template, input_variables=["context", "question"]
        )


        fubon_combine_prompt_template = f"""你是一位語音客服人員，在顧客轉接至真人客服人員之前，由你來解決他們的問題，你將負責兩個部分的工作內容。step1：理解客戶的問題，將問題與Ｑ選單配對找出一個最符合的選項，只能需擇選單有的選項，不可自行創造Ｑ選項，step2：直接複製所選的Ｑ對應的Ａ，一字不漏的將內容貼上，注意每一個內容都會出現類似“|有匹配意圖|Z20|無卡提款序號時間限制|N|N”的字串，請忽略這些不要將其複製貼上，你必須回答問題，不可回答我 [非常抱歉，我無法提供您需要的資訊。我建議您直接聯繫我們的客服人員，他們將能夠為您提供更詳細的解答]，依指定格式輸出：
        step1: 
        客戶的問題理解：
        對應的problem選項：
        step2: 
        語音回覆：
        —
        客戶問題：{{question}}
        —
        {{summaries}}
        —
        你是一位語音客服人員，在顧客轉接至真人客服人員之前，由你來解決他們的問題，你將負責兩個部分的工作內容。step1：理解客戶的問題，將問題與Ｑ選單配對找出一個最符合的選項，只能需擇選單有的選項，不可自行創造Ｑ選項，step2：直接複製所選的Ｑ對應的Ａ，一字不漏的將內容貼上，注意每一個內容都會出現類似“|有匹配意圖|Z20|無卡提款序號時間限制|N|N”的字串，請忽略這些不要將其複製貼上，你必須回答問題，不可回答我[非常抱歉，我無法提供您需要的資訊。我建議您直接聯繫我們的客服人員，他們將能夠為您提供更詳細的解答]，依指定格式輸出：
        step1: 
        客戶的問題理解：
        對應的problem選項：
        step2: 
        語音回覆：
        """
        FUBON_COMBINE_PROMPT = PromptTemplate(
            template=fubon_combine_prompt_template, input_variables=["summaries", "question"]
        )
        docsearch = FAISS.load_local(str(faiss_index_path), embeddings, index_name="index_QA")
        similarity = docsearch.similarity_search(query, k=3)

        global similarity_QA
        global similarity_QA_source 
        similarity_QA = [i.page_content for i in similarity]
        similarity_QA_source = [i.metadata for i in similarity]

        data_retriever = RetrievalQA.from_chain_type(ChatOpenAI(engine="gpt-4-32k"), chain_type="map_reduce", memory=readonlymemory,
                                                    retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
                                                    chain_type_kwargs = {"verbose": True,
                                                                        "question_prompt": FUBON_QUESTION_PROMPT,
                                                                        "combine_prompt": FUBON_COMBINE_PROMPT})
        output = data_retriever.run(result)
        output = re.sub(r"\n+","", output)
        print('我是OUTPUT：', output)
        return output


    customize_tools = [
        
        Tool(
            name="Voice_message",
            func=voice_message,
            description="ONLY use when user told you to use Voice_message."
        )
    ]   
   
    memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", 
                                            input_key="input", 
                                            output_key='output', return_messages=True)
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    def _handle_error(error) -> str:
       return str(error)[:150]
    

    llmAzure = AzureChatOpenAI(
        openai_api_version="openai_api_version",
        deployment_name="deployment_name",
        model_name="model_name"
    )

    my_agent = initialize_agent(
        tools=customize_tools,
        llm =llmAzure,
        agent='conversational-react-description',
        verbose=True,
        memory=memory,
        max_iterations=4,
        early_stopping_method='generate',
        handle_parsing_errors=_handle_error
    )
  
    #https://www.youtube.com/watch?v=q-HNphrWsDE
    agent_prompt_prefix = """
    Assistant is a large language model in 富邦銀行. Always answer question with "Traditional Chinese", By default, I use a Persuasive, Descriptive style, but if the user has a preferred tone or role, assistant always adjust accordingly to their preference. If a user has specific requirements, (such as formatting needs, answer in bullet point) they should NEVER be ignored in your responses

    Assistant is designed to be able to assist with a wide range of tasks,It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to questions. 
    Additionally, Assistant is able to generate its own text based on the observation it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and in-depth explanations on a wide range of topics, like programming, summarizing. 

    Unfortunately, assistant is terrible at current affairs and bank topic, no matter how simple, assistant always refers to it's trusty tools for help and NEVER try to answer the question itself.
    Re-emphasizing, you must respond in "Traditional Chinese" and Don't forget to provide 來源路徑 if the observation from the tool includes one.
    TOOLS:
    ------

    Assistant has access to the following tools:
    """


    agent_prompt_format_instructions = """To use a tool, use the following format:

    ```
    Thought: Do I need to use a tool(DO NOT USE THE SAME TOOL that has been used)? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```

    When you gathered all the observation and have final response to say to the Human,
    or you do not need to use a tool, YOU MUST follow the format(the prefix of "Thought: " and "{ai_prefix}: " are must be included):
    ```
    Thought: Do I need to use a tool? No
    {ai_prefix}: [your response]
    ```"""

    agent_prompt_suffix = """Begin! 

    Previous conversation history:
    {chat_history}

    New user question: {input}
    {agent_scratchpad}
    """

    #自己填充的prompt Costco信用卡可以在哪裡繳款，給我詳細資訊
    new_sys_msg = my_agent.agent.create_prompt(
        tools = customize_tools,
        prefix = agent_prompt_prefix,
        format_instructions= agent_prompt_format_instructions,
        suffix = agent_prompt_suffix,
        ai_prefix = "AI",
        human_prefix = "Human"
    ) 
    from langchain.schema import AgentAction, AgentFinish, OutputParserException
    class MyConvoOutputParser(AgentOutputParser):
        ai_prefix: str = "AI"
        def get_format_instructions(self) -> str:
            return agent_prompt_format_instructions

        def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
            if f"{self.ai_prefix}:" in text:
                return AgentFinish(
                    {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
                )
            regex = r"Action: (.*?)[\n]*Action Input: (.*)"
            match = re.search(regex, text)
            if not match:
                raise OutputParserException(f"IIIII Could not parse LLM output: `{text}`")
            action = match.group(1)
            action_input = match.group(2)
            return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)
        @property
        def _type(self) -> str:
            return "conversational"
        
    my_agent.agent.llm_chain.prompt = new_sys_msg
    my_agent.agent.llm_chain.prompt.output_parser = MyConvoOutputParser() #沒有連到 改用預設的
    return my_agent


'''
initialize()
model = get_my_agent()
result = model.run("受理開戶及更換負責人應注意事項 firstly, try DOCsearch_only if there are not engough information, DOCsearch_synonym")
source_text = similarity_QA
source_doc = similarity_QA_source
source = [i + "\n\n" + json.dumps(j) for i, j in zip(source_text, source_doc)]
print(source)
'''
"視障人士開戶需要見證人嗎? firstly, try DOCsearch_only if there are not engough information, DOCsearch_synonym"
"""
my_agent.run('我叫吳花油')

my_agent.run('我的名字是什麼')

my_agent.run('幫我翻譯這句話成英文：您好，請問何時能夠洽談合作')

my_agent.run('''I have a dataframe, the three columns named 'played_duration', title_id, user_id, I want to know which title_id is the most popular. please add played_duration by title_id and return the title and their sum list''')

my_agent.run('''I want to sort them by their sum, the largest at front, return a list of title_id''')

my_agent.run('我昨天弄丟信用卡了，幫我搜尋補發方法')

"""
"""
llm_chat, embeddings = initialize()
process_and_store_documents(['/Users/kevin/Desktop/python_code/測試文件/測試.txt'])
current_script_path = Path(__file__).resolve().parent
faiss_index_path = current_script_path / 'faiss_index'
index_path = faiss_index_path / 'index.faiss'
index = faiss.read_index(str(index_path))
# 獲取向量數據庫中的向量數量
num_vectors = index.ntotal
print("向量數據庫中的向量數量：", num_vectors)
my_agent = get_my_agent()
my_agent.run('我叫吳花油')
"""
''''
llm_chat, embeddings = initialize()
my_agent = get_my_agent()
my_agent.run("""
:閱讀以下文本後，根據我上傳的文檔，判斷該文本製在作設計時，是否觸犯「銀行辦理財富管理及金融商品銷售業務自律規範」，如有不符合法規之處請列出，並列出觸犯的法條是哪一條。 輸出格式如下： 文本所有不符合之處： -->觸犯的法條：（"章節＿：第＿條"）
---
投資警語
奈米投注意事項
1. 信託財產之管理運用並非絕無風險，本行以往之經理績效不保證指定信託運用信託財產之最低收益；本行除盡善良管理人之注意義務外，不負責指定運用信託財產之盈虧，亦不保證最低之收益，委託人簽約前應詳閱「奈米投指定營運範圍或方法單獨管理運用金錢信託投資國內外有價證券信託契約條款」(下稱「奈米投約定條款」)。')
""")
'''