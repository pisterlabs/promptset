# 必要ライブラリのimport
import os
# gradioのインポート
import gradio as gr

# langchain関連のパッケージインポート

# エージェント系のライブラリインポート
from langchain.agents import load_tools
from langchain.agents import initialize_agent

# OpenAI
from langchain.llms import OpenAI

# 会話用のメモリ
from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory # 要約しながら会話をするとき
from langchain.memory import ConversationBufferWindowMemory # ある時点までの過去の会話を記憶しながら会話をするとき


# ChatGPTっぽく使うためにはプロンプトが必要なので
from langchain.agents  import ZeroShotAgent


from langchain.agents  import AgentExecutor


from langchain.chains  import LLMChain

os.environ["OPENAI_API_KEY"]  = "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"
os.environ["GOOGLE_CSE_ID"]   = "2767b8fda4c8f44c6" 
os.environ["GOOGLE_API_KEY"]  = "AIzaSyClYf2bVMVX5kVJIMNByssyw4Hz0DC_jFI"

llm = OpenAI(temperature=0.5)

# 利用するツールの定義
# ※llm-mathは必須だから加えておく

tools = load_tools(["google-search", "llm-math"],
                   llm = llm)

# プロンプトの生成&定義
prefix = " Have a conversation with a human, answering the following questions as best you can. You have access to the following tools: "

suffix = """ Begin!
           Lets work this out in a step by step way to be sure we have the right answer.
            {chat_history}
            Question: {input}
            {agent_scratchpad}
         """

prompt = ZeroShotAgent.create_prompt(tools,
                                     prefix = prefix,
                                     suffix = suffix,
                                     input_variables = ["input", "chat_history", "agent_scratchpad"])                   
# LLM Chainの定義

llm_chain = LLMChain(llm    = llm,
                     prompt = prompt)

# エージェントのインスタンス化

agent = ZeroShotAgent(llm_chain = llm_chain,
                      tools     = tools,
                      verbose   = True)

# メモリの定義

memory = ConversationBufferMemory(memory_key="chat_history")

# 定義したエージェントやツール、メモリを使って、エージェントのチェーンを生成

agent_chain = AgentExecutor.from_agent_and_tools(agent   = agent,
                                                 tools   = tools,
                                                 verbose = True,
                                                 memory  = memory
                                                 )

def chat(message, history):
  history  = history or []
  response = agent_chain.run(input = message) # 人間が入力したテキストをmessageとして受け取って、responseを返す
  
  history.append((message, response))
  
  return history, history

# 見た目の設定
chatbot = gr.Chatbot().style(color_map=('green', 'pink'))

demo = gr.Interface(
    chat,
    ['text',  'state'],
    [chatbot, 'state'],
    allow_flagging = 'never',
)

demo.launch()

