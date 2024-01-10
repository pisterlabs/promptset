from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
import gradio as gr
import pickle
import os

os.environ["PROJECT_ROOT"] = '/Users/nishitsujiyouhei/Documents/RPA/input_and_study/liny-manual-chatbot'
os.environ["CHROME_DRIVER_PATH"] = '/Users/nishitsujiyouhei/Documents/RPA/input_and_study/liny-manual-chatbot/chromedriver'
os.chdir(os.getenv('PROJECT_ROOT'))

with open("./storage/make/help_dash/vectorstore.pkl", "rb") as f:
    vectorstore:FAISS = pickle.load(f)

retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

template = """
### 命令文
あなたはQ&Aシステムです。以下の制約に従って[question:]に対しての回答をしてください。

### 制約
- [question:]について[reference text:]の内容をもとに回答をしてください。
- 結果を[### output] セクションの <answer here> に出力してください。
- [reference text:]の中から参考にしたものを[### output] セクションの <refarence sources here> に出力してください。

question: {question}

reference text:
'''
{reference_text}
'''
### output
<answer here>
参考:<refarence sources here>
"""


def chat(message, history=[]):
    try:
        llm = ChatOpenAI(model_name= "gpt-3.5-turbo")
        prompt = PromptTemplate(
            input_variables=["question","reference_text"],
            template=template,
        )
        docs = retriever.get_relevant_documents(message)
        reference_text = ""
        print(docs)
        for doc in docs:
            if len(reference_text) > 1000: break
            reference_text += f'text:\n{doc.page_content}\nsource:\n{doc.metadata["source"]}\n\n'
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({"question":message,"reference_text":reference_text})
    except Exception as e:
        response = f"予期しないエラーが発生しました: {e}"

    history.append((message, response))
    return history, history

chatbot = gr.Chatbot()

demo = gr.Interface(
    chat,
    ['text',  'state'],
    [chatbot, 'state'],
    allow_flagging = 'never',
)

demo.launch()