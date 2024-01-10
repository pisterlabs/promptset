from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(
    openai_api_base="https://aiapi.xing-yun.cn/v1",
    openai_api_key="sk-3e5wTBAl2iFDvQvW9b5693C90a97425eBf3b4bEa558eC66a",
    temperature=0.1,
   
   
    model_name="gpt-3.5-turbo",
    # streaming=True,  # ! important
    # callbacks=[StreamingStdOutCallbackHandler()]  # ! important
)

llm("hi")