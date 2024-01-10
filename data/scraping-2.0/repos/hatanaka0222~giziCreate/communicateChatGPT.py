from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
import os
import traceback

def communicate(transcript, openAiKey, templateMsg, templateMsg2, chunkSyze):
    #template = """
    #        次の文章から議事録を作成してください。また、日本語で回答してください。　文章 : {text} 
    #       """
    template = templateMsg + """
             文章 : {text} 
           """
    PROMPT = PromptTemplate(
        input_variables=["text"],
        template=template,
    )
    #template2 = """
    #        次の文章を要約してください。また、日本語で回答してください。　文章 : {text} 
    #       """
    template2 = templateMsg2 + """
            文章 : {text} 
           """
    PROMPT2 = PromptTemplate(
        input_variables=["text"],
        template=template2,
    )
    os.environ["OPENAI_API_KEY"] = openAiKey
    
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=1000)

        text_splitter = CharacterTextSplitter(separator = " ", chunk_size=int(chunkSyze))
        state_of_the_union = transcript
        texts = text_splitter.split_text(state_of_the_union)
        docs = [Document(page_content=t) for t in texts]

        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT2, combine_prompt=PROMPT, verbose=True)
        result = chain.run(docs)

        return result

    except:
        print('例外発生')
        print('------------------------------')
        print('# traceback.format_exc()')
        t = traceback.format_exc()
        print(t)
        print('------------------------------')
        if 'openai.error.Timeout:' in t:
            print('タイムアウトエラー')
            return "タイムアウトが発生しました。"
        
        if 'openai.error.InvalidRequestError:' in t:
            print('トークン数エラー')
            return "ChatGPTの扱えるトークン数以上の文章が送られました。設定画面でチャンク数を減らしてください。"
        
        return "予期せぬエラーが発生しました。"