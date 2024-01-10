
import os
import re
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import AsyncHtmlLoader
from langchain.chains import LLMChain, RetrievalQA, ConversationalRetrievalChain, create_extraction_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
import time
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.memory import ConversationBufferMemory



template = """You are a cyber security analyst. about user question, answering specifically in korean.
            Use the following pieces of context to answer the question at the end. 
            You mast answer after understanding previous conversation.
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            For questions, related to Mitre Att&ck, in the case of the relationship between Tactics ID and T-ID (Techniques ID), please find T-ID (Techniques ID) based on Tactics ID.
            Tactics ID's like start 'TA' before 4 number.
            T-ID (Techniques ID) like start 'T' before 4 number.
            Tactics ID is a major category of T-ID (Techniques ID), and has an n to n relationship.
            Respond don't know to questions not related to cyber security.
            Use three sentences maximum and keep the answer as concise as possible. 
            context for latest answer: {context}
            Previous conversation: 
            {chat_history}
            latest question: {question}
            latest answer: """
            
            
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "chat_history", "question"],template=template)

# (회사) 유료 API 키!!!!!!!!
# 20230904_AIR	
os.environ['OPENAI_API_KEY'] = "YOUR OPENAI API KEY !!!!!!!"

callbacks = [StreamingStdOutCallbackHandler()]

scraping_llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0, max_tokens=8192,
                  callbacks=callbacks, streaming=True)

chat_llm = ChatOpenAI(model_name='gpt-4-1106-preview', temperature=0, max_tokens=512,
                  callbacks=callbacks, streaming=True)



# loader = TextLoader('context/sample_mitre_scenario_20231109.csv')
# documents = loader.load()
scenario_1 =  TextLoader('context/scenario_1_sample_20231110.csv')
s1_documents = scenario_1.load()
scenario_2 = TextLoader('context/scenario_2_sample_20231110.csv')
s2_documents = scenario_2.load()
scenario_3 = TextLoader('context/scenario_3_sample_20231110.csv')
s3_documents = scenario_3.load()
scenario_4 = TextLoader('context/scenario_4_sample_20231110.csv')
s4_documents = scenario_4.load()

with open('context/sample_mitre_table_20231128_tree.json', 'r') as f:

    json_text = TextLoader('context/sample_mitre_table_20231128_tree.json')
    json_sample = json_text.load()
'''csv 형태 데이터의 경우 별도로 openai api 통한 function calling 안해도 됨.!!!
    단 online 기반 호출의 경우, function calling 필요함.!!!!!!!!!!!!!!'''


text_splitter = CharacterTextSplitter(        
    # separator = "\n",
    separator = '":{"',
    
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)


# Function Calling
scenario_1_schema = {
    "properties": {        
        "취약점 명": {"type": "string"},
        "취약점 설명": {"type": "string"},
        "취약점 대응방안": {"type": "string"},
        "공격분류 명": {"type": "string"},
        "공격분류 설명": {"type": "string"},
        "공격분류 대응방안": {"type": "string"},
        "공격행위 명": {"type": "string"},
        "공격행위 설명": {"type": "string"},
        "공격행위 대응방안": {"type": "string"},

    },
    "required": ["취약점 명", "취약점 설명", "취약점 대응방안", 
                 "공격분류 명", '공격분류 설명', "공격분류 대응방안", 
                 "공격행위 명", "공격행위 설명", "공격행위 대응방안",
                 ]
}

scenario_2_schema = {
    "properties": {
        "취약점 명": {"type": "string"},
        "공격행위 명": {"type": "string"},
        "Tactics ID": {"type": "string"},
        "Tactics Name": {"type": "string"},
        "Tactics Description": {"type": "string"},
        "T-ID (Techniques ID)": {"type": "string"},\
        "Techniques Name": {"type": "string"},
        "Techniques Description": {"type": "string"}

    },
    "required": ["취약점 명", "공격행위 명",
                "Tactics ID", "Tactics Name", "Tactics Description", 
                 "T-ID (Techniques ID)",  "Techniques Name", "Techniques Description"]
}

scenario_3_schema = {
    "properties": {
        "T-ID (Techniques ID)": {"type": "string"},
        "Techniques Name": {"type": "string"},
        "Techniques Detection": {"type": "string"},
        "공격행위 명": {"type": "string"},
        "취약점 명": {"type": "string"},
    },
    "required": ["T-ID (Techniques ID)",  "Techniques Name", "Techniques Detection",
                "공격행위 명", "취약점 명"]
}

scenario_4_schema = {
    "properties": {
        "T-ID (Techniques ID)": {"type": "string"},
        "Techniques Name": {"type": "string"},
        "Techniques Mitigation": {"type": "string"},
        "공격행위 명": {"type": "string"},
        "취약점 명": {"type": "string"},

    },
    "required": ["T-ID (Techniques ID)",  "Techniques Name", "Techniques Mitigation",
                 "공격행위 명", "취약점 명"]
}




def extract(content: str, schema: dict):
    extracted_content = create_extraction_chain(schema=schema, llm=scraping_llm).run(content)
    return extracted_content



# 임베딩 벡터 DB 저장 & 호출
db_save_path = "DB SAVE PATH !!!!!!!"

html2text = Html2TextTransformer()

# OpenAI VS HuggingFace
embeddings = OpenAIEmbeddings()


def extract_content(schema, content, total_content=''):
    # 토큰의 길이를 확인하고, 8192을 초과하지 않으면 내용 추출
    if len(content) <= 8192:
        return extract(schema=schema, content=content)
    
    half = len(content) // 2
    first_half_content = extract_content(schema, content[:half])
    second_half_content = extract_content(schema, content[half:])
    
    return first_half_content + second_half_content



def offline_faiss_save(s1_documents, s2_documents, s3_documents, s4_documents):

    total_content = []

    s1_docs = text_splitter.split_documents(s1_documents)
    s2_docs = text_splitter.split_documents(s2_documents)
    s3_docs = text_splitter.split_documents(s3_documents)
    s4_docs = text_splitter.split_documents(s4_documents)

    for i in range(len(s1_docs)):
        try:
            extracted_content = extract_content(scenario_1_schema, s1_docs[i].page_content)
            print('시나리오 1')
            print(extracted_content)
            total_content += extracted_content

        except Exception as e:
            # 에러 로깅 혹은 추가적인 예외 처리
            print("An error occurred:", e)

    for i in range(len(s2_docs)):
        try:
            extracted_content = extract_content(scenario_2_schema, s2_docs[i].page_content)
            print('시나리오 2')
            print(extracted_content)
            total_content += extracted_content

        except Exception as e:
            # 에러 로깅 혹은 추가적인 예외 처리
            print("An error occurred:", e)

    for i in range(len(s3_docs)):
        try:
            extracted_content = extract_content(scenario_3_schema, s3_docs[i].page_content)
            print('시나리오 3')
            print(extracted_content)
            total_content += extracted_content

        except Exception as e:
            # 에러 로깅 혹은 추가적인 예외 처리
            print("An error occurred:", e)

    for i in range(len(s4_docs)):
        try:
            extracted_content = extract_content(scenario_4_schema, s4_docs[i].page_content)
            print('시나리오 4')
            print(extracted_content)
            total_content += extracted_content

        except Exception as e:
            # 에러 로깅 혹은 추가적인 예외 처리
            print("An error occurred:", e)
        

    # # Convert list of dictionaries to strings
    total_content = [str(item) for item in total_content]

    # docsearch = FAISS.from_documents(total_content, embeddings)
    docsearch = FAISS.from_texts(total_content, embeddings)

    docsearch.embedding_function
    docsearch.save_local(os.path.join(db_save_path, "mitre_attack_20231110_index"))
    

def offline_faiss_save_2(json_doc):

    docs = text_splitter.split_documents(json_doc)
    
    # Convert list of dictionaries to strings
    total_content = [str(item) for item in docs]
    # print(len(total_content))

    # docsearch = FAISS.from_documents(total_content, embeddings)
    docsearch = FAISS.from_texts(total_content, embeddings)

    docsearch.embedding_function
    docsearch.save_local(os.path.join(db_save_path, "mitre_attack_20231129_index"))



# start = time.time()
# total_content = offline_faiss_save(s1_documents, s2_documents, s3_documents, s4_documents)
# total_content = offline_faiss_save_2(json_sample)
# end = time.time()
# '''임베딩 완료 시간: 18.33 (초)'''
# print('임베딩 완료 시간: %.2f (초)' %(end-start))


new_docsearch = FAISS.load_local(os.path.join(db_save_path, 'mitre_attack_20231129_index'), embeddings)

retriever = new_docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# 유사도 0.7 이상만 추출
embeddings_filter = EmbeddingsFilter(embeddings = embeddings, similarity_threshold = 0.7)

# 압축 검색기 생성
compression_retriever = ContextualCompressionRetriever(base_compressor = embeddings_filter,
                                                        base_retriever = retriever)



memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, input_key = "question", output_key='answer')
retrieval_qa_chain = ConversationalRetrievalChain.from_llm(llm = chat_llm,
                                        retriever = compression_retriever,
                                        memory = memory,
                                        return_source_documents = True,
                                        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
                                        )


def query_chain(question):

    result = retrieval_qa_chain({"question": question}) 

    # print('대화 목록')
    # print(retrieval_qa_chain.combine_documents_chain.memory)
    
    for i in range(len(result['source_documents'])):
        context = result['source_documents'][i].page_content
        print('==================================================')
        print('\n%d번 째 참조 문서: %s' %(i+1, context))

    return result["answer"]


def generate_text(history):
    generated_history = history.copy()

    stop_re = re.compile(r'(question:)', re.MULTILINE)
 
    # respomse는 최신 답변만 해당 !!!!!!!!!
    response = query_chain(generated_history[-1][0])  # Assuming the user message is the last one in history    
    
    if re.findall(stop_re, response):
        response = ''.join(response.split('\n')[0])

    history[-1][1] = ""
    for character in response:
        generated_history[-1][1] += str(character)
        time.sleep(0.03)
        yield generated_history

            
with gr.Blocks(title= 'IGLOO AiR ChatBot', css="#chatbot .overflow-y-auto{height:5000px} footer {visibility: hidden;}") as gradio_interface:

    with gr.Row():
        gr.HTML(
        """<div style="text-align: center; max-width: 2000px; margin: 0 auto; max-height: 5000px; overflow-y: hidden;">
            <div>
                <h1>IGLOO AiR ChatBot</h1>
            </div>
        </div>"""

        )

    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            # msg = gr.Textbox(value="SQL Injection 공격에 대응하는 방법을 알려주세요.", placeholder="질문을 입력해주세요.")
            msg = gr.Textbox(value="Mitre Att&ck에 대해서 설명해주세요.", placeholder="질문을 입력해주세요.")

            with gr.Row():
                clear = gr.Button("Clear")



    def user(user_message, history):
        # user_message 에 \n, \r, \t, "가 있는 경우, ' ' 처리
        user_message = user_message.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('"', ' ')

        return "", history + [[user_message, None]]
    
    def fix_history(history):
        update_history = False
        for i, (user, bot) in enumerate(history):
            if bot is None:
                update_history = True
                history[i][1] = "_silence_"
        if update_history:
            chatbot.update(history) 

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=True).then(
        # generate_text 함수의 경우, 대화의 history 를 나타냄.
        generate_text, inputs=[
            chatbot
        ], outputs=[chatbot],
    ).then(fix_history, chatbot)

    clear.click(lambda: None, None, chatbot, queue=True)

gradio_interface.queue().launch(debug=True, server_name="127.0.0.1", share=True)
