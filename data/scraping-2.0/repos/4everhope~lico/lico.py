
import os
import hashlib
import time
import random
import asyncio

from openai import OpenAI
import edge_tts
import pinecone
from quart import Quart, jsonify, request, render_template, send_file
from werkzeug.utils import secure_filename


from langchain.chat_models import ChatOpenAI 
from langchain.callbacks import get_openai_callback 
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv('./env')

OPENAI_API_KEY_1 = os.environ.get("OPENAI_API_KEY")
OPENAI_API_KEY_2 = os.environ.get("CHATGPT_ACCESS_TOKEN")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")
VECTOR_SPACE_NAME = os.environ.get("VECTOR_SPACE_NAME")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT")
API_BASE = os.environ.get("API_BASE")
MODEL_NAME = os.environ.get("MODEL_NAME")

client = OpenAI(api_key=OPENAI_API_KEY_1)
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)



chat_model_1 = ChatOpenAI(
    model_name=MODEL_NAME,
    openai_api_base=API_BASE,
    openai_api_key = OPENAI_API_KEY_2
    )  
chat_model_2 = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    openai_api_base=API_BASE,
    openai_api_key=OPENAI_API_KEY_2
    )
embed_model = "text-embedding-ada-002"


global memory
memory = ConversationBufferWindowMemory(
                memory_key="history",
                input_key="question",
                output_key='answer',
                return_messages=True,
                k=5
)

total_tokens = 0


memory_keywords = ["帮我记住", "帮我记下来", "写入长期记忆"]


chat_history = []




sysprompt_main = SYSTEM_PROMPT
sysprompt_summary = "Summarize the conversation in Chinese, retaining only core information, discarding irrelevant or repetitive content, for easy database retrieval."

prompt_maingpt = ChatPromptTemplate.from_messages([
            ("system", ""),
            ("ai", "history"),
            ("human", "user_input")
        ])

prompt_summary = ChatPromptTemplate.from_messages([
        ("system", sysprompt_summary),
        ("human", "combined_text")
    ])


llm_chain = LLMChain(
    llm=chat_model_1,
    prompt=prompt_maingpt,
    memory=ConversationBufferWindowMemory(k=3,max_token_limit=650),
    verbose=True  
    
)

llm_chain_summary = LLMChain(
    llm=chat_model_2,
    prompt=prompt_summary,
    
    
)



app = Quart(__name__)

def get_current_datetime():
    
    local_time = time.localtime()
    
    datetime_str = time.strftime("%Y-%m-%d %H:%M", local_time)
    return datetime_str



try:
    if VECTOR_SPACE_NAME not in pinecone.list_indexes():
        
        pinecone.create_index(
            name=VECTOR_SPACE_NAME,
            metric='dotproduct',
            dimension=1536  
        )
    index = pinecone.Index(index_name=VECTOR_SPACE_NAME)
    print("LICO!!-向量数据库-连接到Pinecone并成功连接到现有索引！")
except Exception as e:
    print(f"LICO!!-向量数据库-连接Pinecone时出错：{e}")



def generate_unique_id(prefix):
    unique_str = f"{prefix}_{time.time()}_{random.random()}"
    unique_id = hashlib.sha256(unique_str.encode()).hexdigest()[:16]  
    return f"{prefix}_{unique_id}"


def save_embeddings_to_pinecone(combined_embedding,combined_text):

    index = pinecone.Index(index_name=VECTOR_SPACE_NAME)
    
    combined_embedding_data_vector = combined_embedding.data[0].embedding 
    
    session_id=generate_unique_id("chat_id")
    
    
    data_to_upsert = [
        (session_id, combined_embedding_data_vector, {"text": combined_text})
    ]

    
    upsert_result = index.upsert(data_to_upsert)
    
    if upsert_result:
        print("LICO!!-向量数据库-Data successfully uploaded to Pinecone.")
    else:
        print("LICO!!-向量数据库-Failed to upload data to Pinecone.")
    
    
    


def query_and_retrieve(query, embed_model, index, limit=2000):

    index = pinecone.Index(index_name=VECTOR_SPACE_NAME)
    
    embedding_response = client.embeddings.create(input=[query], model=embed_model,encoding_format="float")

    query_vector = embedding_response.data[0].embedding

    
    retrieval_response = index.query(query_vector, top_k=2, include_metadata=True)

    
    contexts = []
    if retrieval_response['matches']:
        contexts = [match['metadata']['text'] for match in retrieval_response['matches'] if 'text' in match['metadata']]

    
    if not contexts:
        return f"human_input:{query}"

    
    prompt_start = "下面的是数据库里的记忆\nContext:\n"
    prompt_end = f"\nhuman_input:{query}\n "
    combined_contexts = "\n".join(contexts)  
    if len(combined_contexts) <= limit:
        return prompt_start + combined_contexts + prompt_end

    
    for i in range(1, len(contexts)):
        if len("\n".join(contexts[:i])) >= limit:  
            return prompt_start + "\n".join(contexts[:i-1]) + prompt_end  
        
    return prompt_start + combined_contexts + prompt_end  


def extract_and_vectorize(vextorinput, embed_model,sysprompt_summary):
    
    human_input_text = vextorinput['human_input']
    ai_message_text = vextorinput['text']

    
    combined_text = "human_input: " + human_input_text + "ai_reply: " + ai_message_text

    prompt_summary = ChatPromptTemplate.from_messages([
        ("system", sysprompt_summary),
        ("human", combined_text)
    ])

    llm_chain_summary.prompt = prompt_summary

    summary_response = llm_chain_summary({"text": combined_text})
    summary_response_text = summary_response['text']
    print("检测到以下关键内容：", summary_response_text)

    
    combined_embedding = client.embeddings.create(input=[summary_response_text],
    model=embed_model)

    return combined_embedding,summary_response_text


def process_user_input(user_input,sysprompt_main):
    
    
    if "你还记得" in user_input:

        
        query_with_contexts = query_and_retrieve(user_input, embed_model, index, limit=2000)
        human_message_template = query_with_contexts
        print("LICO回忆中……")
    else:
        human_message_template = user_input
    
    
    current_datetime = get_current_datetime()
    
    prompt_maingpt= ChatPromptTemplate.from_messages([
                ("system", "Time:"+current_datetime+sysprompt_main),
                ("ai", "chat-history: {history}"),
                ("human", human_message_template)
            ])
    
    
    llm_chain.prompt = prompt_maingpt


    
    global total_tokens  
    with get_openai_callback() as cb:
        
        response = llm_chain({'human_input': user_input})
        
    total_tokens += cb.total_tokens
    
    print(f'Tokens used in this round: {cb.total_tokens}')
    print(f'Total tokens used: {total_tokens}')

    
    return response


async def speak_text(text, filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, filename)
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoyiNeural")
    await communicate.save(full_path)



async def audio_generation(response_text, filename):
    await speak_text(response_text, filename)  
    print(f'Audio file generated: {filename}')  


@app.route('/<filename>')
async def serve_audio(filename):
    filename = secure_filename(filename)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)
    return await send_file(file_path, as_attachment=True)

@app.route('/delete_audio/<filename>', methods=['POST'])
async def delete_audio(filename):
    filename = secure_filename(filename)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)
    try:
        os.remove(file_path)  
        return jsonify({"status": "success", "message": "Audio file deleted successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})



@app.route('/', methods=['GET', 'POST'])
async def chat():
    user_message = ""
    response = ""
    bot_reply = ""
    if request.method == 'POST':
        form = await request.form
        user_message = form.get('user_message')
        enable_voice = form.get('enable_voice') == 'true'  

        
        if user_message.lower() == 'exit':
            bot_reply = "随时在哦,主人！"
            chat_history.append(("You", user_message))
            chat_history.append(("Lico", bot_reply))
            return await render_template('chat.html', chat_history=chat_history)

        
        allresponse = process_user_input(user_message, sysprompt_main)
        response = allresponse['text']

        
        if enable_voice:
            timestamp = int(time.time())
            filename = f"output_{timestamp}.mp3"
            await audio_generation(response, filename)  
        else:
            filename = None  

        
        if any(keyword in user_message for keyword in memory_keywords):
            combined_embedding, summary_response_text = extract_and_vectorize(allresponse, embed_model, sysprompt_summary)
            save_embeddings_to_pinecone(combined_embedding, summary_response_text)

        
        chat_history.append(("You", user_message))
        chat_history.append(("Lico", response))

        
        if filename:
            return jsonify({"user_message": user_message, "bot_reply": response, "audio_file": filename})
        else:
            return jsonify({"user_message": user_message, "bot_reply": response})

    return await render_template('chat.html', chat_history=chat_history)



if __name__ == "__main__":
    app.run(debug=False, host='127.0.0.1', port=8000)
