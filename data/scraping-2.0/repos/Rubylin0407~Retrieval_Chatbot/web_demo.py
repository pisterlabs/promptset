"""
Gradio Demo
"""
import os
import opencc
import openai
import mdtex2html
import gradio as gr
from dotenv import load_dotenv
from data_pipeline.utils import (
    read_and_process_knowledge_to_langchain_docs,
    initial_langchain_embeddings,
    initial_or_read_langchain_database_faiss,
)

load_dotenv()

# Use the absolute path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess

def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        line = line.replace("$", "")
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def messeage_prepare(system_info, prompt_info):
        message = [
            {"role": "system", "content": system_info},
            {"role": "user", "content": prompt_info}
            ]
        return message

def predict(user_input, chatbot, temperature, top_k):

    system_info = "你是聊天機器人 Retrieval Bot, [檢索資料]是由 Ruby Lin 提供的。 參考[檢索資料]使用中文簡潔和專業的回覆顧客的[問題], 如果答案不在公開資料中, 請說 “對不起, 我所擁有的資料中沒有相關資訊, 請您換個問題或將問題描述得更詳細, 讓我能正確完整的回答您”\n\n"
    docs_and_scores_list = vectordb.similarity_search_with_score([user_input], k=top_k)[0]
    knowledge = "\n".join([docs_and_scores[0].page_content for docs_and_scores in docs_and_scores_list])
    prompt_info =  "[檢索資料]\n{}\n\n[問題]\n{}".format(knowledge, user_input)
    chatbot.append((parse_text(user_input), ""))
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messeage_prepare(system_info, prompt_info),
        temperature=temperature,
    )

    chatbot[-1] = (parse_text(user_input), parse_text(response["choices"][0]["message"]["content"]))
                
    yield chatbot, [], None, parse_text(knowledge)

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], [], None, []

# os.environ["OPENAI_API_KEY"] = open("openai_api.txt", "r").readline()
openai.api_key = os.getenv("OPENAI_API_KEY")
model_kwargs = {'device': 'cpu'}
docs = read_and_process_knowledge_to_langchain_docs("data/knowledge.txt", separator = '\n', chunk_size=1028)
embedding_function = initial_langchain_embeddings("moka-ai/m3e-base", model_kwargs, False)
vectordb = initial_or_read_langchain_database_faiss(docs, embedding_function, "vectordb/vectordbPrivate", True) # renew vector database
s2t = opencc.OpenCC('s2t.json')

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Retreival Chatbot</h1>""")

    with gr.Row(scale=6):
        with gr.Column(scale=4):
            chatbot = gr.Chatbot()
        with gr.Column(scale=1):
            showHtml = gr.HTML()

    with gr.Row(scale=2):
        with gr.Column(scale=4):
            with gr.Column(scale=4):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            temperature = gr.Slider(0, 1, value=0.1, step=0.1, label="Temperature", interactive=True)
            top_k = gr.Slider(0, 10, value=5, step=1, label="top_k", interactive=True)

    history = gr.State([])
    past_key_values = gr.State(None)

    submitBtn.click(predict, [user_input, chatbot, temperature, top_k],
                    [chatbot, history, past_key_values, showHtml], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values, showHtml], show_progress=True)

demo.queue().launch(share=False, inbrowser=True)
