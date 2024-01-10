#
# GPTハッカソン24耐参加作品
#
# 長いコンテキスト長に対応したGPT-4-1106-previewを使って物語の自動生成を行います

import os
os.environ["OPENAI_API_KEY"] = "<APIキー>"
import openai
from openai import OpenAI
openai.apikey=os.environ["OPENAI_API_KEY"]
def gpt(utterance):
    #response = openai.chat(
    print("call gpt")
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        #model="gpt-3.5-turbo-1106",
        #model="gpt-3.5-turbo",
        #response_format={"type":"json_object"},
        messages=[
            #{"role": "system", "content": "あなたは役に立つアシスタントで、全てJSON形式で出力します。"},
            {"role": "system", "content": "あなたは有能な脚本家で、感動的な脚本を書きます"},
            {"role": "user", "content": utterance},
        ]
    )
    return response.choices[0].message.content

client = OpenAI()
# テキストからの画像生成の実行

def generate_image(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return(response.data[0].url)

##print(generate_image("cute cat eared maid character in a cafe"))
import gradio as gr
import time
def generate(text):
    str=""
    prompt=f"物語のテーマ:{text}\nあらすじ:"
    str+=f"{prompt}\n\n".replace("\n","<br>")
    yield str
    synopsis=gpt(prompt)
    str+=f"{prompt}\n\n{synopsis}".replace("\n","<br>")
    context=f"{prompt}\nあらすじ:{synopsis}\n"
    yield str
    for i in range(10):
        prompt=f"{context}シーン{i+1}:\n"
        scene=gpt(f"{prompt}を書け")
        print(scene)
        scene=scene[:scene.find(f"シーン{i+2}")]
        print(scene)
        context=f"{prompt}{scene}\n"
        imgprompt=gpt(f"以下の場面を表現するDALL-E用プロンプトを書け\n{scene}")
        while True:
            img=None
            try:
                print("try to make image")
                img=generate_image(imgprompt)
            except:
                pass
            if img is not None:
                break

        print(img)
        str+="<table>"
        str+="<tr>"
        str+="<td>"
        str+=f"<img width=256 src={img}>"
        str+="</td>"
        str+="<td>"
        str+=scene.replace("\n","<br>")
        str+="</td>"
        str+="</tr>"
        str+="</table>"
        yield  str
        time.sleep(3)
    yield str


# Blocksの作成
with gr.Blocks() as demo:
    # コンポーネント
    name = gr.Textbox(label="Name",value="UberEats配達員が主人公の冒険活劇")
    greet_btn = gr.Button("Make Story")
    result = gr.HTML("result")

    # イベントリスナー
    greet_btn.click(fn=generate, inputs=name, outputs=result, api_name="generate")


# 起動
demo.launch()
