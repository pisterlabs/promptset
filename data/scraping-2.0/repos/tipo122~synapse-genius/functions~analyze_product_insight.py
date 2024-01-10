import openai
import logging
import json
import os
from firebase_functions import firestore_fn, https_fn
import requests
from bs4 import BeautifulSoup
from firebase_admin import initialize_app, firestore
import google.cloud.firestore
import asyncio
from pyppeteer import launch
from pyppeteer.launcher import Launcher
from subprocess import check_output

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def main(req:https_fn) -> https_fn.Response:
    # print(' '.join(Launcher().cmd))
    # return json.dumps({"data": "target_url is not in req.data"})
    try:
        out = check_output('apt-get -y install libx11-xcb1 libxcb1 libxcursor1 libxss1 libxtst6 libgtk-3-0 libgdk-pixbuf2.0-0')
        print(out)
        out = check_output("ldd /root/.local/share/pyppeteer/local-chromium/588429/chrome-linux/chrome  | grep 'not found'", shell = True)
        print(out)
    except Exception as err:
        print(err)

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    params = req.get_json()["data"]
    target_url = params["target_url"] if "target_url" in params else None
    canvas_id = params["canvas_id"] if "canvas_id" in params else None
    if target_url is None:
        return json.dumps({"data": "target_url is not in req.data"})

    firestore_client: google.cloud.firestore.Client = firestore.client()
    # return json.dumps({"data": "end"})
    item_name = ""
    item_category = ""
    item_description = ""

    result = {
        "item_property" : {
            "item_name" : item_name,
            "item_category" : item_category,
            "item_description" : item_description,
        },
        "copy_data"  :  [
          {"text": "Everybody Hurts"},
          {"text": "Nothing Compares 2 U"},
          {"text": "Tears in Heaven"},
          {"text": "Hurt"},
          {"text": "Yesterday"}
        ]
    }
 
    text = asyncio.run(fetch_webpage_text(target_url))
    count = 5
    example_json = """
    """
    messages=[
      {"role": "system", "content": """
       あなたは優秀なコピーライターですなです。
       あなたはユーザーから、商品販売ページのコンテンツをテキストとして受け取ります
       その情報から、商品の名前をitem_nameとして、
       商品が属すると推測されるカテゴリーを item_category、
       商品のデスクリプションを item_description として、
       その情報からインスタグラムの広告で使えるようなキャッチーなコピーを
       copy_dataの中に配列として返してください。
       item_descriptionは、100文字以内にしてください。
       """},
      {"role": "assistant", "content": json.dumps(result)}, 
      {"role": "user", "content": f"""
       以下の情報を参考にして、
       コピーを {count}個作成してください。

       参考情報:{text[0:1000]}

       """}
    ]

    response = openai.ChatCompletion.create(
        messages=messages,
        model="gpt-3.5-turbo",
        max_tokens=1000
    )
    
    print(response)

    try:
        result = json.loads(response["choices"][0]["message"]["content"])
        # result = json.loads(response)["choices"][0]["message"]["content"]

        item_name = result["item_property"]["item_name"]
        item_category = result["item_property"]["item_category"]
        item_description = result["item_property"]["item_description"]
        copy_data = result["copy_data"]

        result = {
            "item_property" : {
                "item_url" : target_url,
                "item_name" : item_name,
                "item_category" : item_category,
                "item_description" : item_description,
            },
            "copy_data" : copy_data
        }
        print(result)
        doc_ref = firestore_client.collection("canvases").document(canvas_id)
        doc_ref.set(result, merge=True)
    except:
        import traceback
        traceback.print_exc()
        return json.dumps({"data" : "error"})
    return json.dumps({"data" : "ok"})

async def fetch_webpage_text(url):
    # browser = await launch(
    #     headless=True,
    #     handleSIGINT=False,
    #     handleSIGTERM=False,
    #     handleSIGHUP=False,
    #     logLevel=0,
    #     args=[
    #         '--no-sandbox',
    #         '--single-process',
    #         '--disable-dev-shm-usage',
    #         '--disable-gpu',
    #         '--no-zygote'
    #     ],
    # )
    # page = await browser.newPage()
    # await page.goto(url)
    # source = await page.content()
    # await browser.close()

    headers = {
        'User-Agent': 'synapse genius client 0.01',
        'From': 'akih@keroling.net' 
    }
    soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html.parser')
    # client = ScrapingBeeClient(api_key=sb_api_key)
    # response = client.get(url)
    # soup = BeautifulSoup(source, 'html.parser')
    text_parts = soup.stripped_strings
    text = " ".join(text_parts)
    
    return text

