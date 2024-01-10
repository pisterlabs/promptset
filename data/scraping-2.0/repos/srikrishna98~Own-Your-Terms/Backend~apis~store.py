from flask_restx import Resource
from flask import request, render_template, Response
import openai
import os
import json
from llama_index import GPTSimpleVectorIndex
from llama_index import Document
from furl import furl

os.environ["OPENAI_API_KEY"] = "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"
openai.organization = "org-Ddi6ZSgWKe8kPZlpwd6M6WVe"
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_domain(link):
    print("link", link)
    f = furl(link)
    host = f.host
    tld = host.split(".")
    if len(tld) > 2:
        return tld[1]
    else:
        return tld[0]


def get_title(title):
    print(title)
    title_copy = f"{title}"
    f = furl(title)
    host = f.host
    if host is not None:
        return host
    else:
        return title_copy


class Store(Resource):
    def post(self):
        data = json.loads(request.get_data())
        linkID = get_domain(data['link'])
        pageTitle = get_title(data['title'])
        userid = data['userid']
        data = data['TOS']
        print(linkID, pageTitle)
        unique_doc = linkID + " " + pageTitle
        file_name = str(hash(userid + unique_doc)) + ".txt"
        # dict_obj = {"userid":userid,"pageTitle":pageTitle}
        alreadyPresentList = []
        userDataJson = {}
        if os.path.exists("./userData.json"):
            with open('./userData.json', 'r') as userDataJsonFile:
                userDataJson = json.loads(userDataJsonFile.read())
                if userid in userDataJson:
                    alreadyPresentList = userDataJson[userid]
                    if unique_doc not in alreadyPresentList:
                        alreadyPresentList.append(unique_doc)
        else:
            alreadyPresentList.append(unique_doc)
        userDataJson[userid] = alreadyPresentList
        print("New data : ", str(userDataJson))
        userDataJsonFileWrite = open('./userData.json', "w")
        userDataJsonFileWrite.write(json.dumps(userDataJson))
        userDataJsonFileWrite.close()
        with open(str(file_name), 'w') as fl:
            fl.write(data)
        llama_doc = Document(data, doc_id=userid + "<sep>" + unique_doc)
        index = GPTSimpleVectorIndex.from_documents(documents=[llama_doc])
        index.update(llama_doc)
        question = "Highlight atleast 5 red flags in this terms and conditions."
        res = index.query(question)
        if os.path.exists("database.json"):
            existing_index = GPTSimpleVectorIndex.load_from_disk(
                'database.json')
            existing_index.update(llama_doc)
            existing_index.save_to_disk("database.json")
        else:
            index.save_to_disk("database.json")
        response = {
            "sentences": str(res).strip().split("\n")
        }
        return response, 200
