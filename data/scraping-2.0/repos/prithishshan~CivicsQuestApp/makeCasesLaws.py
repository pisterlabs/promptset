from PyPDF2 import PdfReader 
from openai import OpenAI
import requests
import pandas as pd
import io
import json

openai = OpenAI(organization="org-GeYYLqj76eyVILZQMYl1nc3x", api_key="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",)

cases = pd.read_csv('./OverruledLaws.csv')
oldCases = cases.loc[cases['Supreme Court October Term'] < 2010]
oldCases = oldCases.sample(3)
# print(oldCases)
#reverses the order of the dataframe
# oldCases = oldCases[::-1]
casePageLink = ""
caseFileRead = open(f"./cases.json", "r")
cases = json.load(caseFileRead)
caseFileRead.close()


for index, case in oldCases.iterrows():
    # print("Keep Generating: y/n")
    # if input().lower() == "n":
    #     caseFileWrite.close()
    #     break
    caseTitleSpaced = case["Case"].split(",")[0]
    caseTitle = caseTitleSpaced.replace(" ", "-")
    print(caseTitle)

    if any(case['title'] == caseTitleSpaced for case in cases):
        print("Case already in cases.json")
        continue

    resCount = 0
    pdfNotFound = True
    while pdfNotFound:
        res = requests.get(f"https://www.loc.gov/search/?q={caseTitle}&fo=json&at=results.{resCount}").json()
        # print(f"https://www.loc.gov/search/?q={caseTitle}&fo=json")
        if "resources" in res[f"results.{resCount}"]:
            caseResources = res[f"results.{resCount}"]["resources"]
            # print(caseResources, resCount)
            for resource in caseResources:
                if "pdf" in resource:
                    caseLink = resource["pdf"]
                    if casePageLink == "":
                        casePageLink = resource["url"]
                    pdfNotFound = False
                    break
            resCount += 1
        elif "access_restricted" in res[f"results.{resCount}"]:
            if "researchguides" in res[f"results.{resCount}"]["group"]:
                casePageLink = res[f"results.{resCount}"]["url"]
            resCount += 1
        else: break

    if pdfNotFound:
        print("PDF not found")
        continue

    print(caseLink)
    res = requests.get(caseLink)
    pdfFromMem = io.BytesIO(res.content)

    reader = PdfReader(pdfFromMem) 
    casePdfText = ""
    print(len(reader.pages))
    if len(reader.pages) < 51:
        for i, page in enumerate(reader.pages):
            casePdfText += page.extract_text()
            print(page.extract_text())
            # if i > 50:
            #     break
    else:
        print("PDF too long")
        continue

    # print("Generate Case: y/n")
    # if input().lower() == "y":
    messages = [{"role":"system", "name":"instructor", "content":"Your job is to take the information you are given about a law that was deemed unconstitutional by supreme court, and generate a short original scenario that involves the case you are provided with. The generated scenario will be used as content for a game, where players will attempt to select the articles and amendments of the constitution that relate most to the scenario. Your response must be concise and must only include the generated scenario that the player will try and match constitutional elements to. Your response must not include titles, headings, or formatting, and should make no mention of the player, the game, or the original case the scenario is inspired from. Use simple language."}, {"role":"system", "name":"pdf_contents", "content":casePdfText}, {"role":"system", "name":"unconstitutional_description", "content":case["Description of Unconstitutional Provision(s)"]}, {"role":"system", "name":"unconstitutional_description", "content":case["Case"]}]
    completion = openai.chat.completions.create(messages=messages, model="gpt-4-1106-preview", max_tokens=300, n=1, temperature=0.9)
    print(completion.choices[0].message.content)
    case = {
        "id": index,
        "name": case["Case"],
        "year": case["Supreme Court October Term"],
        "description": case["Description of Unconstitutional Provision(s)"],
        "title": caseTitle,
        "pdfLink": caseLink,
        "pageLink": casePageLink,
        "scenario": completion.choices[0].message.content,
        "isLaw": "true",
    }
    cases.append(case)
    jsonCases = json.dumps(cases, indent=2)
    caseFileWrite = open(f"./cases.json", "w")
    caseFileWrite.write(jsonCases)
    caseFileWrite.close()
    # else:
    #     continue


for i, case in enumerate(cases):
    case["id"] = i
    case["isLaw"] = "true"
    case["title"] = case["title"].replace("-", " ")

caseFileWrite = open(f"./cases.json", "w")
jsonCases = json.dumps(cases, indent=2)
caseFileWrite.write(jsonCases)