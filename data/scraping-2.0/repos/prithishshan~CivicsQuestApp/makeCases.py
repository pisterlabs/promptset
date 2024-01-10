from PyPDF2 import PdfReader 
from openai import OpenAI
import requests
import pandas as pd
import io
import json

openai = OpenAI(organization="org-GeYYLqj76eyVILZQMYl1nc3x", api_key="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",)

cases = pd.read_csv('./OverruledDecisions.csv')
oldCases = cases.loc[cases['Year of Overruling Decision'] < 2010]
# print(oldCases)
#reverses the order of the dataframe
oldCases = oldCases[::-1]

caseFileRead = open(f"./cases.json", "r")
cases = json.load(caseFileRead)
caseFileRead.close()
caseFileWrite = open(f"./cases.json", "w")

for index, case in oldCases.iterrows():
    print("Keep Generating: y/n")
    if input().lower() == "n":
        break
    caseTitle = case["Overruling Decision"].split(",")[0]
    caseTitle = caseTitle.replace(" ", "-")
    print(caseTitle)

    if any(case['title'] == caseTitle for case in cases):
        print("Case already in cases.json")
        continue

    resCount = 0
    pdfNotFound = True
    while pdfNotFound:
        res = requests.get(f"https://www.loc.gov/search/?q={caseTitle}&fo=json&at=results.{resCount}").json()
        if "resources" in res[f"results.{resCount}"]:
            caseResources = res[f"results.{resCount}"]["resources"]
            # print(caseResources, resCount)
            for resource in caseResources:
                if "pdf" in resource:
                    caseLink = resource["pdf"]
                    casePageLink = resource["url"]
                    pdfNotFound = False
                    break
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
    if len(reader.pages) < 10:
        for page in reader.pages:
            casePdfText += page.extract_text()
            print(page.extract_text())
    else:
        print("PDF too long")
        continue

    print("Generate Case: y/n")
    if input().lower() == "y":
        messages = [{"role":"system", "name":"instructor", "content":"Your job is to take the information you are given about an overruled supreme court decision, and generate a short realistic scenario that involves the case you are provided with. The generated case will be used as content for a game, where players will attempt to remedy it with works of legislation and political work that their characters were involved with at some point. Your response must be very concise."}, {"role":"system", "name":"pdfcontents", "content":casePdfText}]
        completion = openai.chat.completions.create(messages=messages, model="gpt-4-1106-preview", max_tokens=200, n=1, temperature=0.9)
        print(completion.choices[0].message.content)
        case = {
            "id": index,
            "name": case["Overruling Decision"],
            "year": case["Year of Overruling Decision"],
            "title": caseTitle,
            "pdfLink": caseLink,
            "pageLink": casePageLink,
            "scenario": completion.choices[0].message.content,
            "isLaw": "false",
        }
        cases.append(case)
        jsonCases = json.dumps(cases, indent=2)
        caseFileWrite.write(jsonCases)
    else:
        continue


for i, case in enumerate(cases):
    case["id"] = i

jsonCases = json.dumps(cases, indent=2)
caseFileWrite.write(jsonCases)