import os
import requests
import json
from openai_helper import get_client
from sql_helpers import get_text_from_id
from document_highlight import return_highlighted_pdf

class backend:
    def __init__(self):
        self.client = get_client()
        self.TOCs = {}
        
    def query_gpt4(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt4",
            messages=[
                {"role": "user", "content": prompt}
                ])
        return response.choices[0].message.content
    
    def get_keyword(self, query):
        return self.query_gpt4(f"What key characteristic the user is looking for in this resume?\n\nUser query: {query}\nKeywords:")

    def add_tag_fields(self, TOC):
        new_TOC = TOC
        new_TOC["tags"] = []

        if "workExperience" in new_TOC and isinstance(new_TOC["workExperience"], list):
            for entry in new_TOC["workExperience"]:
                if isinstance(entry, dict):
                    entry["tags"] = []

        if "education" in new_TOC and isinstance(new_TOC["education"], list):
            for entry in new_TOC["education"]:
                if isinstance(entry, dict):
                    entry["tags"] = []

        return new_TOC

    def get_toc(self, doc_id):
        if doc_id == "mock":
            return self.metadata
        elif doc_id in self.TOCs.keys():
            return self.TOCs[doc_id]
        else:
            text = get_text_from_id(doc_id)
            json_text = self.query_gpt4(f"You are a semantic parser. Use the following resume to populate a Json object \n\n Schema: {self.schema}\n\ndocument: {text}\nJSON:")
            json_result = json.loads(json_text)
            final_TOC = self.add_tag_fields(json_result)
            self.TOCs[doc_id] = final_TOC
            return final_TOC


    def find_string_in_TOC(self, d, target, path=[]):
        for key, value in d.items():
            if isinstance(value, str) and target in value:
                return path + [key]

            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        result = self.find_string_in_TOC(item, target, path + [key, i])
                        if result:
                            return result
                    elif isinstance(item, str) and target in item:
                        return path + [key, i]

            if isinstance(value, dict):
                result = self.find_string_in_TOC(value, target, path + [key])
                if result:
                    return result

        # Return None if no match is found
        return None

    def add_tags(self, TOC, resume_text, keyword):
        path = self.find_string_in_TOC(TOC, resume_text)
        if path is None:
            TOC["tags"].append(keyword)
        elif path[0] == "workExperience":
            work_experience_index = path[1]
            if 0 <= work_experience_index < len(TOC["workExperience"]):
                if keyword not in TOC["workExperience"][work_experience_index]["tags"]:
                    TOC["workExperience"][work_experience_index]["tags"].append(keyword)
        elif path[0] == "education":
            education_index = path[1]
            if 0 <= education_index < len(TOC["education"]):
                if keyword not in TOC["education"][education_index]["tags"]:
                    TOC["education"][education_index]["tags"].append(keyword)
        else:
            if keyword not in TOC["tags"]:
                TOC["tags"].append(keyword)


    def query(self, doc_id, prompt):
        keyword = self.get_keyword(prompt)
        TOC = self.get_toc(doc_id)
        citations = return_highlighted_pdf(doc_id, prompt)
        for citation in citations:
            self.add_tags(TOC, citation, keyword)
        return citations, TOC 

    def inject_query(self, prompt, highlighted_text):
        return self.query_gpt4(f"You are a semantic parser. Rephrase the following query to incorporate the asker's intent given the text the asker has highlighted and refers to. The query is: {prompt}. The text to incorporate into the query is: {highlighted_text}.")



    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Resume",
        "type": "object",
        "properties": {
            "basic_info": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                    "phone": {"type": "string"},
                    "linkedin": {"type": "string"},
                    "github": {"type": "string"},
                    "website": {"type": "string"}
                    },
                "required": ["first_name", "last_name", "email"]
                },
            "summary": {"type": "string"},
            "work_experience": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "position": {"type": "string"},
                        "start_date": {"type": "string", "format": "date"},
                        "end_date": {"type": "string", "format": "date"},
                        "description": {"type": "string"}
                        },
                    "required": ["company", "position", "start_date"]
                    }
                },
            "education": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "institution": {"type": "string"},
                        "degree": {"type": "string"},
                        "start_date": {"type": "string", "format": "date"},
                        "end_date": {"type": "string", "format": "date"},
                        "description": {"type": "string"}
                        },
                    "required": ["institution", "degree", "start_date"]
                    }
                },
            "skills": {
                "type": "array",
                "items": {"type": "string"}
                },
            "languages": {
                "type": "array",
                "items": {"type": "string"}
                },
            "hobbies": {
                "type": "array",
                "items": {"type": "string"}
                },
            "references": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "title": {"type": "string"},
                        "company": {"type": "string"},
                        "contact": {"type": "string"}
                        }
                    }
                }
            }
        }
    
    metadata = {
        "firstName": "",
        "lastName": "",
        "email": "",
        "phone": "",
        "summary": "Accounting professional with twenty years of experience in inventory and manufacturing accounting. Ability to fill in at a moment's notice, quickly mastering new systems, processes and workflows. Take charge attitude, ability to work independently, recommend and implement ideas and process improvements.",
        "workExperience": [
            {
                "company": "Company Name",
                "position": "Accountant",
                "startDate": "04/2011",
                "endDate": "05/2017",
                "description": "Performed general accounting functions, journal entries, reconciliations and accruals. Implemented and oversaw RGA spreadsheet for returns used by customer service, accounting and upper management. Initiated and tracked claim process with carriers for damages. Participated in identifying and executing the company's business process improvement efforts"
                },
            {
                "company": "Company Name",
                "position": "Inventory Control Manager",
                "startDate": "01/2008",
                "endDate": "01/2010",
                "description": "Became an expert user and handled rollout and training of a new ERP system (Syspro). Handled the purchasing and receiving of raw and semi-finished material, tools, supplies. Continuously renegotiated payment terms with suppliers/vendors resulting in improved cash flow"
                },
            {
                "company": "Company Name",
                "position": "Accounting Manager",
                "startDate": "01/1995",
                "endDate": "01/2008",
                "description": "Prepared all relevant documentation and submitted data for auditors during corporate takeover in 2008. Prepared monthly general ledger entries, reconcile G/L accounts to subsidiary journals or worksheets and posted monthly G/L journal. Managed the payroll function which was outsourced to ADP"
            },
            {
                "company": "Company Name",
                "position": "Full Charge Bookkeeper",
                "startDate": "01/1993",
                "endDate": "01/1995",
                "description": ""
            }
            ],
        "education": [
            {
                "school": "Montclair State College",
                "degree": "B.S Business Administration Accounting",
                "fieldOfStudy": "Accounting",
                "startDate": "",
                "endDate": ""
            }
            ],
        "skills": [
            "Microsoft Office Excel",
            "Outlook",
            "Word",
            "SAGE 100",
            "Ramp (WMS software)",
            "Syspro (ERP program)"
            ],
        "languages": [],
        "certifications": []
        }
