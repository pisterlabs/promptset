import sys
sys.path.append(["../"])

from pathlib import Path
from typing import Any, List, Dict
import random
import json

from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.embeddings.openai import OpenAIEmbeddings

from pymilvus import (
    connections,
    Collection,
)

from utils.prompts import *
from utils.chatmodel import ChatModel
from app.exception.custom_exception import CustomException
import re

def parser_result(string):
    matches = re.search(r'\{([^}]*)\}', string)

    return matches.group()

class MedicineAgent:
    def __init__(self,
        top_p: float = 1,
        max_tokens: int = 512,
        temperature: float = 0,
        n_retry: int = 2,
        request_timeout: int = 30, **kwargs) -> None:
        
        connections.connect("default", host="localhost", port="19530")
        self.chatmodel = ChatModel(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n_retry=n_retry,
            request_timeout=request_timeout,
            **kwargs
        )
        self.illness_collect = Collection("illness")
        self.drug_collect = Collection("drug")
        self.usage_collect = Collection("drug_usage")
        for collect in [self.illness_collect, self.drug_collect, self.usage_collect]:
            collect.load()
        
    def generate(self, message, prompt=None):
        """
        Chat model generate function
        Args:
            - message (str): human query/message
            - prompt (str): optional system message
        Return:
            - str: Generated output
        """
        # print(50*"=", prompt)
        # print(50*"=", message)
        try:
            messages = []
            if prompt:
                messages.append(SystemMessage(content=prompt))
            messages.append(HumanMessage(content=message))
            generate = self.chatmodel.generate(messages)
        
            return generate
        except Exception as exc:
            raise CustomException(exc)
        
    def embed(self, message):
        """
        Embedding string input
        Args:
            - message (str): message to embed
        Return:
            - List: List of embedding output
        """
        try:
            assert type(message) == str
            embed = self.chatmodel.embed(message=message)
            return embed
        except Exception as exc:
            raise CustomException(exc)
        
    def _search(self, 
        query: str, 
        collection: Collection, 
        output_fields: List, 
        anns_field: str = "embeddings",
        top_k=5, offset=5, nprobe=1) -> str:
        
        query = self.embed(query)
        search_params = {
            "metric_type": "COSINE", 
            "offset": offset, 
            "ignore_growing": False, 
            "params": {"nprobe": nprobe}
        }
        
        results = collection.search(
            data=[query],
            anns_field=anns_field,
            param=search_params,
            limit=top_k,
            output_fields=output_fields,
            expr=None,
            consistency_level="Strong"
        )
        doc = ""
        for result in results[0]:
            for field in output_fields:
                if field == "url":
                    doc += "Nguồn: "
                doc += result.entity.get(field) + "\n"
            doc += "\n"

        return doc
    
    def _query(self, field, value, collection, output_fields: List, top_k=10):
        results = collection.query(
            expr=f"{field} == '{value}'",
            limit=top_k,
            output_fields=output_fields,
        )
        doc = ""
        for result in results:
            for field in output_fields:
                if field == "url":
                    doc += "Nguồn: "
                doc += result.entity.get(field) + "\n"
            doc += "\n"
        return doc

    async def diagnose(self, symptoms: str):
        # 1. query disease from diagnose_input
        # 2. get disease info + user info -> openai generate to concluse disease
        # 3. parse the result
        diagnose_input = symptoms.symptom
        user_info = symptoms.user_info
        
        doc = self._search(
            diagnose_input,
            collection=self.illness_collect,
            output_fields=["title", "diagnosis", "summary", "url"],
            top_k=5,
        )
        
        message = DIAGNOSE_TEMPLATE.format(disease_info=doc, symptom=diagnose_input)
        response = self.generate(message, DIAGNOSE_PROMPT)
        
        return json.loads(response)


    async def check_medicine(self, medicine, disease):
        # 1. query disease's treatment (drugs)
        # 2. query medicine usage
        # 3. send to openai to compare 1. and 2.
        
        # disease_doc = self._search(
        #     disease.name,
        #     collection=self.illness_collect,
        #     output_fields=["title", "treatment", "overview"],
        #     top_k=1,
        # )
        
        # medicine_doc = self._search(
        #     medicine.name,
        #     collection=self.drug_collect,
        #     output_fields=["name", "uses", "warning",],
        #     top_k=2,
        # )
        
        disease_doc = self._query(
            field="title",
            value=disease.name,
            collection=self.illness_collect,
            output_fields=["title", "treatment", "overview"],
            top_k=1
        )
        
        medicine_doc = self._query(
            field="name",
            value=medicine.name,
            collection=self.drug_collect,
            output_fields=["name", "uses", "warning"],
            top_k=1
        )
        
        usage_doc = self._search(
            medicine.name,
            collection=self.usage_collect,
            output_fields=["title", "description"],
            top_k=3,
        )
        
        message = CHECK_MEDICINE_TEMPLATE.format(
            disease_doc=disease_doc,
            drug_doc=medicine_doc + "\n" + usage_doc,
            drug=medicine.name,
            disease=disease.name
        )
        
        response = self.generate(message, CHECK_MEDICINE_PROMPT)
        return json.loads(response)

    
    async def suggest_medicine(self, disease, listed = None):
        # 1. query disease's treatment (drugs)
        # 2. send to openai to concluse
        disease_doc = self._search(
            disease.name,
            collection=self.illness_collect,
            output_fields=["title", "treatment"],
            top_k=3,
        )
        
        # if listed:
        #     exclude = " ".join(listed)
            
        message = SUGGEST_MEDICINE_TEMPLATE.format(disease_info=disease_doc, disease=disease.name)
        
        # result = random.choice(["Paracetamol", "Quinine"])
        # explain = random.choice(["Paracetamol phù hợp để điều trị bệnh, xét với thể trạng bệnh nhân và triệu chứng đang gặp phải.", "Nước tiểu chuột không phù hợp với bệnh nhân. Đây là một chất có hại và không nên sử dụng.", "Một lời cầu nguyện cần xem xét thêm. Vì mặc dù không có vấn đề gì, nhưng bác sĩ nên xem xét lại thuốc này."])
        response = self.generate(message, SUGGEST_MEDICINE_PROMPT)
        response = json.loads(response)
        
        return response
        
        # drug_name = response["suggestion"]
        
        # drug_name = self._search(
        #     drug_name,
        #     collection=self.drug_collect,
        #     output_fields=["name"],
        #     top_k=1,
        # ).strip()
        
        # return dict(suggestion=disease_doc, explain=response["explain"])
    
    
    async def compatible_calculator(self, medicines: List):
        # send multi thread
        # 1. retrieve top_k drugs usage -> causion, bla bla
        # 2. send to openai to compare
        args = []
        for i in range(len(medicines) - 1):
            for j in range(i + 1, len(medicines)):
                args.append(dict(drug1=medicines[i].name, drug2=medicines[j].name))
        
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self._check_two_drugs, **_args) for _args in args]
            for future in as_completed(futures):
                if future.result():
                    results.append(future.result())
        
        return results 

    def _check_two_drugs(self, drug1, drug2):
        print(drug1, drug2)
        drug1_doc = self._search(
            drug1,
            collection=self.drug_collect,
            output_fields=["uses", "caution", "warning"],
            top_k=1,
        )
        drug2_doc = self._search(
            drug2,
            collection=self.drug_collect,
            output_fields=["uses", "caution", "warning"],
            top_k=1,
        )
        
        message = COMPATIBLE_TEMPLATE.format(
            drug_info=drug1_doc+"\n"+drug2_doc,
            drug1=drug1, drug2=drug2
        )
        try:
            response = self.generate(message, COMPATIBLE_PROMPT)
            response = json.loads(response)
            response["source"] = drug1
            response["target"] = drug2
            
        except Exception:
            return

        return response
