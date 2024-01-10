import argparse
import json
import os
import shutil
from typing import List, Optional
import urllib
import asyncio
import nltk
import pydantic
import uvicorn
from fastapi import Body, FastAPI, File, Form, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing_extensions import Annotated
from starlette.responses import RedirectResponse

from chains.local_doc_qa import LocalDocQA
from configs.model_config import (KB_ROOT_PATH, EMBEDDING_DEVICE,
                                  EMBEDDING_MODEL, NLTK_DATA_PATH,
                                  VECTOR_SEARCH_TOP_K, LLM_HISTORY_LEN, OPEN_CROSS_DOMAIN)
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint
from langchain.prompts import PromptTemplate
# from datetime import datetime
import uuid
import logging
# from jsonformer import Jsonformer

from langchain.output_parsers import PydanticOutputParser

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="HTTP status code")
    msg: str = pydantic.Field("success", description="HTTP status message")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class ListDocsResponse(BaseResponse):
    data: List[str] = pydantic.Field(..., description="List of document names")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }


class ChatMessage(BaseModel):
    question: str = pydantic.Field(..., description="Question text")
    response: str = pydantic.Field(..., description="Response text")
    history: List[List[str]] = pydantic.Field(..., description="History text")
    source_documents: List[str] = pydantic.Field(
        ..., description="List of source documents and their scores"
    )

    class Config:
        schema_extra = {
            "example": {
                "question": "工伤保险如何办理？",
                "response": "根据已知信息，可以总结如下：\n\n1. 参保单位为员工缴纳工伤保险费，以保障员工在发生工伤时能够获得相应的待遇。\n2. 不同地区的工伤保险缴费规定可能有所不同，需要向当地社保部门咨询以了解具体的缴费标准和规定。\n3. 工伤从业人员及其近亲属需要申请工伤认定，确认享受的待遇资格，并按时缴纳工伤保险费。\n4. 工伤保险待遇包括工伤医疗、康复、辅助器具配置费用、伤残待遇、工亡待遇、一次性工亡补助金等。\n5. 工伤保险待遇领取资格认证包括长期待遇领取人员认证和一次性待遇领取人员认证。\n6. 工伤保险基金支付的待遇项目包括工伤医疗待遇、康复待遇、辅助器具配置费用、一次性工亡补助金、丧葬补助金等。",
                "history": [
                    [
                        "工伤保险是什么？",
                        "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                    ]
                ],
                "source_documents": [
                    "出处 [1] 广州市单位从业的特定人员参加工伤保险办事指引.docx：\n\n\t( 一)  从业单位  (组织)  按“自愿参保”原则，  为未建 立劳动关系的特定从业人员单项参加工伤保险 、缴纳工伤保 险费。",
                    "出处 [2] ...",
                    "出处 [3] ...",
                ],
            }
        }



class WorkExperience(BaseModel):
    work_start_time: str = pydantic.Field(..., title="工作开始时间", description="请输入 YYYY-MM-DD 格式的日期")
    work_end_time: str = pydantic.Field(..., title="工作结束时间", description="请输入 YYYY-MM-DD 格式的日期")
    company: str = pydantic.Field(..., title="公司", description="请输入公司全称")
    industry: str = pydantic.Field(..., title="行业", description="请输入所属行业")
    position: str = pydantic.Field(..., title="职位", description="请输入职位名称")
    work_content: str = pydantic.Field(..., title="工作内容", description="请输入工作内容简介")

class EducationExperience(BaseModel):
    study_start_time: str = pydantic.Field(..., title="学习开始时间", description="请输入 YYYY-MM-DD 格式的日期")
    study_end_time: str = pydantic.Field(..., title="学习结束时间", description="请输入 YYYY-MM-DD 格式的日期")
    school: str = pydantic.Field(..., title="学校", description="请输入学校全称")
    degree: str = pydantic.Field(..., title="学历/学位", description="请输入学历或学位名称")
    major: str = pydantic.Field(..., title="专业", description="请输入专业名称")

class ResumeJson(BaseModel):
    name: str = pydantic.Field(..., title="姓名", description="请输入姓名")
    age: int = pydantic.Field(..., title="年龄", description="请输入年龄", ge=16)
    birth: str = pydantic.Field(..., title="出生日期", description="请输入出生日期")
    gender: str = pydantic.Field(..., title="性别", description="请输入性别", regex=r"男|女")
    contact: str = pydantic.Field(..., title="联系方式", description="请输入手机号码", regex=r"^\d{11}$")
    residence: str = pydantic.Field(..., title="现居住地", description="请输入现居住地")
    work_experience: List[WorkExperience] = pydantic.Field(..., title="工作经验", description="请输入工作经验列表")
    education_experience: List[EducationExperience] = pydantic.Field(..., title="教育经历", description="请输入教育经历列表")






class ResumeMessage(BaseModel):
    # question: str = pydantic.Field(..., description="Question text")
    response: ResumeJson = pydantic.Field(..., description="Resume message in json format")
    # history: List[List[str]] = pydantic.Field(..., description="History text")
    # source_documents: List[str] = pydantic.Field(
    #     ..., description="List of source documents and their scores"
    # )

    class Config:
        schema_extra = {
            "example": {
                "姓名": "黎智豪",
                "年龄": "31岁",
                "出生日期":"1986年1月13日",
                "性别": "男",
                "联系方式": "13580442780",
                "现居住地": "广州-荔湾区",
                "工作经验": [
                    {
                        "工作开始时间": "2015年7月",
                        "工作结束时间": "2016年6月",
                        "公司": "广发银行营销中心",
                        "行业": "无",
                        "职位": "电话销售",
                        "工作内容": "财智金销售"
                    },
                    {
                        "工作开始时间": "2012年4月",
                        "工作结束时间": "2013年3月",
                        "公司": "安利(中国)日用品有限公司",
                        "行业": "美容/保健/外资(欧美)",
                        "职位": "全国业务部客服专员/助理",
                        "工作内容": "负责解答营销人员及顾客关于产品或者业务上的疑难问题"
                    },
                    {
                        "工作开始时间": "2011年3月",
                        "工作结束时间": "2012年3月",
                        "公司": "TVB香港电视有限公司广州分公司",
                        "行业": "生活服务|500-1000人|外资(非欧美)",
                        "职位": "账项跟催部催收专员"
                    }
                ],
                "教育经历": [
                    {
                        "学习开始时间": "2005年9月",
                        "学习结束时间": "2008年7月",
                        "学校": "广东技术师范学院",
                        "学历/学位": "大专",
                        "专业": "贸易经济"
                    }
                ]
            }
        }



def get_folder_path(local_doc_id: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "content")


def get_vs_path(local_doc_id: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "vector_store")


def get_file_path(local_doc_id: str, doc_name: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "content", doc_name)


async def upload_file(
        file: UploadFile = File(description="A single binary file"),
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    file_content = await file.read()  # 读取上传文件的内容

    file_path = os.path.join(saved_path, file.filename)
    if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
        file_status = f"文件 {file.filename} 已存在。"
        return BaseResponse(code=200, msg=file_status)

    with open(file_path, "wb") as f:
        f.write(file_content)

    vs_path = get_vs_path(knowledge_base_id)
    vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store([file_path], vs_path)
    if len(loaded_files) > 0:
        file_status = f"文件 {file.filename} 已上传至新的知识库，并已加载知识库，请开始提问。"
        return BaseResponse(code=200, msg=file_status)
    else:
        file_status = "文件上传失败，请重新上传"
        return BaseResponse(code=500, msg=file_status)







async def local_doc_chat(
        knowledge_base_id: str = Body(..., description="Knowledge Base Name", example="kb1"),
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    vs_path = get_vs_path(knowledge_base_id)
    if not os.path.exists(vs_path):
        # return BaseResponse(code=1, msg=f"Knowledge base {knowledge_base_id} not found")
        return ChatMessage(
            question=question,
            response=f"Knowledge base {knowledge_base_id} not found",
            history=history,
            source_documents=[],
        )
    else:
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=question, vs_path=vs_path, chat_history=history, streaming=True
        ):
            pass
        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]

        return ChatMessage(
            question=question,
            response=resp["result"],
            history=history,
            source_documents=source_documents,
        )











json_schema = {
        "姓名": "",
        "年龄": "",
        "性别": "",
        "联系方式":"",
        "现居住地": "",
        "工作经验": [
            {   "工作开始时间": "",
                "工作结束时间": "",
                "公司": "",
                "行业": "",
                "职位": "",
                "工作内容": ""
            },
        ...
        ],
        "教育经历": [
            {   "学习开始时间": "",
                "学习结束时间": "",
                "学校": "",
                "学历/学位": "",
                "专业": ""
            },...
        ]
}





async def ask_resume(
        file: UploadFile = File(description="A single binary file")
):
    # Generate a unique knowledge_base_id based on the current timestamp
    knowledge_base_id = str(uuid.uuid4())

    # Define the fixed question
    # question='''
    # 按照{json_schema}生成json格式简历信息，其中联系方式是11位数字的手机号码

    # '''.format(
    # json_schema=json_schema
    # )   

  

    resumeparser = PydanticOutputParser(pydantic_object=ResumeJson)

    prompt = PromptTemplate(
    template="Extract information from the local resume.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": resumeparser.get_format_instructions()},
    )

    query = "Please extract the name, age, birth, gender, contact, residence, work experience and education experience from the resume."
    prompt_value = prompt.format_prompt(query=query)

    question=query
    # Upload the file,ChatMessage 的 response 改成response=f"文件上传失败，请重新上传"之类的,
    upload_response = await upload_file(file=file, knowledge_base_id=knowledge_base_id)


    if upload_response.code != 200:
        return ChatMessage(question=question, response=upload_response.msg, history=[], source_documents=[])
        

    vs_path = get_vs_path(knowledge_base_id)
    if not os.path.exists(vs_path):
        return BaseResponse(code=1, msg=f"Knowledge base {knowledge_base_id} not found")
        

  
    else:
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=question, vs_path=vs_path, chat_history=[], streaming=True
        ):
            pass
        # return ResumeMessage(
        #     response=resp['result']
        # )
        logging.info("response results:/n")
        
        logging.info(resumeparser.parse(resp['result']))

        return resp['result']

    # Ask the question
    # chat_response = await local_doc_chat(knowledge_base_id=knowledge_base_id, question=question, history=[])
    # return chat_response




def api_start(host, port):
    global app
    global local_doc_qa

    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)

    app = FastAPI()
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.websocket("/local_doc_qa/stream-chat/{knowledge_base_id}")(stream_chat)

    app.get("/", response_model=BaseResponse)(document)

    app.post("/chat", response_model=ChatMessage)(chat)

    app.post("/local_doc_qa/upload_file", response_model=BaseResponse)(upload_file)
    app.post("/local_doc_qa/upload_files", response_model=BaseResponse)(upload_files)
    app.post("/local_doc_qa/local_doc_chat", response_model=ChatMessage)(local_doc_chat)
    app.post("/local_doc_qa/bing_search_chat", response_model=ChatMessage)(bing_search_chat)
    app.get("/local_doc_qa/list_knowledge_base", response_model=ListDocsResponse)(list_kbs)
    app.get("/local_doc_qa/list_files", response_model=ListDocsResponse)(list_docs)
    app.delete("/local_doc_qa/delete_knowledge_base", response_model=BaseResponse)(delete_kb)
    app.delete("/local_doc_qa/delete_file", response_model=BaseResponse)(delete_doc)
    app.post("/local_doc_qa/update_file", response_model=BaseResponse)(update_doc)
    # app.post("/ask_resume", response_model=ResumeMessage)(ask_resume)
    app.post("/ask_resume")(ask_resume)

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(
        llm_model=llm_model_ins,
        embedding_model=EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
        top_k=VECTOR_SEARCH_TOP_K,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--reload")
    parser.add_argument("--timeout-keep-alive",type=int,default=5000)
    # 初始化消息
    args = None
    args = parser.parse_args()
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    api_start(args.host, args.port)





from typing import List, Tuple
from pydantic import ValidationError

def get_knowledge_based_answer(self, query, vs_path, chat_history=[], streaming: bool = STREAMING):
    vector_store = load_vector_store(vs_path, self.embeddings)
    vector_store.chunk_size = self.chunk_size
    vector_store.chunk_conent = self.chunk_conent
    vector_store.score_threshold = self.score_threshold
    related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)
    torch_gc()

    if len(related_docs_with_score) > 0:
        prompt = generate_prompt(related_docs_with_score, query)
    else:
        prompt = query

    answer_result_stream_result = self.llm_model_chain(
        {"prompt": prompt, "history": chat_history, "streaming": streaming})

    for answer_result in answer_result_stream_result['answer_result_stream']:
        resp = answer_result.llm_output["answer"]
        history = answer_result.history
        history[-1][0] = query

        # Try to parse the response into ResumeJson format
        try:
            resume_data = ResumeJson.parse_raw(resp)
            return resume_data, history
        except ValidationError as e:
            # If the response cannot be parsed into ResumeJson format, return the error
            return {"error": str(e)}, history
