import asyncio
import logging
import re
import time
from functools import wraps
from logging.handlers import RotatingFileHandler
from typing import Optional
import datetime

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain.agents import AgentExecutor
import config
from MyOpenAI import myOpenAi, call_qwen_funtion
from bm25 import BM25
from config import saveinterfacepath
from doc import Doc
from intentAgent_model import IntentAgent
from prompt_helper import init_all_fun_prompt, FUNTION_CALLING_FORMAT_INSTRUCTIONS
from redis_manger import get_version, set_version
from tool_model import Model_Tool, Unknown_Intention_Model_Tool
from utils import load_interface_template, save_interface_template, is_true_number, is_xxCH, get_current_weekday

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
)
file_handler = RotatingFileHandler("./data/chat_api.log", 'a', 10*1024*1024, 360,encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
app = FastAPI()
get_bearer_token = HTTPBearer(auto_error=False)
from api_protocol import (
    InitInterfaceResponse,
    InitInterfaceRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    FunCompletionRequest,
    ChatResponse,
    TemplateResponse,
    Intention_Search_Response,
    Funtion, Beautify_ChatCompletionRequest, ChatMessage, Chat_LinksResponse, LinksResp, Full_CompletionRequest,
    FuntionResp
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name
@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=401,
        content={"status": 402,"message":f"{exc.name}"},)


####启动程序,初始化全局变量
def init_run():
    global  agent_exec,toos_dict,llm,initparam,search
    initparam = load_interface_template(saveinterfacepath)
    if not initparam:
        return  None,None,None,None,None
    llm = myOpenAi()
    toos_dict = {}
    docs=[]
    prompt_dict=init_all_fun_prompt(initparam)
    for param in initparam.params  :
        if param.usableFlag:
            sub_param_type={e.name:e.type for e in param.inputParams}
            toos_dict[param.id]=Model_Tool(name=param.name,description=param.functionDesc,id=param.id,llm=llm,prompt_dict=prompt_dict,sub_param_type=sub_param_type)
            docs.append(Doc(funtion_id=param.id, name=param.name))
    if len(docs)==0:
        search=None
        agent_exec=None
        toos_dict=None
    else:
        search =BM25(docs)
        tools=list(toos_dict.values())
        unknowntool=Unknown_Intention_Model_Tool(llm=llm)
        tools.append(unknowntool)
        # # 选择工具
        agent = IntentAgent(tools=tools, llm=llm,default_intent_name=unknowntool.name)
        agent_exec = AgentExecutor.from_agent_and_tools(agent=agent,  tools=tools, verbose=False,max_iterations=1)
    return agent_exec,toos_dict,llm,initparam,search

agent_exec,toos_dict,llm,initparam,search=init_run()
current_version=get_version()
def raise_UnicornException(func):  # 定义一个名为 raise_UnicornException 的装饰器函数，它接受一个参数 func这个 func 就是即将要被修饰的函数
    @wraps(func)
    async def wrapper( *args, **kwargs):  # 在 raise_UnicornException() 函数内部，定义一个名为 wrapper() 的闭包函数
        global agent_exec, toos_dict, llm, initparam, search, current_version
        try:
            start_time = time.time()  # 程序开始时间
            version=get_version()
            if current_version != version:
                agent_exec, toos_dict, llm, initparam, search = init_run()
                current_version=version
            res = await func(*args, **kwargs)
            ###认证机制不记录参数
            if func.__name__!="check_api_key" :
                end_time = time.time()  # 程序结束时间
                run_time = end_time - start_time  # 程序的运行时间，单位为秒
                logging.info(f"接口：{func.__name__},前端参数为：{args} {kwargs},运行时间：{run_time},返回值：{res}")
        except HTTPException as e:
            info = str(e.detail)
            logging.info(f"接口：{func.__name__}，接口异常错误提示：{info}")
            raise UnicornException(name=info)
        except  Exception as e:
            info=str(e)
            logging.info(f"接口：{func.__name__}，接口异常错误提示：{info}")
            raise  UnicornException(name=info)
        return res
    return wrapper

@raise_UnicornException
async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if config.api_keys and config.check_api_key :
        if auth is None or (token := auth.credentials) not in config.api_keys:
            raise HTTPException(status_code=401,detail="invalid Authorization api_key")
        return True
    else:
        return False

@app.post("/chat/completions", response_model=ChatResponse,summary="跟OPENAI 接口一样,无业务处理", dependencies=[Depends(check_api_key)])
# @app.post("/chat/completions", response_model=ChatResponse,summary="跟OPENAI 接口一样,无业务处理")
@raise_UnicornException
async def chat(request: ChatCompletionRequest):
    response=call_qwen_funtion(request.message)
    resp=response.choices[0].message.content
    return ChatResponse(status=200,message=resp)


@app.post("/beautify_chat/completions", response_model=ChatResponse,summary="调用函数，产生的结果，由AI自动组织语言回复")
@raise_UnicornException
async def beautify_chat(request: Beautify_ChatCompletionRequest):
    global  toos_dict,llm
    current_time = datetime.datetime.now()
    current_time = str(current_time)[:19] + "," + get_current_weekday()
    current_date = current_time[:10]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    before_yesterday = today - datetime.timedelta(days=2)

    funname_resp = request.funname_resp
    funname_resp.append(FuntionResp(funtion_id="00000",resp="未查到信息，请尝试咨询其他业务"))
    query=[]
    for i, res in enumerate(funname_resp):
        tool=toos_dict.get(res.funtion_id,None)
        if tool:
            description=tool.description
            resp=f"信息{i+1}:{description},查询结果详情:{res.resp}"
        else:
            # if i>0:
            #     resp = f"信息{i+1}:未知业务,查询结果详情,用户问题无法解答,【以上业务不满足用户问题,选择该业务信息回复】"
            # else:
            resp = f"信息{i+1}:{res.resp}"
        query.append(resp)
    query="\n".join(query)

    content = FUNTION_CALLING_FORMAT_INSTRUCTIONS.format(content=query,current_time=current_time)

    if request.message[-1].content.strip()=="":
       return ChatResponse(status=200, message="请输入你的问题")

    mess = request.message
    if mess[0].role == "system":
        mess.pop(0)
    mess = mess[-6:]
    mess.insert(0, ChatMessage(role="system", content=content))
    for i,role in enumerate(mess):
        if role.role in [ "assistant", "system","function"]:
            continue
        mc = role.content.strip()
        if re.findall(r"XS[a-zA-Z0-9]+", mc) and i==len(mess)-1:
            role.content = mc + "(查看订单号详情,请仅仅根据幸福西饼已知查询业务信息回复我的问题相关业务)"
        else:
            if i==len(mess)-1:
                role.content = mc + "(请仅仅根据聊天上下文或幸福西饼已知查询信息回复我的问题)"

    history = merge_message(mess)
    top_p = 0.0 if np.random.random()<0.8 else 0.1
    response = call_qwen_funtion(mess, top_p=top_p)
    resp = response.choices[0].message.content
    i=1
    n=3
    while i<=n:
        print(i, resp)
        top_p=0.8
        i+=1
        #"？" not in resp and "?" not in resp and
        if  is_true_number(resp,history)  and not is_xxCH(resp,history) and  "你的回复" not in resp and "系统背景" not in resp and "用户问题" not in resp :
            logging.info(f"<chat>\n\nquery:\t{history}\n<!-- *** -->\nresponse:\n{resp}\n\n</chat>")
            i=0
            break
        else:
            response = call_qwen_funtion(mess,top_p=top_p)
            resp = response.choices[0].message.content
            logging.info(f"<chat>\n\nquery:\t{history}\n<!-- *** -->\nresponse:\n{resp}\n\n</chat>")
    if i>n:
        resp="很抱歉，我无法提供您需要的信息。请咨询客服以获取更多帮助"

    return ChatResponse(status=200, message=resp)



@app.post("/init_funtion_template/completions", response_model=InitInterfaceResponse,summary="初始化函数模板")
@raise_UnicornException
async def init_funtion_template(request: InitInterfaceRequest):
    global  initparam,current_version
    if initparam  :
        interface_fun = {param.id:param for param in initparam.params}
        for param in request.params:
            interface_fun[param.id] = param
            if not param.usableFlag:
                del interface_fun[param.id]
        initparam.params=list(interface_fun.values())
    else:
        initparam=request
    save_interface_template(initparam, saveinterfacepath)
    init_run()
    set_version()
    current_version=get_version()
    res=InitInterfaceResponse(status=200,message="添加模板成功")
    return res


@app.post("/get_all_template/completions", response_model=TemplateResponse,summary="获取上传的所有模板")
@raise_UnicornException
async  def get_all_template():
    initparam = load_interface_template(saveinterfacepath)
    return TemplateResponse(status=200,message="获取模板成功",template=initparam)

def merge_message(message):

    if isinstance(message,str):
        return "user:"+message
    history=[]
    if isinstance(message,list):
        for chatMessage in message:
            history.append(f"{chatMessage.role}:{chatMessage.content}")
    history="\n".join(history)
    # logging.info(f"具体参数：{history}")
    return history


def inject_order_detail(request):
    for role in request.message:
        content = role.content.strip()
        order = re.findall(r"XS[a-zA-Z0-9]+", content)
        if len(order) > 0 and len(order[0])>=5:
            order = order[0]
            role.content=role.content.replace(order, f"{order}(订单号)")

@app.post("/chat_funtion_intention/completions", response_model=ChatCompletionResponse,summary="意图识别跟函数解析，带funtion_id参数就是函数参数解析，否则就是意图识别")
@raise_UnicornException
async def chat_funtion_intention(request: FunCompletionRequest):
    global  agent_exec,toos_dict
    inject_order_detail(request)
    if request.funtion_id is None or request.funtion_id=='':
        query=merge_message(request.message)
        fun_id,message=agent_exec.run(query)
        fun_id=fun_id or ""
        return ChatCompletionResponse(status=200,funtion_id=fun_id,message=message)
    else:

        tool = toos_dict.get(request.funtion_id, None)
        if not tool:
            raise  HTTPException(status_code=401,detail=f"invalid funtion_id:{request.funtion_id} 不存在")
        query = merge_message(request.message)
        _, message = tool._run(query)
        return ChatCompletionResponse(status=200, funtion_id=request.funtion_id, message=message)

async  def intention_search(request):
    global agent_exec, toos_dict,search
    query = request.message[-1].content.strip()
    inject_order_detail(request)
    mess = merge_message(request.message)
    m = re.findall(r"^XS[a-zA-Z0-9]+$", query)
    if len(query) >= 5 and len(m) > 0 and len(m[0]) == len(query):
        docs1 = search.calc_similarity_rank("订单号查询")
        docs2 = []
    elif query == "帮助" or query == "#帮助":
        docs1 = search.calc_similarity_rank("帮助")
        docs2 = []
        mess="帮助"
    elif "#" in query and query.index("#")==0:
        docs1 = search.calc_similarity_rank(query)
        docs2 = []
        mess = query
    else:
        docs1, docs2 = await asyncio.gather(search.cal_similarity_rank(query), agent_exec.agent.choose_tools(mess))

    docs1, docs2=set(docs1),set(docs2)
    d=docs2.intersection(docs1)
    docs1=docs1-d
    docs=list(docs2)+list(docs1)
    return docs,mess
@app.post("/chat_multi_intention/completions", response_model=Chat_LinksResponse,summary="多意图识别并函数解析")
@raise_UnicornException
async def chat_multi_intention(request: Full_CompletionRequest):
    global agent_exec, toos_dict,search
    docs,mess=await  intention_search(request)
    tools=[]
    docs.sort(key=lambda x: x.funtion_id)

    for doc in docs:
        tool = toos_dict[doc.funtion_id]
        tools.append(tool)

    tools_res=await asyncio.gather(*[task._arun(mess) for task in tools ])
    ret=[]
    for doc,tool_res in zip(docs,tools_res):
        ret.append(LinksResp(funtion_id=doc.funtion_id,fro=doc.fro,name=doc.name,message=tool_res[1]))
    return Chat_LinksResponse(status=200, tools=ret)



@app.post("/chat_intention_search/completions", response_model=Intention_Search_Response,summary="多意图识别")
@raise_UnicornException
async def chat_intention_search(request: ChatCompletionRequest):
    global agent_exec, toos_dict,search
    docs, mess = await  intention_search(request)
    funtions=[Funtion(id=doc.funtion_id,name=doc.name,fro=doc.fro) for doc in docs]
    funtions.sort(key=lambda  x:x.id)
    return Intention_Search_Response(status=200,funtions=funtions)




if __name__ == "__main__":


    uvicorn.run("chat_api:app", host='0.0.0.0', port=8084, workers=1)



