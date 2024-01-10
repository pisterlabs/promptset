import datetime
import json
import logging
import re
from typing import Any, Optional
from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from config import topp_
from funtion_basetool import functional_Tool
from utils import parse_json_markdown, get_current_weekday, is_xyzchar, have_Ch


class Model_Tool(functional_Tool):
    llm: BaseLanguageModel

    llm_chain: LLMChain = None
    name: str
    description: str
    id: str
    prompt_dict: dict
    prompt: Optional[Any]
    sub_param_type: dict

    def _call_func(self, query):

        sub_param = {k: None for k, v in self.sub_param_type.items()}
        id = self.id

        if "#" in query and query.index("#") == 0:
            return self.id, json.dumps(sub_param, ensure_ascii=False)

        if len(sub_param) <= 0 or len(query) <= 7 or query == "帮助":
            return self.id, json.dumps(sub_param, ensure_ascii=False)

        self.prompt = self.prompt_dict[id]
        current_time = datetime.datetime.now()
        current_time = str(current_time)[:19] + "," + get_current_weekday()
        current_date = current_time[:10]
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        before_yesterday = today - datetime.timedelta(days=2)
        parm_str = json.dumps(sub_param, ensure_ascii=False)
        user_input = self.prompt.format(user_input=query, current_time=current_time, current_date=current_date,
                                        yesterday=yesterday.__str__(), before_yesterday=before_yesterday.__str__(),
                                        parm_str=parm_str)
        i = 0
        n = 3
        self.llm.top_p = 0
        while i < n:
            resp_p = self.llm.predict(user_input)
            resp = resp_p
            logging.info(f"<chat>\n\nquery:\t{user_input}\n<!-- *** -->\nresponse:\n{resp}\n\n</chat>")
            try:
                resp = parse_json_markdown(resp)
                for k, v in resp.items():
                    if not v or k not in sub_param:
                        continue
                    if "未知" in v.strip() or "幸福西饼"==v.strip() or is_xyzchar(v.strip(), query.strip()):
                        resp[k] = None

                for k, v in sub_param.items():
                    if k in resp and sub_param[k] is None:
                        sub_param[k] = resp[k]

                    if self.sub_param_type[k].lower() == "string" and isinstance(sub_param[k], list):
                        if len(sub_param[k]) > 0:
                            sub_param[k] = str(sub_param[k][0])

                    if sub_param[k] :
                        orderno=re.findall(r"XS[a-zA-Z0-9]+", sub_param[k])
                        if len(orderno)>0:
                            orderno=orderno[0]
                            sub_param[k]=orderno
                            continue
                        if  k.lower() in ["orderno"] or "order" in k.lower() :
                            orderno = re.findall(r"[a-zA-Z0-9]{5,}", sub_param[k])
                            if len(orderno) > 0:
                                orderno = orderno[0]
                                sub_param[k] = orderno
                            else:
                                sub_param[k]=None
                                continue
                    ##带shop的字段
                    if sub_param[k] and "shop" in k.lower():
                        if sub_param[k].endswith("省") or sub_param[k].endswith("市") :
                            sub_param[k]=None
                            continue

                        ##名称长度小于三就忽略
                        if len(sub_param[k])<3  and sub_param[k] :
                            sub_param[k]=None
                            continue
                        shopid=re.match(r"\d{8,}", sub_param[k])
                        if shopid:
                            sub_param[k]=shopid.group(0)
                break
                # ##超过一半的参数被提取出来之后，就不再启动重试机制
                # s=sum([1 if e == None else 0 for e in sub_param.values()])
                # if s/len(sub_param) >=0.5 and i<n :
                #     i+=1
                #     continue
                # else:
                #     break
            except:
                if i >= n - 1:
                    break
                i += 1
            finally:
                self.llm.top_p = topp_

        return self.id, json.dumps(sub_param, ensure_ascii=False)

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm)


class Unknown_Intention_Model_Tool(functional_Tool):
    llm: BaseLanguageModel
    id: str = "000000"
    llm_chain: LLMChain = None
    name: str = "意图类别不明"
    description: str = "当其他意图类别不匹配时请选择该意图,用户随便询问的内容,比如用户说：你好、hello、HI、写一个策划、西红柿炒蛋怎么做等等与幸福西饼意图无关的问题"

    def _call_func(self, query):
        if query.count("user"):
            query = f"system:当前你是幸福西饼AI客服助手，请你尽量使用中文回答用户的问题\n{query}"
        else:
            query = f"system:当前你是幸福西饼AI客服助手，请你尽量使用中文回答用户的问题\nuser:{query}"

        # response = self.llm.predict(query)
        # logging.info(f"<chat>\n\nquery:\t{query}\n<!-- *** -->\nresponse:\n{response}\n\n</chat>")
        return "", ""

    def get_llm_chain(self):
        pass
