from typing import List
import random
import _thread as thread
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.llms.base import LLM
import copy
import json
from typing import Optional, List, Dict, Mapping, Any
from datetime import datetime
from time import mktime
from wsgiref.handlers import format_date_time
import hmac
import base64
import hashlib
import ssl
import websocket

from urllib.parse import urlencode
from urllib.parse import urlparse


class Params():
    url = "wss://spark-api.xf-yun.com/v2.1/chat"
    domain = "generalv2"
    app_id = ""
    api_secret = ""
    api_key = ""
    max_tokens = 1024
    temperature = 0.5


class XunFeiSparkWsModel():

    class Ws_Param(object):
        def __init__(self, APPID, APIKey, APISecret, Spark_url):
            self.APPID = APPID
            self.APIKey = APIKey
            self.APISecret = APISecret
            self.host = urlparse(Spark_url).netloc
            self.path = urlparse(Spark_url).path
            self.Spark_url = Spark_url

        def create_url(self):
            now = datetime.now()
            date = format_date_time(mktime(now.timetuple()))

            signature_origin = "host: " + self.host + "\n"
            signature_origin += "date: " + date + "\n"
            signature_origin += "GET " + self.path + " HTTP/1.1"

            signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                     digestmod=hashlib.sha256).digest()

            signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

            authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

            authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

            v = {
                "authorization": authorization,
                "date": date,
                "host": self.host
            }
            url = self.Spark_url + '?' + urlencode(v)
            return url

    def __init__(self,
                 appid = Params.app_id,
                 api_key = Params.api_key,
                 api_secret = Params.api_secret,
                 url = Params.url):
        wsParam = XunFeiSparkWsModel.Ws_Param(appid, api_key, api_secret, url)
        self.ws_param = wsParam
        websocket.enableTrace(False)
        self.answer = ""

    def on_error(self, ws, error):
        print("### error:", error)

    def on_close(self, ws, one, two):
        print(" ")

    def on_open(self, ws):
        thread.start_new_thread(self.run, (ws,))

    def run(self, ws, *args):
        data = json.dumps(self.gen_params(appid=ws.appid, domain= ws.domain,question=ws.question))
        ws.send(data)

    def on_message(self, ws, message):
        # print(message)
        data = json.loads(message)
        code = data['header']['code']
        if code != 0:
            print(f'an error occurred: {code}, {data}')
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            print(content,end ="")

            self.answer += content
            # print(1)
            if status == 2:
                ws.close()

    def gen_params(self, appid, domain, question):
        data = {
            "header": {
               "app_id": appid,
                "uid": "1234"
            },
            "parameter": {
                "chat": {
                    "domain": domain,
                    "temperature": Params.temperature,
                    "max_tokens": Params.max_tokens,
                    "auditing": "default"
                }
            },
            "payload": {
                "message": {
                   "text": question
                }
            }
        }
        return data

    def __call__(self, *args, **kwargs):
        wsUrl = self.ws_param.create_url()
        ws = websocket.WebSocketApp(wsUrl,
                                    on_message=self.on_message, on_error=self.on_error,
                                    on_close=self.on_close, on_open=self.on_open)
        ws.appid = self.ws_param.APPID
        ws.question = kwargs["question"]
        ws.domain = Params.domain
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        ws.close()
        return self.answer

    def clear(self):
        self.answer = ""


class XunFeiSparkModel(LLM):
    def __init__(self):
        super(XunFeiSparkModel, self).__init__()
        super.__setattr__(self, "_ws", XunFeiSparkWsModel())

    def _llm_type(self) -> str:
        return "xunfei_spark"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        answer = self._ws(question=[{'role':'user','content':prompt}])
        self._ws.clear()
        return answer


class Instruction:
    def __init__(self,instruction):
        self.instruction = instruction
        self.info = {}
        self.deep = 1


class Evol:

    def __init__(self,llm = XunFeiSparkModel()):

        self.chain_bs_0 = self.gen_chain(llm,self._bs_template())
        self.chain_ds_1 = self.gen_chain(llm,self.deepening_ds_template())
        self.chain_ds_2 = self.gen_chain(llm,self.concretizing_ds_template())
        self.chain_ds_3 = self.gen_chain(llm,self.increased_reasoning_steps_ds_template())
        self.chain_ds_4 = self.gen_chain(llm,self.add_constraints_ds_template())

        self.chain_check = self.gen_chain(llm,self.check_template())

        self.load_function = self.load_instructions

        self.save_function = self.save_instructions

    def register_load_function(self,func):
        self.load_function = func

    def register_save_function(self,func):
        self.save_function = func

    def _bs_template(self) -> PromptTemplate:
        '''
        I want you act as a Prompt Creator.
        Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.
        This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.
        The LENGTH and difficulty level of the #Created Prompt# should be similar to that of the #Given Prompt#.
        The #Created Prompt# must be reasonable and must be understood and responded by humans. '#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#.
        #Given Prompt#:
        <Here is instruction.>
        #Created Prompt#:
        :return:
        '''
        t = PromptTemplate(
            input_variables=["instruction"],
            template=
                "I want you act as a Prompt Creator.\n"+
                "Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\n"+
                "This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\n"+
                "The LENGTH and difficulty level of the #Created Prompt# should be similar to that of the #Given Prompt#.\n"+
                "The #Created Prompt# must be reasonable and must be understood and responded by humans."+
                "'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' "
                "are not allowed to appear in #Created Prompt#.\n"+
                "#Given Prompt#:\n"+
                "<{instruction}>\n"+
                "#Created Prompt#:",
        )
        return t

    def add_constraints_ds_template(self) -> PromptTemplate:
        '''
        I want you act as a Prompt Rewriter.
        Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle.
        But the rewritten prompt must be reasonable and must be understood and responded by humans.
        Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the input in #Given Prompt#.
        You SHOULD complicate the given prompt using the following method:
        Please add one more constraints/requirements into #Given Prompt#
        You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
        '#Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
        #Given Prompt#:
        <Here is instruction.>
        #Rewritten Prompt#:
        :return:
        '''
        t = PromptTemplate(
            input_variables=["instruction"],
            template = "I want you act as a Prompt Rewriter.\n"+
                       "Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle.\n"+
                       "But the rewritten prompt must be reasonable and must be understood and responded by humans.\n"+
                       "Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the input in #Given Prompt#.\n"+
                       "You SHOULD complicate the given prompt using the following method:\n"+
                       "Please add one more constraints/requirements into #Given Prompt#\n"+
                       "You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.\n"+
                       "'#Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n"+
                       "#Given Prompt#:\n"+
                       "<{instruction}>\n"+
                       "#Rewritten Prompt#:",
        )
        return t

    def deepening_ds_template(self) -> PromptTemplate:
        '''
        I want you act as a Prompt Rewriter.
        Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle.
        But the rewritten prompt must be reasonable and must be understood and responded by humans.
        Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the input in #Given Prompt#.
        You SHOULD complicate the given prompt using the following method:
        If #Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased. or
        You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
        '#Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
        #Given Prompt#:
        <Here is instruction.>
        #Rewritten Prompt#:
        :return:
        '''
        t = PromptTemplate(
            input_variables=["instruction"],
            template = "I want you act as a Prompt Rewriter.\n"+
                        "Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle.\n"+
                        "But the rewritten prompt must be reasonable and must be understood and responded by humans.\n"+
                        "Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the input in #Given Prompt#.\n"+
                        "You SHOULD complicate the given prompt using the following method:\n"+
                        "If #Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased. or\n"+
                        "You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.\n"+
                        "'#Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n"+
                        "#Given Prompt#:\n"+
                        "<{instruction}>\n"+
                        "#Rewritten Prompt#:",
        )
        return t

    def concretizing_ds_template(self) -> PromptTemplate:
        """
        I want you act as a Prompt Rewriter.
        Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle.
        But the rewritten prompt must be reasonable and must be understood and responded by humans.
        Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the input in #Given Prompt#.
        You SHOULD complicate the given prompt using the following method:
        Please replace general concepts with more specific concepts. or
        You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
        '#Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
        #Given Prompt#:
        <Here is instruction.>
        #Rewritten Prompt#:
        """
        t = PromptTemplate(
            input_variables=["instruction"],
            template = "I want you act as a Prompt Rewriter.\n"+
                       "Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle.\n"+
                       "But the rewritten prompt must be reasonable and must be understood and responded by humans.\n"+
                       "Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the input in #Given Prompt#.\n"+
                       "You SHOULD complicate the given prompt using the following method:\n"+
                       "Please replace general concepts with more specific concepts. or\n"+
                       "You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.\n"+
                       "'#Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n"+
                       "#Given Prompt#:\n"+
                       "<{instruction}>\n"+
                       "#Rewritten Prompt#:",
        )
        return t

    def increased_reasoning_steps_ds_template(self) -> PromptTemplate:
        '''
        I want you act as a Prompt Rewriter.
        Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle.
        But the rewritten prompt must be reasonable and must be understood and responded by humans.
        Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the input in #Given Prompt#.
        You SHOULD complicate the given prompt using the following method:
        If #Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.
        You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
        '#Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
        #Given Prompt#:
        <Here is instruction.>
        #Rewritten Prompt#:
        :return:
        '''
        t = PromptTemplate(
            input_variables=["instruction"],
            template =  "I want you act as a Prompt Rewriter.\n"+
                        "Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle.\n"+
                        "But the rewritten prompt must be reasonable and must be understood and responded by humans.\n"+
                        "Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the input in #Given Prompt#.\n"+
                        "You SHOULD complicate the given prompt using the following method:\n"+
                        "If #Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.\n"+
                        "You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.\n"+
                        "'#Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n"+
                        "#Given Prompt#:\n"+
                        "<{instruction}>\n"+
                        "#Rewritten Prompt#:",
        )
        return t

    def check_template(self) -> PromptTemplate:
        '''
        Here are two Instructions to ChatGPT AI, do you think they are equal to each other, which meet the following requirements:
        1. They have same constraints and requirments.
        2. They have same depth and breadth of the inquiry.
        The First Prompt: <Here is first instruction.>
        The Second Prompt: <Here is second instruction.>
        Your Judgement (Just answer: Equal or Not Equal. No need to explain the reason.):
        :return:
        '''

        t = PromptTemplate(
            input_variables=["first_instruction", "second_instruction"],
            template = "Here are two Instructions to ChatGPT AI, do you think they are equal to each other, which meet the following requirements:\n"+
                       "1. They have same constraints and requirments.\n"+
                       "2. They have same depth and breadth of the inquiry.\n"+
                       "The First Prompt: <{first_instruction}>\n"+
                       "The Second Prompt: <{second_instruction}>\n"+
                       "Your Judgement (Just answer: Equal or Not Equal. No need to explain the reason.):",
        )
        return t


    def gen_chain(self,llm,prompt_template):

        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
        )

        # llm_chain = None
        return llm_chain

    def run_chain(self,chain,input_variables):
        return chain(inputs=input_variables)
        #return {"text":"use Chain to generate a prompt."}


    def inner_loop(self,
                   max_evol_deep=5,
                   max_evol_step=30,
                   **kwarg):
        instructions_list = self.load_function(**kwarg)
        step = 0
        while step < max_evol_step:
            step += 1

            # 1. randomly select a instruction
            instruction = random.choice(instructions_list)
            # 2. randomly select a template
            if instruction.deep >= max_evol_deep:
                i = 0
                c = self.chain_bs_0
                input_variables = {"instruction": instruction.instruction}
            else:
                i = random.choice([0,1,2,3,4])
                if i == 0:
                    c = self.chain_bs_0
                    input_variables = {"instruction":instruction.instruction}
                elif i == 1:
                    c = self.chain_ds_1
                    input_variables = {"instruction": instruction.instruction}
                elif i == 2:
                    c = self.chain_ds_2
                    input_variables = {"instruction": instruction.instruction}
                elif i == 3:
                    c = self.chain_ds_3
                    input_variables = {"instruction": instruction.instruction}
                else:
                    c = self.chain_ds_4
                    input_variables = {"instruction": instruction.instruction}

            # 3. generate a prompt
            result_prompt = self.run_chain(c, input_variables)
            if result_prompt is None:
                continue
            else:
                r = result_prompt["text"]

            # 4. check the prompt
            check_input_variables = {"first_instruction": instruction.instruction, "second_instruction": r}

            check_prompt = self.run_chain(self.chain_check, check_input_variables)
            if check_prompt is None:
                continue
            else:
                equal = check_prompt["text"]
                if "equal" in equal.lower():
                    new_i = copy.deepcopy(instruction)
                    new_i.instruction = r
                    if i != 0:
                        new_i.deep = instruction.deep + 1
                    else:
                        new_i.deep = instruction.deep
                    # add new instruction into instructions_list
                    instructions_list.append(new_i)
                    print("step: ", step, "add instruction by chain: ", i)

        self.save_function(instructions_list,**kwarg)

    def save_instructions(self,instructions_list,save_path,**kwarg):
        json_list = []

        for i in instructions_list:
            inst = i.instruction
            info = i.info
            deep = i.deep
            json_dict = {"instruction":inst,"info":info,"deep":deep}
            json_str = json.dumps(json_dict,ensure_ascii=False)
            json_list.append(json_str)
        with open(save_path,"w",encoding='utf-8') as f:
            json.dump(json_list,f,ensure_ascii=False,indent=4)


    def load_instructions(self,load_path,**kwarg):
        inst_lists = []
        with open(load_path,"r",encoding='utf-8') as f:
            json_list = json.load(f)
        for json_str in json_list:
            q = json_str["question"]
            a = json_str["answer"]
            if "context" in json_str.keys():
                c = json_str["context"]
            else:
                continue
            inst = "根据内容：{}\n回答以下问题：{}".format(c,q)
            info = {"question":q,"answer":a,"context":c}
            i = Instruction(inst)
            i.info = info
            inst_lists.append(i)
        return inst_lists


if __name__ == '__main__':

    orign_instruction_save_path = r""
    save_path = r""

    e = Evol()
    e.inner_loop(
        max_evol_deep=5,
        max_evol_step=30,
        load_path=orign_instruction_save_path,
        save_path=save_path,
    )