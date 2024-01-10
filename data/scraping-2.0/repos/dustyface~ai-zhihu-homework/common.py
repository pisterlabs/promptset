import os
import json
import copy
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from .prompt_text import instruction, output_format, examples

_ = load_dotenv(find_dotenv())

# print("OPENAI_API_KEY=", os.getenv("OPENAI_API_KEY"))
# print("OPENAI_API_BASE=", os.getenv("OPENAI_API_BASE"))
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE")
)


def _debug_print(*args):
    position = 0
    name = ""
    for arg in args:
        if position % 2 == 1:
            print(f"type({name[0:-1]})=", type(arg))
            print(f"{name if name else '='}", arg)
            print("===")
        else:
            if isinstance(arg, str) and arg[-1] == "=":
                name = arg
        position += 1


def get_completion(prompt, **kwargs):
    response = get_completion_through_dict(prompt=prompt, **kwargs)
    return response.choices[0].message.content


def get_completion_through_dict(**kwargs):
    args = {k: v for k, v in kwargs.items() if v}
    if not "model" in args:
        args["model"] = "gpt-3.5-turbo"
    if not "temperature" in args:
        args["temperature"] = 0
    if "prompt" in args and not "message" in args:
        messages = [{"role": "user", "content": args["prompt"]}]
        args["messages"] = messages
        del args["prompt"]
    # remove the deubg param, openai api doesn't need it
    debug = False
    if "debug" in args:
        debug = args["debug"]
        del args["debug"]
    if debug:
        _debug_print("args=", args)
    response = client.chat.completions.create(**args)

    # 有时 completions接口返回的是string
    # response_dict = json.loads(response)
    if debug:
        _debug_print("response=", response)
        # debug_print("response_dict", response_dict)
    return response


session = []


def fill_context_in_session(prompt):
    session.append(prompt)


def get_completion_with_session(prompt, **kwargs):
    debug = kwargs.get("debug") if kwargs.get("debug") else False
    model = kwargs.get("model") if kwargs.get("model") else "gpt-3.5-turbo"
    temperature = kwargs.get("temperature") if kwargs.get("temperature") else 0

    session.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model, messages=session, temperature=temperature
    )
    if debug:
        _debug_print(response)
    response_msg = response.choices[0].message.content
    session.append({"role": "assistant", "content": response_msg})
    return response_msg


def func_calling_call_func(response_message, messages, tools, callback={}, *func_name):
    """
    根据第1轮LLM回复的functoin calling参数，调用 callback 中的函数，得到第2轮LLM的回复;(注意对messages历史信息的更新)
    @param response_message: 第1轮LLM回复的functoin calling参数
    @param messages: 对话历史信息
    @param callback: callback 是function calling的函数定义
    @param func_name: func_name是function calling的函数名字; 如果不指定，则调用function calling中所有的函数
    """
    tool_calls = response_message.tool_calls
    if tool_calls is not None:
        messages.append(response_message)
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            if len(func_name) > 0 and fn_name not in func_name:
                continue
            fn_args = json.loads(tool_call.function.arguments)
            fn = callback[fn_name]
            if fn:
                fn_response = fn(**fn_args)
                print("result=", fn_response)
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": fn_name,
                        "content": str(fn_response),
                    }
                )
        second_response = get_completion_through_dict(
            messages=messages,
            model="gpt-3.5-turbo-1106",
            tools=tools,
            seed=1024,  # seed是保证确定性的参数;
            # debug=True,
        )
        return second_response


class NLU:
    """
    NLU: Natural Language Understanding
    - NLU prompt_template 包含 instruction, ouput_format等第一轮和LLM对话的模板，__INPUT__将会替换为用户输入的内容;
    - NLU => (json) => DST
    """

    def __init__(self):
        self.prompt_template = (
            f"{instruction}\n\n{output_format}\n\n{examples}\n\n用户输入: \n__INPUT__"
        )

    def _get_completion(self, prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        # json.loades, 和js中的JSON.parse("{a:１}")类似;
        # {k: v for .... if ...}, 是dictionary comprehension; 是创建openai response content中的值不为falsy的部分为另一个dictionary
        semantics = json.loads(response.choices[0].message.content)
        return {k: v for k, v in semantics.items() if v}

    # 和LLM的第一轮对话，获得用户意图中的json结构信息;
    def parse(self, user_input):
        prompt = self.prompt_template.replace("__INPUT__", user_input)
        return self._get_completion(prompt)


class DST:
    """
    DST: Dialogue State Tracking
    - 目前DST update是做一些数据的清洗/转换工作；i.e. 如果name出现在从NLU传过来的json中，则清除原有state中的信息(可能意味着换了一个套餐); 根据sort信息清洗，etc
    - 目前以上两种清洗，在很多json中都没有用到, 基本上成了透传;

    NLU => (json) => DST => (json) => DB
    """

    def __init__(self):
        pass

    # 把LLM回答的内容，做数据转换，更新到state中
    def update(self, state, nlu_semantics):
        print("nlu_semantics=")
        print(nlu_semantics)

        if "name" in nlu_semantics:
            state.clear()
        # 清除已有sort的信息;
        if "sort" in nlu_semantics:
            slot = nlu_semantics["sort"]["value"]
            if slot in state and state[slot]["operator"] == "==":
                del state[slot]
        for k, v in nlu_semantics.items():
            state[k] = v
        return state


class MockedDB:
    """
    MockedDB
    - 可以被视为Policy处理的部分；
    - retrieve方法依据DST吐出的json信息，在MockedDB.data中检索符合条件的套餐，把符合的item加进records; 核心的比较是 `if not eval(str(r[k]) + v["operator"] + str(v["value"])):`这句; 把符合条件的都加入了records，然后排序;
    """

    def __init__(self):
        self.data = [
            {"name": "经济套餐", "price": 50, "data": 10, "requirement": None},
            {"name": "畅游套餐", "price": 180, "data": 100, "requirement": None},
            {"name": "无限套餐", "price": 300, "data": 1000, "requirement": None},
            {"name": "校园套餐", "price": 150, "data": 200, "requirement": "在校生"},
        ]

    # 根据LLM回答的信息， 从MockedDB中检索出符合条件的套餐信息
    # params: kwargs, 是一个dictionary, 里面包含了LLM回答的信息
    def retrieve(self, **kwargs):
        records = []
        for r in self.data:
            select = True
            if r["requirement"]:
                if "status" not in kwargs or kwargs["status"] != r["requirement"]:
                    continue
            for k, v in kwargs.items():
                if k == "sort":
                    continue
                if k == "data" and v["value"] == "无上限":
                    if r[k] != 1000:
                        select = False
                        break
                if "operator" in v:
                    # 用eval验证r[k]是否和kwargs中的v["value"]满足v["operator"]的关系;(e.g. v["operator"] <= 的关系)
                    if not eval(str(r[k]) + v["operator"] + str(v["value"])):
                        select = False
                        break
                elif str(r[k]) != str(v):
                    select = False
                    break
            print("select=")
            print(select)
            if select:
                records.append(r)
        if len(records) <= 1:
            return records
        # 如果找到的信息大于1条, 则根据sort字段进行排序;
        key = "price"
        reverse = False
        if "sort" in kwargs:
            key = kwargs["sort"]["value"]
            reverse = kwargs["sort"]["ordering"] == "descend"
        print("records=")
        print(records)
        return sorted(records, key=lambda x: x[key], reverse=reverse)


class DialogManager:
    """
    DialogManager: 管理了2轮用户和LLM的对话
    - constructor接受的参数prompt_templates是第2轮user询问LLM的模板，其中需要加入第一轮的用户输入和LLM回答的信息(从MockDB retrieve()得来的)
    - _wrap, 是将第一轮的信息，加入到第二轮的模板中，形成第二轮的用户输入;
    - run, 执行2轮对话;
    """

    def __init__(self, prompt_templates):
        self.state = {}
        self.session = [
            {"role": "system", "content": "你是一个手机流量套餐的客服代表，你叫小瓜。可以帮助用户选择最合适的流量套餐产品。"}
        ]
        self.nlu = NLU()
        self.dst = DST()
        self.db = MockedDB()
        self.prompt_templates = prompt_templates

    # 将prompt_templates中的__DATA__, __INPUT__等字段，替换成records中的具体的值;
    def _wrap(self, user_input, records):
        if records:
            prompt = self.prompt_templates["recommand"].replace("__INPUT__", user_input)
            # 只取records(LLM回答的dictionary中的第一个元素)
            r = records[0]
            for k, v in r.items():
                prompt = prompt.replace(f"__{k.upper()}__", str(v))
        else:
            prompt = self.prompt_templates["not_found"].replace("__INPUT__", user_input)
            for k, v in self.state.items():
                if "operator" in v:
                    prompt = prompt.replace(
                        f"__{k.upper()}__", v["operator"] + str(v["value"])
                    )
                else:
                    prompt = prompt.replace(f"__{k.upper()}__", str(v))
        return prompt

    # 根据第一轮LLM的回答，把第一轮回答的第一个record, 作为第二轮LLM的输入;
    def _call_chatgpt(self, prompt, model="gpt-3.5-turbo"):
        session = copy.deepcopy(self.session)
        session.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model,
            messages=session,
            temperature=0,
        )
        # 有时会 ChatCompletion error?
        return response.choices[0].message.content

    def run(self, user_input):
        # 第一轮user对话LLM
        semantics = self.nlu.parse(user_input)
        print("semantics=")
        print(semantics)

        # 从第一轮对话LLM回答中，抽取出json结构信息，更新到state中;
        self.state = self.dst.update(self.state, semantics)
        print("self.state=")
        print(self.state)

        # 从MockedDB中检索出符合条件的套餐信息
        records = self.db.retrieve(**self.state)

        # 根据第一轮LLM的回答，把第一轮回答的第一个record, 作为第二轮LLM的输入;
        # 向_wrap传入的user_input, 是第一轮的user_input, 该input作为第二轮LLM的一部分, 见prompt_templates["requirement"];
        prompt_recommand = self._wrap(user_input, records)
        print("prompt_recommand=")
        print(prompt_recommand)

        response = self._call_chatgpt(prompt_recommand)
        # 把用户第一轮的输入，LLM的回答，加入session(这个session并没有在后续用到)
        self.session.append({"role": "user", "content": user_input})
        self.session.append({"role": "assistant", "content": response})
        return response
