
import openai

# To start an OpenAI-like Qwen server, use the following commands:
#   git clone https://github.com/QwenLM/Qwen-7B;
#   cd Qwen-7B;
#   pip install fastapi uvicorn openai pydantic sse_starlette;
#   python openai_api.py;
#
# Then configure the api_base and api_key in your client:
openai.api_base = "http://192.168.0.11:8081/v1"
openai.api_key = "none"

from config import  llm_model
def call_qwen(messages, functions=None):
    # print("-----",messages)
    if functions:
        response = openai.ChatCompletion.create(
            model=llm_model, messages=messages, functions=functions
        )
    else:
        response = openai.ChatCompletion.create(model=llm_model, messages=messages)
    # print(response)
    # print("*****",response.choices[0].message.content)
    return response


def test_1():
    messages = [{"role": "user", "content": "你好"}]
    call_qwen(messages)
    messages.append({"role": "assistant", "content": "你好！很高兴为你提供帮助。"})

    messages.append({"role": "user", "content": "给我讲一个年轻人奋斗创业最终取得成功的故事。故事只能有一句话。"})
    call_qwen(messages)
    messages.append(
        {
            "role": "assistant",
            "content": "故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。李明想要成为一名成功的企业家。……",
        }
    )

    messages.append({"role": "user", "content": "给这个故事起一个标题"})
    call_qwen(messages)


def test_2():
    functions = [
        {
            "name_for_human": "谷歌搜索",
            "name_for_model": "google_search",
            "description_for_model": "谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。"
            + " Format the arguments as a JSON object.",
            "parameters": [
                {
                    "name": "search_query",
                    "description": "搜索关键词或短语",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
        {
            "name_for_human": "文生图",
            "name_for_model": "image_gen",
            "description_for_model": "文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL。"
            + " Format the arguments as a JSON object.",
            "parameters": [
                {
                    "name": "prompt",
                    "description": "英文关键词，描述了希望图像具有什么内容",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
    ]

    messages = [{"role": "user", "content": "你好"}]
    call_qwen(messages, functions)
    messages.append(
        {"role": "assistant", "content": "你好！很高兴见到你。有什么我可以帮忙的吗？"},
    )
    messages.append({"role": "user", "content": "谁是周杰伦"})
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": "Thought: 我应该使用Google搜索查找相关信息。",
            "function_call": {
                "name": "google_search",
                "arguments": '{"search_query": "周杰伦"}',
            },
        }
    )

    messages.append(
        {
            "role": "function",
            "name": "google_search",
            "content": "Jay Chou is a Taiwanese singer.",
        }
    )
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": "周杰伦（Jay Chou）是一位来自台湾的歌手。",
        },
    )

    messages.append({"role": "user", "content": "他老婆是谁"})
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": "Thought: 我应该使用Google搜索查找相关信息。",
            "function_call": {
                "name": "google_search",
                "arguments": '{"search_query": "周杰伦 老婆"}',
            },
        }
    )

    messages.append(
        {"role": "function", "name": "google_search", "content": "Hannah Quinlivan"}
    )
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": "周杰伦的老婆是Hannah Quinlivan。",
        },
    )

    messages.append({"role": "user", "content": "给我画个可爱的小猫吧，最好是黑猫"})
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": "Thought: 我应该使用文生图API来生成一张可爱的小猫图片。",
            "function_call": {
                "name": "image_gen",
                "arguments": '{"prompt": "cute black cat"}',
            },
        }
    )

    messages.append(
        {
            "role": "function",
            "name": "image_gen",
            "content": '{"image_url": "https://image.pollinations.ai/prompt/cute%20black%20cat"}',
        }
    )
    call_qwen(messages, functions)


def test_3():
    functions = [{
            "name_for_human": "谷歌搜索",
            "name_for_model": "google_search",
            "description_for_model": "谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。"
            + " Format the arguments as a JSON object.",
            "parameters": [
                {
                    "name": "search_query",
                    "description": "搜索关键词或短语",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
        {
            "name_for_human": "文生图",
            "name_for_model": "image_gen",
            "description_for_model": "文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL。"
            + " Format the arguments as a JSON object.",
            "parameters": [
                {
                    "name": "prompt",
                    "description": "英文关键词，描述了希望图像具有什么内容",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
        {
            "name": "get_order",
            "description": "根据订单ID查询订单详情",
            "parameters": {
                "type": "object",
                "properties": {
                    "orderNo": {
                        "type": "string",
                        "description": "订单ID,如果为空，使用null作为默认值",
                    }
                },
                "required": ["orderNo"],
            },
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "我想查询订单4567896554详情",
            # "content": "波士顿天气如何？",
        }
    ]


    res=call_qwen(messages, functions)
    # print(res.choices[0].message.content)
    # print(res.choices[0].message.function_call)

    if res.choices[0].message.function_call:
        inp={
                "role": "assistant",
                "content": None,
                "function_call": res.choices[0].message.function_call,
            }
        messages.append(
            inp
        )
        if res.choices[0].message.function_call["name"]=="get_order":
            # if res.choices[0].message.function_call["orderNo"]==
            messages.append(
                {
                    "role": "function",
                    # "name": "get_order",
                    "content": '{"订单时间": "2023-11-27", "数量": "1", "金额":"178","配送状态": "正在配送之中","商品名称":"黑森林蛋糕","配送城市":"深圳南山"}',
                    # "content": '检测不到用户的订单ID，需要用户回复订单ID',
                }
            )
    res=call_qwen(messages,[] )
    print(res.choices[0].message.content)
    print(res.choices[0].message.function_call)


def test_4():
    from langchain.chat_models import ChatOpenAI
    from langchain.agents import load_tools, initialize_agent, AgentType

    llm = ChatOpenAI(
        model_name=llm_model,
        openai_api_base=openai.api_base ,
        openai_api_key="EMPTY",
        streaming=False,
    )
    tools = load_tools(
        ["arxiv"],
    )
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    # TODO: The performance is okay with Chinese prompts, but not so good when it comes to English.
    agent_chain.run("查一下论文 1605.08386 的信息")


if __name__ == "__main__":
    # print("### Test Case 1 - No Function Calling (普通问答、无函数调用) ###")
    # test_1()
    # print("### Test Case 2 - Use Qwen-Style Functions (函数调用，千问格式) ###")
    # test_2()
    # print("### Test Case 3 - Use GPT-Style Functions (函数调用，GPT格式) ###")
    test_3()
    # print("### Test Case 4 - Use LangChain (接入Langchain) ###")
    # test_4()
    print("end")
