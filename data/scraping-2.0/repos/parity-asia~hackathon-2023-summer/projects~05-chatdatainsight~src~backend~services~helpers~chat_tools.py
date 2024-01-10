import openai

import sys
import os
import json


from langchain.llms import OpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

current_directory = os.path.dirname(os.path.realpath(__file__))
backend_directory = os.path.abspath(os.path.join(current_directory))
sys.path.insert(0, backend_directory)

# sys.path.insert(0, '/Users/qinjianquan/Career/redstone-network/chatdata-insight/backend')

from core.config import Config
openai.api_key = Config.OPENAI_API_KEY
MODEL_NAME = Config.MODEL_NAME
LLM = OpenAI(model_name=MODEL_NAME, temperature=0)  


def conversation(prompt):
     
     message = chat_with_gpt(prompt)

     return message

def chat_with_gpt(input):

    prom = """You are an omniscient artificial intelligence, please assist users in solving various problems"""

    conv = Conversation(prom, 20)
    return conv.ask(input) 

# upgrade
class Conversation:
    def __init__(self, prompt, num_of_round):
        self.prompt = prompt
        self.num_of_round = num_of_round
        self.messages = []
        self.messages.append({"role": "system", "content": self.prompt})

    def ask(self, question):
        try:
            self.messages.append({"role": "user", "content": question})
            response = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=self.messages,
                # stream=True,
                temperature=0.5,
                max_tokens=2048,  
                top_p=1,
            )
        except Exception as e:
            print(e)
            return e

        message = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": message})

        if len(self.messages) > self.num_of_round*2 + 1:
            del self.messages[1:21] # Remove the first round conversation left.
        return message



# prompt = "假设你是一个家庭服务聊天机器人，你需要回答一些关于家庭生活的日常问题"
# num_of_round = 5  # 对话轮数
# conversation = Conversation(prompt, num_of_round)

# question = "我烧伤了，应该怎么办？"
# print(conversation.ask(question))



from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
)

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain

from langchain.schema import (
    HumanMessage,
    SystemMessage
)


ANSWER_PROMPT = '''
Suppose you are an information organization expert. Your goal is to integrate and compile resources including text and image links, and then present them in a smooth, professional way to end users, and the content you output is directly facing users.

Please note that sometimes the source material may not contain image links. In this case, the project output only contains textual content. However, when image links are included in the source material, we will incorporate them into the final output.

Let's illustrate with an example:

Suppose we receive the following input:

'Over the past 24 hours, Bitcoin opened at 25647.69, reached a high of 25792.3, dropped to a low of 25571.37, and finally closed at 25727.3. This suggests that Bitcoin has been relatively stable in the past few hours. http://137.184.5.217:3005/static/image/chart.png'

The output will be:

'Over the past 24 hours, Bitcoin opened at 25647.69, reached a high of 25792.3, dropped to a low of 25571.37, and finally closed at 25727.3. This suggests that Bitcoin has been relatively stable in the past few hours. ![Bitcoin Price Chart](http://137.184.5.217:3005/static/image/chart.png) If you are considering making an investment, we recommend that you visit an exchange to view price trends over a longer period. If the current price is relatively low, the investment risk may be significantly lower compared to when the price is at a high point.'

Remember, your aim is to provide users with detailed, accurate and easy-to-understand information to assist them in making decisions."

'''



def stream_output(prompt):

    messages = [
    SystemMessage(content=ANSWER_PROMPT),
    HumanMessage(content=prompt)
    ]
         
    chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0,model_name=MODEL_NAME)
    return chat(messages)

    
# data_summary_prompt = '''
# Assume you are a professional data analyst. 
# You need to summarize the characteristics of the data in the dataframe, and provide investment advice. 
# The data given to you is generally about blockchain data, such as decentralized exchange data, stablecoin data, etc.
# '''

# def data_summary(data,image_link):
#     messages = [
#     SystemMessage(content=data_summary_prompt),
#     HumanMessage(content=data)
#     ]
         
#     chat = ChatOpenAI(streaming=False, callbacks=[StreamingStdOutCallbackHandler()], temperature=0,model_name=MODEL_NAME)
#     # chat = ChatOpenAI( callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
#     return chat(messages)

data = '''  
24 Hours Volume  7 Days Volume             Project  Rank
0      1.307926e+09   7.862556e+09             uniswap     1
1      4.187895e+08   2.377991e+09         pancakeswap     2
2      1.125999e+09   2.194102e+09               curve     3
3      9.258607e+07   7.502900e+08                DODO     4
4      5.469759e+07   3.594711e+08           quickswap     5
5      4.566395e+07   2.394654e+08            balancer     6
6      5.180647e+07   2.176834e+08            maverick     7
7      1.138057e+07   1.151459e+08           sushiswap     8
8      2.121930e+07   1.041825e+08               thena     9
9      9.867429e+06   8.626906e+07              biswap    10
10     1.004169e+07   7.176702e+07           velodrome    11
11     1.239834e+07   6.783979e+07              0x API    12
12     4.507492e+06   3.891650e+07             camelot    13
13     6.042391e+06   3.861647e+07            integral    14
14     2.303781e+06   2.960370e+07          trader_joe    15
15     2.902905e+06   2.801989e+07          spookyswap    16
16     1.442798e+07   2.720039e+07              wombat    17
17     7.029458e+06   2.180536e+07                 gmx    18
18     4.789188e+06   1.962993e+07             airswap    19
19     1.106220e+06   1.348747e+07            fraxswap    20
20     8.216870e+05   1.268815e+07           shibaswap    21
21     2.206660e+06   1.106626e+07         beethoven_x    22
22     1.328366e+06   8.037718e+06             clipper    23
23     5.244274e+05   6.939683e+06                mdex    24
24     7.025524e+05   6.209683e+06             apeswap    25
25     2.211005e+06   5.254385e+06    ellipsis_finance    26
26     5.153891e+05   5.198721e+06            hashflow    27
27     3.176448e+05   4.469678e+06      Bancor Network    28
28     7.903274e+05   3.051099e+06  spartacus_exchange    29
29     1.053042e+06   2.181084e+06  equalizer_exchange    30
30     3.101874e+04   2.033000e+06            defiswap    31
31     1.013178e+05   1.967260e+06            babyswap    32
32     1.585287e+05   1.512171e+06            wigoswap    33
33     1.061397e+06   1.386140e+06             mstable    34
34     1.109875e+05   1.196685e+06          spiritswap    35
35     4.073950e+05   9.873257e+05             arbswap    36
36     9.547153e+04   8.764512e+05           verse_dex    37
37     2.108004e+05   8.584624e+05           kyberswap    38
38     6.144034e+04   6.374561e+05             glacier    39
39     2.889735e+04   5.047607e+05           synthetix    40
40     5.128405e+04   2.978529e+05             iziswap    41
41     9.883904e+02   3.601762e+04             zipswap    42
42     6.387785e+03   1.842489e+04             rubicon    43
43     1.639312e+02   1.809792e+03            nomiswap    44
44     1.318094e+01   7.162410e+02               swapr    45

'''


    
DATA_SUMMARY_PROMPT = '''
Assume you are a professional data analyst. 
You've been given a string converted from a dictionary that contains three fields: "question", "data", and "image_link". 
The data, typically about blockchain information like decentralized exchange data or stablecoin data, is retrieved based on the specified "question". 
Your task is to analyze and summarize the characteristics of the data, provide investment advice based on your analysis, and restructure these insights into a new dictionary. 
The new dictionary should contain three fields: "problem", "answer", and "image_link". 
You can directly extract the contents for "problem" and "image_link" from the original string, while "answer" should be filled with the results of your data analysis and conclusions.
'''

def data_summary(x):
    response_schemas = [
        ResponseSchema(name="problem", description="problem is the question itself."),
        ResponseSchema(name="answer", description="summary of data"),
        ResponseSchema(name="image_link",description ="image link from data content")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template=DATA_SUMMARY_PROMPT+"\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    model = LLM 

    _input = prompt.format_prompt(question=x)
    output = model(_input.to_string())
    result = output_parser.parse(output)

    result_str = json.dumps(result)  # 将字典转换为 JSON 字符串

    print("INFO:     DATA ANALYZE RESULT:", result_str)
    # print("----------------- result type", type(result_str))

    return result_str
