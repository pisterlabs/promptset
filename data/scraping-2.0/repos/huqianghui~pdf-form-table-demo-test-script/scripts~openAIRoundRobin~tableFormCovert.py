from openAIRoundRobin.openAI_sc_us_03  import openaiscus03
from openAIRoundRobin.openAI_east_us_02 import openaieastus02
from openAIRoundRobin.openAI_east_us_03 import openaieastus03
from openAIRoundRobin.openAI_east_us_01 import openaieastus01
from openAIRoundRobin.openAI_sc_us_02 import openaiscus02
from openAIRoundRobin.openAI_sc_us_01 import openaiscus01


openai=[openaieastus01,openaieastus02,openaieastus03,openaiscus01,openaiscus02,openaiscus03]

openAICallCount = 0

def table_html_to_text_by_gpt4(htmltable):
    global openAICallCount
    global openai

    index = openAICallCount % 6
    currentOpenAI=openai[index]
    print("openai index:",index)
    messages=[{"role":"system","content":"You are an AI assistant that helps people find information.When Users input context in html table format，you extract all valid information from the html table and convert it into a human-readable format in a correct, complete and detailed manner. \nAnswer the content in a declarative tone, do not answer with enumeration or comprehension In Chinese.\nDon't talk about things beyond the content.\nDon't talk about things beyond the content.\nDon't talk about things beyond the content"}]
    messages.append({"role":"user","content":htmltable})

    try:
        response = currentOpenAI.ChatCompletion.create(
                        engine="gpt-4-32k",
                        messages = messages,
                        temperature=0,
                        max_tokens=4000,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None)
        result = response['choices'][0]['message']['content']
        print(result)
        openAICallCount = openAICallCount + 1
        return result
    except Exception as e :
        print("Call error:", e)
        index = (openAICallCount+1) % 6
        currentOpenAI=openai[index]
        print("after exception openai index:",index)
        response = currentOpenAI.ChatCompletion.create(
                        engine="gpt4",
                        messages = messages,
                        temperature=0,
                        max_tokens=4000,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None)
        result = response['choices'][0]['message']['content']
        openAICallCount = openAICallCount + 2
        print(result)
        return result
