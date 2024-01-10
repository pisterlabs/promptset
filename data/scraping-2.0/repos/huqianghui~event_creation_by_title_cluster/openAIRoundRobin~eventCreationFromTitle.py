# import os,sys
# sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from openAIRoundRobin.openAI_east_us_01 import gpt4Openaieastus01
from openAIRoundRobin.openAI_east_us_02 import gpt4Openaieastus02
from openAIRoundRobin.openAI_east_us_03 import gpt4Openaieastus03
from openAIRoundRobin.openAI_sc_us_01 import gpt4Openaiscus01
from openAIRoundRobin.openAI_sc_us_02 import gpt4Openaiscus02
from openAIRoundRobin.openAI_sc_us_03  import gpt4Openaiscus03

openai=[gpt4Openaieastus01,gpt4Openaieastus02,gpt4Openaieastus03,gpt4Openaiscus01,gpt4Openaiscus02,gpt4Openaiscus03]

from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate

openAICallCount = 0

def gpt4ProcessByTemplate(template_variabable_value:str,template:ChatPromptTemplate):
    global openAICallCount
    global openai

    index = openAICallCount % 6
    currentOpenAI=openai[index]
    print("openai index:",index)
    

    try:
        response = currentOpenAI(template.format_messages(source=template_variabable_value))
        result = response.content
        print(result)
        openAICallCount = openAICallCount + 1
        return result
    except Exception as e :
        print("Call error:", e)
        index = (openAICallCount+1) % 6
        currentOpenAI=openai[index]
        print("after exception openai index:",index)
        response = currentOpenAI(template.format_messages(source=template_variabable_value))
        result = response.content
        openAICallCount = openAICallCount + 2
        print(result)
        return result

#事件的标准 1.事件一般有一个明确的主体和明确的行为
#          2.观点、发言一般不作为事件，如果是围绕某一个事件进行的评论、观点，则应以该事件为事件名称。
#          5.语言精练简要。用最少的词说明事件。
#          6.如果有时间的话，不要出现明天，昨天，明年，下个月等这样不具体的时间，而是2023年8月3日这样的具体日期。  
eventCreationTemplate = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=('''
            You are a financial journalist who, based on the financial news headlines provided by the user, Summarize into a simple, clear, easily understandable Event which is fewer 15 words.
            Your generate the event content meets the following criteria:
                1. The event should have a clear subject and a clear action.
                2. Opinions and statements are generally not considered events. However, if comments or viewpoints revolve around a specific event, the event itself should be used as the event's name. For example,"Powell Discussed U.S. Interest Rate Hike" cannot be used as the event name, and instead, "U.S. Interest Rate Hike" should be used as the event subject.         
                3. Describe the event with minimal words and in concise and succinct language.   
                4. If possible, please avoid using vague terms like "tomorrow," "yesterday," "next year," "next month," etc., and instead use specific dates like August 3, 2023.
            
            Please provide a concise description after summarization, don't provide the input content.
            If after summarization, there are still many company names present, try categorizing them into industry types and use industries to replace these company names.
            And The final summarization should be less than 15 words.
            
            If the language of headlines is Chinses,please answer in Chinese. ''')
        ),
        HumanMessagePromptTemplate.from_template("The corresponding financial news headlines are as follows: {source}"),
    ]
)

def gpt4ProcessByEventCreationTemplate(source_titles:str):
    return gpt4ProcessByTemplate(source_titles,eventCreationTemplate)

# 对gpt产生的event内容进行二次过滤，过滤条件如下：
# 事件的标准 3. 事件需对市场情绪或行业个股股价有潜在或直接影响，单纯的盘面波动不能作为事件。 
eventFilterTemplate = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=('''
            You are a financial editor who based on the financial  event provided by the financial journalist, 
                     judge whether the event is likely to have a potential 
                     or direct impact on market sentiment 
                     or the stock prices of industry-specific companies.  
            If Pure market fluctuations cannot be considered as events. 
            
            For example:
                journalist: In 2023/08/02,the Hong Kong stock market experienced fluctuations, with multiple sectors showing activity.
                You: no.
                     
                journalist： 8月2日多家基金重仓股涨跌互现，涉及科技、医药、金融等行业。
                You: no.
                
                journalist: 2023/08/02，特斯拉总裁马斯克访问中国，中国总理李强与他会面。
                You: yes.
            ''')
        ),
        HumanMessagePromptTemplate.from_template("The corresponding financial news events is as follow: {source}"),
    ]
)

def gpt4ProcessByEventFilterTemplate(genereated_event:str):
    return gpt4ProcessByTemplate(genereated_event,eventFilterTemplate)