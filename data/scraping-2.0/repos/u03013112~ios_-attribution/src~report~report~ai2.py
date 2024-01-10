#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai

import sys
sys.path.append('/src')

from src.config import openaiApiKey,openaiUrl,openaiApiKey2,openaiUrl2

openai.api_type = "azure"
openai.base_url = openaiUrl
openai.api_version = "2023-07-01-preview"
openai.api_key = openaiApiKey

s1_1 = '''
你是一个数据分析师，主要职责是帮助广告投放人员进行数据分析，协助他们进行广告投放。
广告投放人员的KPI是在ROI合格的情况下尽可能的多花钱。
KPI的ROI指的是7日ROI。并且是分国家进行制定的。
由于7日ROI需要广告投放的第8日才能获得完整数据，所以只能粗略的按照比例估算ROI1、ROI3和ROI24h的KPI，由于这是估算的，所以只能认为是风险，比如3日ROI远小于3日ROI的估算KPI，只能认为是个大风险，适当的给出提示即可。
广告投放的调整分为很多方面，你主要需要帮助他们在不同国家之间的花费调整进行建议。
重点信息，请一定注意：一般的，增加花费会导致ROI下降，减少花费会导致ROI上升。所以在ROI不达标时要降低花费，而ROI达标时要增加花费。
重点信息2：GCC是一个特殊地区，人口较少，但是付费能力很强，他的ROI大幅波动是比较正常的情况。

给出的数据一般的都有 target,group,20231101~20231110,20231022~20231031,环比,KPI 这些列
其中target：ROI24h 24小时ROI，ROI1 1自然日ROI，ROI3、ROI7类似，Cost 花费（美元）
group：这里是分国家分组，这个分组是按照我们KPI的分组
20231101~20231110,20231022~20231031：这两个列的列名不固定，一般的是最近关注的一个周期和上一个周期，这里的周期是指自然日
环比：环比差异，百分比，分析的时候主要针对环比变化大的进行分析。注意，环比是上一个周期与本周期的环比，不是与KPI的比较，注意，不要搞混。
KPI：这里的KPI就是我们要求的KPI，根据target不同，ROI7是实际KPI，ROI24h，ROI1，ROI3是估算KPI，Cost 没有KPI

我需要的分析结果要求：
1、尽量的简洁，突出重点，主要关注环比差距在20%以上内容，以及与KPI差距在20%以上的数据。其他的数据可以用类似整体表现很好之类的措辞来概括。每个部分不要逐行分析，那样还不如直接看数据。
2、所有的点评与措辞都要保守，尽量不要使用错误、问题、不合理等词汇，可以使用 “有风险” “大概率” 等词汇。
3、注意列名，上面例子中20231101~20231110,20231022~20231031的20231022~20231031是上一个周期，20231101~20231110是本周期，这个是不固定的，所以要注意。
4、再强调一次，尽量让投放人员尽快的看到重点，其他数据他们会自己看详细数据。整体不要超过200字。
格式模板，只是格式与措辞的范例，不是数据的范例，请根据具体数据进行分析，不要照抄下面范例内容：
1、上周期摘要：总体表现(好or不好or一般)，特别是XX(上周期ROI7 XX%,比KPI要求的XX% 高or低了XX%)高于or低于KPI比例较大，可以考虑适当放量or减量。
    注意：只关注ROI7列，上个周期的ROI7是完整数据。只关注高于or低于KPI比例较大的数据，比例较小的数据不要出现在分析里！
2、上周期到本周期操作点评：主要花费变化在XX（如果有多个就放一起写）地区花费大幅增加or减少，符合（不符合）一般规律。
    注意：符合或者不符合的一般规律，指的是上周ROI7比KPI高or低，本周花费增加or减少，这是一般规律。如果上周ROI7比KPI低，本周花费增加，那就是不符合一般规律。
3、本周期目前情况与建议：ROI1和ROI3方面，与折算KPI比较，基本满足or不满足KPI的，其中XX（如果有多个就放一起写）在本周表现很好（或者不好），可以适度放量（减量）。
    注意：本周期的ROI7是不完整数据。所以主要关注ROI1、ROI3、ROI24h。
    注意：如果总体上与折算KPI比较都好或者不好，没有任何一个国家满足KPI或折算KPI，那就不要再有后面 “其中XX” 及以后的内容了。用建议性的措辞来概括即可。比如建议适度整体放量或者减量。

下面我会给你一些数据，这条你了解了就回复我“准备好了”就好了。
'''

s2_1 = '''
读下面csv格式表格，并对数据进行分析
target,group,20231101~20231110,20231022~20231031,环比,KPI
ROI24h,US,2.04%,2.11%,-3.56%,1.94%
ROI24h,JP,2.20%,1.77%,23.69%,1.40%
ROI24h,GCC,2.42%,30.84%,-92.14%,1.35%
ROI24h,KR,2.43%,2.34%,3.73%,1.69%
ROI24h,other,2.85%,2.51%,13.68%,2.02%
ROI1,US,1.34%,1.29%,3.86%,1.21%
ROI1,JP,1.74%,1.26%,38.37%,1.05%
ROI1,GCC,1.54%,22.48%,-93.17%,0.98%
ROI1,KR,1.74%,1.78%,-2.08%,1.19%
ROI1,other,2.01%,1.63%,23.42%,1.39%
ROI3,US,3.40%,4.07%,-16.47%,3.45%
ROI3,JP,3.94%,3.65%,7.88%,2.46%
ROI3,GCC,3.73%,70.54%,-94.71%,2.99%
ROI3,KR,3.65%,5.52%,-33.84%,3.42%
ROI3,other,5.23%,4.86%,7.77%,3.75%
ROI7,US,数据不完整,8.22%,数据不完整,6.50%
ROI7,JP,数据不完整,9.29%,数据不完整,5.50%
ROI7,GCC,数据不完整,143.89%,数据不完整,6.00%
ROI7,KR,数据不完整,10.05%,数据不完整,6.50%
ROI7,other,数据不完整,9.80%,数据不完整,7.00%
Cost,US,374197.88,320993.77,16.57%,
Cost,JP,85258.02,71728.82,18.86%,
Cost,GCC,33181.08,8881.30,273.61%,
Cost,KR,188114.26,163887.78,14.78%,
Cost,other,746070.51,763645.26,-2.30%,
'''

message_text1 = [
    {"role":"system","content":"You are an AI assistant that helps people find information."},
    {"role":"user","content":s1_1},
    {"role":"assistant","content":"准备好了"},
    {"role":"user","content":s2_1},
]

s1_2 = '''
你是一个数据分析师，主要职责是帮助广告投放人员进行数据分析，协助他们进行广告投放。
由于7日ROI需要广告投放的第8日才能获得完整数据，所以只能粗略的按照比例估算ROI1、ROI3和ROI24h的KPI，由于这是估算的，所以只能认为是风险，比如3日ROI远小于3日ROI的估算KPI，只能认为是个大风险，适当的给出提示即可。
广告投放的调整分为很多方面，你主要需要帮助他们分析不同媒体之间的花费与ROI的变动情况。
在分媒体的数据中没有KPI，主要的分析方式是媒体间的横向对比。

给出的数据一般的都有 target,group,20231101~20231110,20231022~20231031,环比 这些列
其中target：ROI24h 24小时ROI，ROI1 1自然日ROI，ROI3、ROI7类似，Cost 花费（美元），Cost rate 花费占比，
revenue 1day、revenue 1day rate 是首日回收金额与首日回收金额占比，这两个指标里除了包含媒体外还包含了自然量，注意分析自然量的变化，这对大盘分析很重要。
CPM、CTR、CVR、CPI都是广告转化相关数据，其中字节媒体不能有效获得安装数，所以没有与安装有关的数据，与安装有关的数据请忽略字节。这几个广告转化相关数据，主要关注CPI，在CPI出现异常再分析其他数据。
group：这里是媒体分组，有些指标是没有自然量的，比如广告花费。
20231101~20231110,20231022~20231031：这两个列的列名不固定，一般的是最近关注的一个周期和上一个周期，这里的周期是指自然日
环比：环比差异，百分比，分析的时候主要针对环比变化大的进行分析。

我需要的分析结果要求：
1、尽量的简洁，突出重点，主要关注环比差距在20%以上内容。环比低于20%的数据可以用类似整体表现很好之类的措辞来概括。每个部分不要逐行分析，那样还不如直接看数据。
2、所有的点评与措辞都要保守，尽量不要使用错误、问题、不合理等词汇，可以使用 “有风险” “大概率” 等词汇。
3、注意列名，上面例子中20231101~20231110,20231022~20231031的20231022~20231031是上一个周期，20231101~20231110是本周期，这个是不固定的，所以要注意。
4、再强调一次，尽量让投放人员尽快的看到重点，其他数据他们会自己看详细数据。整体不要超过50字。
格式模板，只是格式与措辞的范例，不是数据的范例。范例中的注意都是提醒你注意的点，不是报告内容：
1、上周期摘要：Google的ROI7比较低，只有5%。自然量收入大幅（40%）提高。
    注意：只关注ROI7这一组ROI相关数据，其他ROI列比如ROI1在上周期摘要环节不要提！这很重要！
    注意：收入比例中自然量也比较敏感，要是环比变化大，要添加到上周期摘要的最后部分。
2、上周期到本周期操作点评：本周期Facebook有较大幅度的花费增加，Google和字节都是较大幅度的减少花费。
    注意：所谓操作也主要针对在不同媒体的花费调整，调整之后ROI是否有大幅变动。
    注意：只关注环比差距在20%以上内容，环比低于20%的数据不要出现在分析里！这对我非常重要！
下面我会给你一些数据，这条你了解了就回复我“准备好了”就好了。
'''

s2_2 = '''
读下面csv格式表格，并对数据进行分析
target,group,20231101~20231110,20231022~20231031,环比
ROI24h,bytedanceglobal,1.87%,1.28%,45.97%
ROI24h,google,1.67%,1.44%,15.90%
ROI24h,facebook,1.88%,2.20%,-14.64%
ROI1,bytedanceglobal,1.44%,0.99%,45.70%
ROI1,google,1.01%,1.00%,1.65%
ROI1,facebook,1.36%,1.36%,0.08%
ROI3,other,0.00%,0.01%,-98.20%
ROI3,bytedanceglobal,3.41%,3.79%,-9.96%
ROI3,google,3.19%,3.03%,5.47%
ROI3,facebook,3.24%,3.93%,-17.59%
ROI7,other,数据不完整,0.04%,数据不完整
ROI7,bytedanceglobal,数据不完整,6.74%,数据不完整
ROI7,google,数据不完整,5.86%,数据不完整
ROI7,facebook,数据不完整,8.50%,数据不完整
Cost,other,15128.35,14956.75,1.15%
Cost,bytedanceglobal,164113.99,252663.21,-35.05%
Cost,google,330869.00,304427.00,8.69%
Cost,facebook,916710.41,757089.97,21.08%
Cost rate,other,1.06%,1.13%,-5.78%
Cost rate,bytedanceglobal,11.50%,19.01%,-39.49%
Cost rate,google,23.19%,22.90%,1.24%
Cost rate,facebook,64.25%,56.96%,12.79%
revenue 1day,other,0.00,0.28,-100.00%
revenue 1day,bytedanceglobal,2363.98,2497.88,-5.36%
revenue 1day,google,3354.37,3036.18,10.48%
revenue 1day,facebook,12447.36,10271.97,21.18%
revenue 1day,organic,7095.71,6569.76,8.01%
revenue 1day rate,bytedanceglobal,9.36%,11.16%,-16.17%
revenue 1day rate,google,13.28%,13.57%,-2.14%
revenue 1day rate,facebook,49.27%,45.91%,7.34%
revenue 1day rate,organic,28.09%,29.36%,-4.33%
CPM,other,416.55,565.47,-26.34%
CPM,bytedanceglobal,2.76,3.10,-10.93%
CPM,google,3.71,3.08,20.43%
CPM,facebook,7.23,6.62,9.32%
CTR,other,19.52%,22.20%,-12.10%
CTR,bytedanceglobal,0.78%,0.59%,31.86%
CTR,google,0.22%,0.24%,-7.81%
CTR,facebook,0.71%,0.63%,13.71%
CVR,other,57.93%,63.77%,-9.15%
CVR,google,12.25%,7.35%,66.63%
CVR,facebook,10.56%,11.70%,-9.71%
CPI,other,3.68,3.99,-7.75%
CPI,google,13.53,17.25,-21.60%
CPI,facebook,9.60,9.02,6.48%
'''

message_text2 = [
    {"role":"system","content":"You are an AI assistant that helps people find information."},
    {"role":"user","content":s1_2},
    {"role":"assistant","content":"准备好了"},
    {"role":"user","content":s2_2},
]

s3_2 = '''
target,group,20231101~20231110,20231022~20231031,环比,KPI
ROI24h,US,1.70%,1.12%,51.70%,1.67%
ROI24h,GCC,0.43%,2.31%,-81.17%,0.90%
ROI24h,JP,2.95%,3.03%,-2.69%,1.05%
ROI24h,KR,1.39%,2.12%,-34.51%,1.18%
ROI24h,other,2.17%,1.12%,94.29%,1.75%
ROI1,US,1.10%,0.68%,61.88%,1.04%
ROI1,GCC,0.31%,1.75%,-82.40%,0.67%
ROI1,JP,2.35%,2.51%,-6.26%,0.84%
ROI1,KR,0.76%,1.94%,-60.94%,0.91%
ROI1,other,1.93%,0.86%,123.00%,1.39%
ROI3,US,2.96%,2.39%,24.23%,3.19%
ROI3,GCC,1.39%,7.24%,-80.85%,2.98%
ROI3,JP,4.92%,7.40%,-33.52%,2.05%
ROI3,KR,1.99%,9.35%,-78.71%,3.97%
ROI3,other,4.28%,2.99%,43.26%,4.17%
ROI7,US,数据不完整,5.28%,数据不完整,6.50%
ROI7,GCC,数据不完整,14.09%,数据不完整,6.00%
ROI7,JP,数据不完整,22.86%,数据不完整,5.50%
ROI7,KR,数据不完整,15.17%,数据不完整,6.50%
ROI7,other,数据不完整,5.06%,数据不完整,7.00%
Cost,US,46248.23,62323.99,-25.79%,
Cost,GCC,1445.98,720.13,100.79%,
Cost,JP,2738.41,2585.37,5.92%,
Cost,KR,34574.05,35458.78,-2.50%,
Cost,other,79107.32,151574.94,-47.81%,
Cost rate,US,28.18%,24.67%,14.24%,
Cost rate,GCC,0.88%,0.29%,209.13%,
Cost rate,JP,1.67%,1.02%,63.07%,
Cost rate,KR,21.07%,14.03%,50.11%,
Cost rate,other,48.20%,59.99%,-19.65%,
'''

message_text3 = [
    {"role":"system","content":"You are an AI assistant that helps people find information."},
    {"role":"user","content":s1_1},
    {"role":"assistant","content":"准备好了"},
    {"role":"user","content":s3_2},
]

# completion = openai.chat.completions.create(
#     model="bigpt4",
#     messages=message_text3,
#     temperature=0.1,
#     top_p=0.95,
#     frequency_penalty=0,
#     presence_penalty=0,
#     stop=None
# )
# print(completion.choices[0].message.content)

def getAiResp(message_text):
    completion = openai.chat.completions.create(
        # model="bigpt4",
        model="gpt4-202311",
        messages=message_text,
        temperature=0.0,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return(completion.choices[0].message.content)

def getAiReport(reportPath):
    retStr = '\n\n## 分国家\n'
    report1 = os.path.join(reportPath,'report1_1.csv')
    # 读取report1_1.csv，存到字符串s1中
    s1 = '读下面csv格式表格，并对数据进行分析\n'
    with open(report1, 'r', encoding='utf-8') as f:
        s1 += f.read()
    
    message_text1 = [
        {"role":"system","content":"You are an AI assistant that helps people find information."},
        {"role":"user","content":s1_1},
        {"role":"assistant","content":"准备好了"},
        {"role":"user","content":s1},
    ]

    report1_ai = os.path.join(reportPath,'report1_1_ai.txt')
    ret1 = ''
    if os.path.exists(report1_ai):
        with open(report1_ai, 'r', encoding='utf-8') as f:
            ret1 = f.read()
    else:
        ret1 = getAiResp(message_text1)
        with open(report1_ai, 'w', encoding='utf-8') as f:
            f.write(ret1)

    retStr += ret1

    retStr += '\n\n## 分媒体\n'
    report2 = os.path.join(reportPath,'report2_1.csv')
    # 读取report2_1.csv，存到字符串s2中
    s2 = '读下面csv格式表格，并对数据进行分析\n'
    with open(report2, 'r', encoding='utf-8') as f:
        s2 += f.read()

    message_text2 = [
        {"role":"system","content":"You are an AI assistant that helps people find information."},
        {"role":"user","content":s1_2},
        {"role":"assistant","content":"准备好了"},
        {"role":"user","content":s2},
    ]
    report2_ai = os.path.join(reportPath,'report2_1_ai.txt')
    ret2 = ''
    if os.path.exists(report2_ai):
        with open(report2_ai, 'r', encoding='utf-8') as f:
            ret2 = f.read()
    else:
        ret2 = getAiResp(message_text2)
        print(message_text2)
        print(ret2)
        with open(report2_ai, 'w', encoding='utf-8') as f:
            f.write(ret2)

    retStr += ret2

    for media in ['bytedanceglobal','facebook','google']:
        retStr += '\n\n## 分媒体细节v1 '+media+'\n'
        reportMedia = os.path.join(reportPath,'report3_1_'+media+'.csv')
        # 读取report3_1_bytedanceglobal.csv，存到字符串s3中
        s3 = '读下面csv格式表格，并对数据进行分析\n'
        with open(reportMedia, 'r', encoding='utf-8') as f:
            s3 += f.read()

        message_text3 = [
            {"role":"system","content":"You are an AI assistant that helps people find information."},
            {"role":"user","content":s1_1},
            {"role":"assistant","content":"准备好了"},
            {"role":"user","content":s3},
        ]

        report3_ai = os.path.join(reportPath,'report3_1_'+media+'_ai.txt')
        ret3 = ''
        if os.path.exists(report3_ai):
            with open(report3_ai, 'r', encoding='utf-8') as f:
                ret3 = f.read()
        else:
            ret3 = getAiResp(message_text3)
            with open(report3_ai, 'w', encoding='utf-8') as f:
                f.write(ret3)

        retStr += ret3

    return retStr

import time
from src.report.feishu.report1 import main as feishuMain
from src.report.feishu.feishu import sendMessageDebug
if __name__ == '__main__':
    reportPath = '/src/data/report/海外iOS速读AI版_20231127_20231203'
    
    # for media in ['bytedanceglobal','facebook','google']:
    for media in ['bytedanceglobal']:
    
        reportMedia = os.path.join(reportPath,'report3_1_'+media+'.csv')
        # 读取report3_1_bytedanceglobal.csv，存到字符串s3中
        s3 = '读下面csv格式表格，并对数据进行分析\n'
        with open(reportMedia, 'r', encoding='utf-8') as f:
            s3 += f.read()

        message_text3 = [
            {"role":"system","content":"You are an AI assistant that helps people find information."},
            {"role":"user","content":s1_1},
            {"role":"assistant","content":"准备好了"},
            {"role":"user","content":'再开始之前，再次确认一次，请注意我在模板中提到的注意事项，不要忽略任何一条注意事项。'},
            {"role":"user","content":s3},
        ]

        ret3 = getAiResp(message_text3)
        print(ret3)        
