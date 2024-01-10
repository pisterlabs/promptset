import os
import time
import json
import copy
import openai
import pandas as pd
from retry import retry
from .llm_remote import get_output
from .llm_remote import get_ws_stream_content
from .llm_remote import get_ws_content
from plugins.common import settings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chat_init(history):
    return history

class DelataMessage:
    def __init__(self):
        self.content=''
    def __getitem__(self, item):
        return getattr(self, item)
def get_web_data(solution_prompt,zhishiku):
    #import ipdb
    #ipdb.set_trace()
    solution_data = zhishiku.zsk[1]['zsk'].find(solution_prompt)
    if not solution_data:
        solution_data = zhishiku.zsk[0]['zsk'].find(solution_prompt)
        if len(solution_data) > 0:
            zhishiku.zsk[1]['zsk'].save(solution_prompt,solution_prompt,solution_data,'','')
            print('save {} mysql successfully'.format(solution_prompt))
    if solution_data:
        return solution_data
    return ''
def get_solution_data(current_plans,zhishiku,chanyeku):
    solution_data = ''
    is_break = False
    for current_plan in current_plans:
        for current in current_plan:
            solution_type,solution_exec = current.split(':')
            solution_exec = solution_exec.strip()
            if '数据库' in solution_type:
                #solution_prompt = "你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，针对以下问题，生成相应的sql指令:\n" + solution_exec
                solution_prompt = "你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题>解决专家，请解决以下问题:\n" + solution_exec
                solution_prompt = solution_prompt.strip()
                #solution_output = get_output(solution_prompt)
                solution_output = get_ws_content(solution_prompt)
                #solution_output = "select `企业名称`,`企业类型`,`产业` from `企业数据` where  城市 like '%景德%' limit 10;"
                #import ipdb
                #ipdb.set_trace()
                print(solution_exec + ':' + solution_output)
                solution_data = zhishiku.zsk[1]['zsk'].find_by_sql(solution_output)
                if solution_data:
                    if len(solution_data) == 1 and ('0' in str(solution_data) or 'None' in str(solution_data)):
                        solution_data = ''
                        continue
                    is_break = True
                    break
            if '使用工具' in solution_type:
                li = solution_exec.split('\t')
                fun,paramater = li[0],li[1:]
                #paramater = [str(x)  for x in paramater ]
                if fun == 'python eval':
                    paramater = paramater[0]
                    solution_data = eval(paramater)
                    break
                else:
                    paramater = ",".join([f"'{x}'" if isinstance(x,str) else str(x) for x in paramater])
                #fun,paramater = solution_exec.split('\t')
                solution_exec = fun + '(' + f'{paramater}' + ')'
                solution_data = chanyeku.chanye(solution_exec)
                if solution_data:
                    is_break = True
            if '搜索引擎' in solution_type:
                solution_prompt = solution_exec
                #solution_data = zhishiku.zsk[1]['zsk'].find(solution_prompt)
                #import ipdb
                #ipdb.set_trace()
                if not solution_data:
                    solution_data = zhishiku.zsk[2]['zsk'].find(solution_prompt)
                    #if len(solution_data) > 0:
                    #    zhishiku.zsk[1]['zsk'].save(solution_prompt,solution_prompt,solution_data,'','')
                    #    print('save {} mysql successfully'.format(solution_prompt))
                if not solution_data:
                    solution_data = zhishiku.zsk[0]['zsk'].find(solution_prompt)
                    #if len(solution_data) > 0:
                    #    zhishiku.zsk[1]['zsk'].save(solution_prompt,solution_prompt,solution_data,'','')
                    #    print('save {} mysql successfully'.format(solution_prompt))
                if solution_data:

                    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=25,separators=["\n\n", "\n","。","\r","\u3000"])
                    is_break = True
                    break
        if is_break:
            break
    return solution_data
def build_question_and_context(question,data:pd.DataFrame,format_str='markdown'):
    question_context = ''
    if isinstance(data,list):
        data = pd.DataFrame(data)
    if  data.empty:
        return question
    if format_str == 'markdown':
        question_context = data.to_markdown() + '\n' + question
    return question_context
def get_answer_with_context(prompt,context_data,history_data):
    #solution_prompt = context_data + '\n\n,从上述文本中，精确的选取有用的部分，回答下面的问题\n' + prompt
    solution_prompt = context_data + ' ' + prompt
    #solution_prompt = context_data + '\n' + '上述文本是和问题相关的文本，请精确的回答下述问题,回答内容中不要出现"根据文本提供的内容"等类似字样:\n' + prompt
    solution_prompt = solution_prompt.strip()
    if history_data:
        history_data.append({"role": "user", "content": solution_prompt})
        solution_prompt = history_data 

    answer = get_ws_stream_content(solution_prompt)
    #answer = get_ws_stream_content(history_data)
    return answer

def generate_answer(solution_data,prompt,current_plan,history_data,zhishiku):
    """
    "{'获取数据': ['从企业数据库中获取数据:获取位于景德镇珠山的企业,>给出企业名称，企业类型,产业', '查询搜索引擎:烟台企业'], '生成答案': [\"'从上述>列表中，选出企业列表'\"], '评价答案': []}"
    {'获取数据': [['从企业数据库中获取数据:北京专精特新企业的企业名称，产业，企业类型'], ['查询搜索引擎:北京专精特新企业']], '生成答案': ['获取答案的前缀', '将答案和前缀进行组合输出'], '评价答案': []}
    """
    if isinstance(solution_data,list):
        solution_data = pd.DataFrame(solution_data)
    prefix = None
    answer = None
    for current in current_plan:
        if '从上述列表中' in current:
        #if 'query_llm' in current:
            context_question = build_question_and_context(prompt,solution_data)
            context_question = context_question.strip()
            solution_prompt = '你的名字叫小星，一个产业算法智能助手，由合享智星算法团>队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，请结合给定的数据,解决以下问题:\n' + context_question
            #answer = get_output(solution_prompt)
            answer = get_ws_content(solution_prompt)
            if not answer:
                raise ValueError('没有获得答案，抛出异常，让生成式模型来获取答案')
        if '获取答案的前缀' in current:
        #if '前缀' in current:
            #solution_prompt = '你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题>解决专家，针对以下问题，生成相应的回答前缀:\n' + prompt
            idx = prompt.find('企业')
            new_prompt = prompt[0:idx+2]
            solution_prompt = '你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题>解决专家，请结合给定的数据,解决以下问题:\n' + new_prompt
            #prefix = get_output(solution_prompt)
            prefix = get_ws_content(solution_prompt)
            print('prefix:' + prefix)
            #import ipdb
            #ipdb.set_trace()
            if not prefix:
                raise ValueError('没有获得答案，抛出异常，让生成式模型来获取答案')
        if '直接作为答案输出' in current:
            if solution_data:
                answer = str(solution_data)
            else:
                answer = get_ws_stream_content(history_data)
        if '将答案和前缀进行组合输出' in current:
            if not prefix:
                 raise ValueError('没有获得答案，抛出异常，让生成式模型来获取答案')
            else:
                if isinstance(solution_data,pd.DataFrame):
                    if len(solution_data) == 1 and len(solution_data.columns) == 1: 
                        answer = prefix + solution_data.to_string(index=False,header=False)
                    else:
                        answer = prefix + ':\n' + solution_data.to_string(index=False,header=False)
                else:
                    if isinstance(solution_data,str):
                        #answer = get_answer_with_context(prompt,solution_data,history_data)
                        answer = get_answer_with_context(prompt,solution_data,[])
        if '直接将问题送给模型' in current:
            history_data.append({"role": "user", "content": prompt})
            answer = get_ws_stream_content(history_data)
        if '将答案和模型进行结合' in current or '将答案与模型进行结合' in current:
            #if isinstance(solution_data,str) and len(solution_data) > 200:
                #answer = get_answer_with_context(prompt,solution_data,history_data)
            answer = get_answer_with_context(prompt,solution_data,[])
            #else:
            #    solution_data = get_web_data(prompt,zhishiku)
            #    #answer = get_answer_with_context(prompt,solution_data,history_data)
            #    answer = get_answer_with_context(prompt,solution_data,[])
    #else:
    #else:
    #    for token in answer.split('\n'):
    #        current_content += token + '\n'
    #        time.sleep(0.005)
    #        yield current_content
    return answer
# delay 表示延迟1s再试；backoff表示每延迟一次，增加2s，max_delay表示增加到120s就不再增加； tries=3表示最多试3次
@retry(delay=8, backoff=4, max_delay=22,tries=2)
def completion_with_backoff(**kwargs):
    try:
        return  openai.ChatCompletion.create(**kwargs)
    except Exception as e:
        #import ipdb
        #ipdb.set_trace()
        print(e)

        #import ipdb
        #ipdb.set_trace()
        dm = DelataMessage()
        if "maximum context length is 8192 tokens" in str(e):
            print('maximum exceed,deal ok')
            content= '历史记录过多，超过规定长度，请清空历史记录'
            setattr(dm,'content',content)
            chunk=[{'choices':[{'finish_reason':'continue','delta':dm,'content':content}]}]
            return chunk
        if "Name or service not known" in str(e):
            print('域名设置有问题，请排查服务器域名')
            content = '域名设置有问题，请排查服务器域名'
            setattr(dm,'content',content)
            #chunk=[{'choices':[{'finish_reason':'continue','delta':{'content':dm}}]}]
            chunk=[{'choices':[{'finish_reason':'continue','delta':dm,'content':content}]}]
            return chunk
        raise e  
def transform_openai2llama2(history_formatted):
    FIRST_PROMPT_TEMPLATE = (
            "<s>[INST] <<SYS>>\n"
            "You are a helpful assistant. 你是一个乐于助人的助手。\n"
            "<</SYS>>\n\n{instruction} [/INST] {response} </s>"
        )
    SECOND_PROMPT_TEMPLATE = (
            "<s>[INST] {instruction} [/INST] {response} </s>"
        )
    res = []
    prompt = ''
    is_editored = False
    if history_formatted is not None:
        for i, old_chat in enumerate(history_formatted):
            #if old_chat['role'] == "system":
            #    sys_prompt = (
            #" <<SYS>>\n"
            #"You are a helpful assistant. 你是一个乐于助人的助手。\n"
            #"<</SYS>>\n\n"
            #    )
            #    res.append(sys_prompt)
            if i%2 == 0:
                if i/2 == 0:
                    prompt = FIRST_PROMPT_TEMPLATE
                else:
                    prompt = SECOND_PROMPT_TEMPLATE
                is_editored = False

            if old_chat['role'] == "user":
                #history_data.append(
                #    {"role": "user", "content": old_chat['content']},)
                #user_prompt = (
                prompt=prompt.replace('{instruction}',old_chat['content'])
                is_editored = True
            elif old_chat['role'] == "AI" or old_chat['role'] == 'assistant':
                #history_data.append(
                #    {"role": "assistant", "content": old_chat['content']},)
                #prompt.format({'response':response})
                prompt=prompt.replace('{response}',old_chat['content'])
                is_editored = True
            if i%2 == 1 and is_editored:
                res.append(prompt)
            if i == len(history_formatted) - 1 and is_editored:
                prompt=prompt.replace('{response}','')
                res.append(prompt)
    return ''.join(res)
#def chat_one(prompt, history_formatted, max_length, top_p, temperature, data):
def chat_one(prompt, history_formatted, max_length, top_p, temperature, web_receive_data,zhishiku=False,chanyeku=False):
    history_data = [ {"role": "system", "content": "You are a helpful assistant. 你是一个乐于助人的助手。\n"}]
    #history_data = [ {"role": "system", "content": "You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。\n"}]
   
        
    ##history_data = []
    #daici = ['以上','这','那','上述','继续','再']
    #is_multi_turn = False
    #for dai in daici:
    #    if dai in prompt:
    #        is_multi_turn = True
    #if not is_multi_turn:
    #    history_formatted = []

    #import ipdb
    #ipdb.set_trace()
    history_formatted = history_formatted[::-1]
    #history_formatted = history_formatted[-5:]
    if history_formatted is not None:
        for i, old_chat in enumerate(history_formatted):
            if 'role' in old_chat:
                if old_chat['role'] == "user":
                    history_data.append(
                        {"role": "user", "content": old_chat['content']},)
                elif old_chat['role'] == "AI" or old_chat['role'] == 'assistant':
                    if i > len(history_formatted) - 4:
                        history_data.append(
                            {"role": "assistant", "content": old_chat['content']},)
            else:
                history_data.append({"role":"user","content":old_chat["question"]})
                history_data.append({"role":"assistant","content":old_chat["answer"]})
    #history_data.append({"role": "user", "content": prompt})
    content = ''.join([x['content'] for x in history_data])
    if len(content) > 7000:
        #import ipdb
        #ipdb.set_trace()
        history_data = []
        history_data.append({"role": "user", "content": prompt},)
        if len(prompt) > 8000:
            raise ValueError('最长只能支持8000个字符，不要超标')

    #import ipdb
    #ipdb.set_trace()
    #history_data = transform_openai2llama2(history_data)
    #import ipdb
    #ipdb.set_trace()
    prompt = prompt.strip()
    plan_question = '你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，针对以下问题，生成相应的解决问题的计划与步骤:\n' + prompt
    #plan_question = plan_question.strip()
    plan_history_data = copy.deepcopy(history_data)
    plan_history_data.append({"role":"user","content":plan_question})
    print(history_data)
    #output = get_output(plan_question)
    #import ipdb
    #ipdb.set_trace()
    output = get_output(plan_history_data)
    #output = get_ws_content(plan_history_data)
    solution_data = ''
    #import ipdb
    #ipdb.set_trace()
    try:
        """
"{'获取数据': ['从企业数据库中获取数据:获取位于景德镇珠山的企业,>给出企业名称，企业类型,产业', '查询搜索引擎:烟台企业'], '生成答案': [\"'从上述>列表中，选出企业列表'\"], '评价答案': []}"
    {'获取数据': [['从企业数据库中获取数据:北京专精特新企业的企业名称，产业，企业类型'], ['查询搜索引擎:北京专精特新企业']], '生成答案': ['获取答案的前缀', '将答案和前缀进行组合输出'], '评价答案': []}
        """
        #xy = 2/0 
        output = eval(output)
        #output['获取数据']=[['搜索引擎:{}'.format(prompt)]]
        solution_data = ''
        print(output)
        #import ipdb
        #ipdb.set_trace()
        steps = ['获取数据','生成答案','评价答案']
        solution_data = ''
        if output["type"] == "llm":
            response = get_ws_stream_content(prompt)
            #if not solution_data:
            #    solution_data = get_web_data(prompt,zhishiku)
            #response = get_answer_with_context(prompt,solution_data,history_data)
            #response = completion_with_backoff(kwargs)
            resTemp=""
            for chunk in response:
                if '[DONE]' in chunk:
                    continue
                if len(chunk) > 2:
                    chunk = json.loads(chunk)
                    yield chunk["response"].replace('\n','<br />\n')
        if output["type"] == "answer":
            response = output["content"]
            curr = ''
            for chunk in list(response):
                curr += chunk
                time.sleep(0.05)
                yield curr.replace('\n','<br />\n')
        if output["type"] == "step" or output["type"]== "tools" or output["type"]== "plan":
            output = output['content']
            for step in steps: 
                current_plan = output[step]
                if step == '获取数据':
                    #import ipdb
                    #ipdb.set_trace()
                    solution_data = get_solution_data(current_plan,zhishiku,chanyeku)
                if step == '生成答案':
                    answer = generate_answer(solution_data,prompt,current_plan,history_data,zhishiku)
                    if answer is None:
                        raise ValueError('answer为None，抛出异常')
                    if isinstance(answer,str):
                        current_content = ''
                        for token in answer.split('\n'):
                            current_content += token + '\n'
                            time.sleep(0.05)
                            #yield current_content
                            yield current_content.replace('\n','<br />\n')
                    else:
                        for chunk in answer:
                            #print(chunk)
                            #if chunk['choices'][0]["finish_reason"]!="stop":
                            #    if hasattr(chunk['choices'][0]['delta'], 'content'):
                            #        resTemp+=chunk['choices'][0]['delta']['content']
                            #        yield resTemp
                            if '[DONE]' in chunk:
                                continue
                            if len(chunk) > 2:
                                chunk = json.loads(chunk)
                                yield chunk["response"].replace('\n','<br />\n')
    except Exception as e:
        print(e)
        #response = completion_with_backoff(model="gpt-4-0613", messages=history_data, max_tokens=2048, stream=True, headers={"x-api2d-no-cache": "1"},timeout=3)
        #response = get_ws_stream_content(history_data)
        #import ipdb
        #ipdb.set_trace()
        if not solution_data:
            solution_data = get_web_data(prompt,zhishiku)
        response = get_answer_with_context(prompt,solution_data,history_data)
        #import ipdb
        #ipdb.set_trace()
        #response = completion_with_backoff(kwargs)
        resTemp=""
        #import ipdb
        #ipdb.set_trace()
        for chunk in response:
            #print(chunk)
            #if chunk['choices'][0]["finish_reason"]!="stop":
            #    if hasattr(chunk['choices'][0]['delta'], 'content'):
            #        resTemp+=chunk['choices'][0]['delta']['content']
            #        yield resTemp
            if '[DONE]' in chunk:
                continue
            if len(chunk) > 2:
                chunk = json.loads(chunk)
                yield chunk["response"].replace('\n','<br />\n')
        #except:
        #    import ipdb
        #    ipdb.set_trace()
        #    print(1)


chatCompletion = None


def load_model():
    #openai.api_key = os.getenv("OPENAI_API_KEY")
    #openai.api_key = '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'
    #openai.api_key = 'sk-YR2Mtp2ht8u0ruHQ1058B5996dFc40C190B22774D5Bc7964'#测试用
    openai.api_key = 'sk-cRujJbZqefFoj5753c8d94B8F7654c57807cCc3b145aC547'
    #openai.api_key = 'fk217408-4KdxNeEDSjmll43jQ0ItKVKmjhkvi7xH'
    openai.api_base = settings.llm.api_host

class Lock:
    def __init__(self):
        pass

    def get_waiting_threads(self):
        return 0

    def __enter__(self): 
        pass

    def __exit__(self, exc_type, exc_val, exc_tb): 
        pass
