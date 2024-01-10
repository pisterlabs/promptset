# encoding:utf-8

from model.model import Model
from config import model_conf, common_conf_val
from common import const
from common import log
from common.redis import RedisSingleton
from common.word_filter import WordFilter
import openai
import os
import time
import json
import re
import requests
import base64
import random
import hashlib
import tiktoken
from datetime import datetime
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from urllib.parse import urlparse, urlunparse

user_session = dict()
md5sum_pattern = r'^[0-9a-f]{32}$'
faiss_store_root= "/opt/faiss/"

def calculate_md5(text):
    md5_hash = hashlib.md5()
    md5_hash.update(text.encode('utf-8'))
    return md5_hash.hexdigest()

def get_org_bot(input_string):
    parts = input_string.split(':')
    org_part = ":".join(parts[:2])
    bot_part = ":".join(parts[2:])
    return org_part, bot_part

def get_org_id(string):
    pattern = r'org:(\d+)'
    match = re.search(pattern, string)
    orgid = 0
    if match:
        orgid = int(match.group(1))
    return orgid

def get_bot_id(string):
    pattern = r'bot:(\d+)'
    match = re.search(pattern, string)
    botid = 0
    if match:
        botid = int(match.group(1))
    return botid

def get_unique_by_key(data, key):
    seen = set()
    unique_list = [d for d in data if d.get(key) not in seen and not seen.add(d.get(key))]
    return unique_list

def num_tokens_from_string(string):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_messages(messages):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens_per_message = 4
    tokens_per_name = -1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def remove_url_query(url):
    parsed_url = urlparse(url)
    clean_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, '', '', ''))
    return clean_url

def is_image_url(url):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp']
    lower_url = url.lower()
    return any(lower_url.endswith(ext) for ext in image_extensions)

def is_video_url(url):
    video_extensions = ['.mp4', '.webm', '.ogv']
    lower_url = url.lower()
    return any(lower_url.endswith(ext) for ext in video_extensions)

def increase_hit_count(fid, category, url=''):
    gqlurl = 'http://127.0.0.1:5000/graphql'
    gqlfunc = 'increaseHitCount'
    headers = { "Content-Type": "application/json", }
    query = f"mutation {gqlfunc} {{ {gqlfunc}( id:{fid}, category:\"{category}\", url:\"{url}\" ) }}"
    gqldata = { "query": query, "variables": {}, }
    try:
        gqlresp = requests.post(gqlurl, json=gqldata, headers=headers)
        log.info(f"GQL/{gqlfunc}: #{fid} {gqlresp.status_code} {query}")
        log.debug(f"GQL/{gqlfunc}: #{fid} {gqlresp.json()}")
    except Exception as e:
        log.exception(e)

def send_query_notification(rid, str1, str2):
    gqlurl = 'http://127.0.0.1:5000/graphql'
    gqlfunc = 'notiSfbotNotification'
    headers = { "Content-Type": "application/json", }
    chatstr = f"{str1}\n\n{str2}"
    content = base64.b64encode(chatstr.encode('utf-8')).decode('utf-8')
    query = f"mutation {gqlfunc} {{ {gqlfunc}( id:{rid}, content:\"{content}\" ) }}"
    gqldata = { "query": query, "variables": {}, }
    try:
        gqlresp = requests.post(gqlurl, json=gqldata, headers=headers)
        log.info(f"GQL/{gqlfunc}: #{rid} {gqlresp.status_code} {query}")
        log.debug(f"GQL/{gqlfunc}: #{rid} {gqlresp.json()}")
    except Exception as e:
        log.exception(e)

def run_word_filter(text, org_id):
    wftool = WordFilter()
    wfdict,_ = wftool.load_words(0)
    if int(org_id)>0:
        wfdict_org,_ = wftool.load_words(org_id)
        wfdict.update(wfdict_org)
    filted_text = wftool.replace_sensitive_words(text, wfdict)
    return filted_text

# OpenAI对话模型API (可用)
class ChatGPTModel(Model):
    def __init__(self):
        openai.api_key = model_conf(const.OPEN_AI).get('api_key')
        api_base = model_conf(const.OPEN_AI).get('api_base')
        if api_base:
            openai.api_base = api_base
        proxy = model_conf(const.OPEN_AI).get('proxy')
        if proxy:
            openai.proxy = proxy
        log.info("[CHATGPT] api_base={} proxy={}".format(
            api_base, proxy))

    def select_gpt_service(self, vendor='default'):
        if vendor == 'azure':
            openai.api_key = model_conf(const.OPEN_AI).get('azure_api_key')
            openai.api_base = model_conf(const.OPEN_AI).get('azure_api_base')
            openai.api_type = 'azure'
            openai.api_version = '2023-05-15'
        else:
            openai.api_key = model_conf(const.OPEN_AI).get('api_key')
            openai.api_base = model_conf(const.OPEN_AI).get('api_base')

    def reply(self, query, context=None):
        # acquire reply content
        if not context or not context.get('type') or context.get('type') == 'TEXT':
            log.info("[CHATGPT] query={}".format(query))
            from_user_id = context['from_user_id']
            from_org_id = context['from_org_id']
            from_org_id, from_chatbot_id = get_org_bot(from_org_id)
            user_flag = context['userflag']
            nres = int(context.get('res','0'))
            fwd = int(context.get('fwd','0'))
            character_id = context.get('character_id')
            character_desc = context.get('character_desc')
            temperature = context['temperature']
            website = context.get('website','undef')
            email = context.get('email','undef')
            sfmodel = context.get('sfmodel','undef')
            if isinstance(sfmodel, str) and (sfmodel == 'undef' or sfmodel == ''):
                sfmodel = None

            clear_memory_commands = common_conf_val('clear_memory_commands', ['#清除记忆'])
            if query in clear_memory_commands:
                log.info('[CHATGPT] reset session: {}'.format(from_user_id))
                Session.clear_session(from_user_id)
                return 'Session is reset.'

            orgnum = str(get_org_id(from_org_id))
            botnum = str(get_bot_id(from_chatbot_id))
            myredis = RedisSingleton(password=common_conf_val('redis_password', ''))

            teammode = int(context.get('teammode','0'))
            teambotkeep = int(context.get('teambotkeep','0'))
            teamid = int(context.get('teamid','0'))
            teambotid = int(context.get('teambotid','0'))
            if teammode == 1:
                if teambotkeep == 1 and teambotid == 0:
                    teambotkeep = 0
                if teambotkeep == 0:
                    newteambot, newteam = self.find_teambot(user_flag, from_org_id, from_chatbot_id, teamid, query)
                    if newteambot > 0:
                        teamid = newteam
                        teambotid = newteambot
                    else:
                        if teambotid == 0:
                            teammode = 0
                else:
                    if teamid == 0 and teambotid > 0:
                        teambot_pattern = "sfteam:org:{}:team:*:bot:{}".format(orgnum,teambotid)
                        keys_matched = myredis.redis.keys(teambot_pattern)
                        for key in keys_matched:
                            teambot_key=key.decode()
                            teamid=int(teambot_key.split(':')[4])

            if teammode == 1:
                teambot_key = "sfteam:org:{}:team:{}:bot:{}".format(orgnum,teamid,teambotid)
                log.info("[CHATGPT] key={} query={}".format(teambot_key,query))
                if myredis.redis.exists(teambot_key):
                    teambot_name = myredis.redis.hget(teambot_key, 'name').decode().strip()
                    teambot_desc = myredis.redis.hget(teambot_key, 'desc').decode().strip()
                    teambot_prompt = myredis.redis.hget(teambot_key, 'prompt').decode().strip()
                    teambot_model = myredis.redis.hget(teambot_key, 'model')
                else:
                    teammode = 0
            if teammode == 0:
                teamid = 0
                teambotid = 0
            if teammode == 1:
                teambot_instruction = (
                    f"You are {teambot_name}.\n{teambot_desc}.\n"
                    "You only provide factual answers to queries, and do not try to make up an answer.\n"
                    "Do not try to answer the queries that are irrelevant to your functionality and responsibility, just reject them politely.\n"
                    "Your functionality and responsibility are described below, separated by 3 backticks.\n\n"
                    f"```\n{teambot_prompt}\n```\n"
                )
                character_id = f"x{teambotid}"
                character_desc = teambot_instruction
                log.info("[CHATGPT] teambot character id={} desc={}".format(character_id,character_desc))
                if sfmodel is None and teambot_model is not None:
                    sfmodel = teambot_model.decode().strip()
            else:
                sfbot_key = "sfbot:org:{}:bot:{}".format(orgnum,botnum)
                sfbot_model = myredis.redis.hget(sfbot_key, 'model')
                if sfmodel is None and sfbot_model is not None:
                    sfmodel = sfbot_model.decode().strip()

            commands = []
            query_embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]['embedding']
            atcs = myredis.ft_search(embedded_query=query_embedding,
                                     vector_field="text_vector",
                                     hybrid_fields=myredis.create_hybrid_field1(orgnum, user_flag, "category", "atc"),
                                     k=3)
            if len(atcs) > 0:
                for i, atc in enumerate(atcs):
                    if float(atc.vector_score) > 0.15:
                        break
                    cid = myredis.redis.hget(atc.id, 'id').decode()
                    csf = 1.0 - float(atc.vector_score)
                    commands.append({'id':cid,'category':"actionTransformer",'score':csf})

            new_query, hitdocs, refurls, similarity, use_faiss = Session.build_session_query(query, from_user_id, from_org_id, from_chatbot_id, user_flag, character_desc, character_id, website, email, fwd)
            if new_query is None:
                return 'Sorry, I have no ideas about what you said.'

            log.info("[CHATGPT] session query={}".format(new_query))
            if new_query[-1]['role'] == 'assistant':
                reply_message = new_query.pop()
                reply_content = reply_message['content']
                logid = Session.save_session(query, reply_content, from_user_id, from_org_id, from_chatbot_id, 0, 0, 0, similarity, use_faiss)
                reply_content = run_word_filter(reply_content, get_org_id(from_org_id))
                reply_content+='\n```sf-json\n'
                reply_content+=json.dumps({'logid':logid})
                reply_content+='\n```\n'
                return reply_content

            # if context.get('stream'):
            #     # reply in stream
            #     return self.reply_text_stream(query, new_query, from_user_id)

            reply_content, logid = self.reply_text(new_query, query, sfmodel, from_user_id, from_org_id, from_chatbot_id, similarity, temperature, use_faiss, 0)
            reply_embedding = openai.Embedding.create(input=reply_content, model="text-embedding-ada-002")["data"][0]['embedding']
            docs = myredis.ft_search(embedded_query=reply_embedding,
                                     vector_field="text_vector",
                                     hybrid_fields=myredis.create_hybrid_field2(orgnum, botnum, user_flag, "category", "kb"),
                                     k=1)
            score = 0.0
            if len(docs) > 0:
                score = 1.0 - float(docs[0].vector_score)

            qnts = myredis.ft_search(embedded_query=query_embedding, vector_field="text_vector", hybrid_fields=myredis.create_hybrid_field(orgnum, "category", "qnt"), k=3)
            if len(qnts) > 0:
                for i, qnt in enumerate(qnts):
                    log.info(f"{i}) {qnt.id} {qnt.orgid} {qnt.category} {qnt.vector_score}")
                    if float(qnt.vector_score) > 0.2:
                        break
                    rid = myredis.redis.hget(qnt.id, 'id').decode()
                    send_query_notification(rid, query, reply_content)

            resources = []
            if nres > 0:
                resources = Session.get_resources(reply_content, from_user_id, from_org_id)
                reply_content = Session.insert_resource_to_reply(reply_content, from_user_id, from_org_id)
            reply_content = run_word_filter(reply_content, get_org_id(from_org_id))
            reply_content+='\n```sf-json\n'
            reply_content+=json.dumps({'docs':hitdocs,'pages':refurls,'resources':resources,'commands':commands,'score':score,'logid':logid,'teammode':teammode,'teamid':teamid,'teambotid':teambotid})
            reply_content+='\n```\n'
            #log.debug("[CHATGPT] user={}, query={}, reply={}".format(from_user_id, new_query, reply_content))
            return reply_content

        elif context.get('type', None) == 'IMAGE_CREATE':
            return self.create_img(query, 0)

    def find_teambot(self, user_flag, org_id, chatbot_id, team_id, query):
        myredis = RedisSingleton(password=common_conf_val('redis_password', ''))
        orgnum = get_org_id(org_id)
        botnum = get_bot_id(chatbot_id)
        team_info = '# Team Information\n'
        team_keys = []
        team_pattern = "sfteam:org:{}:team:*:data".format(orgnum)
        keys_matched = myredis.redis.keys(team_pattern)
        for key in keys_matched:
            team_keys.append(key.decode())
        if team_id > 0:
            team_key = "sfteam:org:{}:team:{}:data".format(orgnum,team_id)
            if team_key in team_keys:
                team_keys.clear()
                team_keys.append(team_key)
        for key in team_keys:
            team_desc = myredis.redis.hget(key, 'team_desc').decode()
            team_publ = 1
            fpub = myredis.redis.hget(key, 'public')
            if fpub is not None:
                team_publ = int(fpub.decode())
            if team_publ == 1:
                team_info += team_desc+'\n'
            else:
                if user_flag == 'internal':
                    team_info += team_desc+'\n'
        if len(team_info) < 20:
            log.info("[CHATGPT] find_teambot: No available team {}/{}".format(org_id,user_flag))
            return 0, 0
        sys_msg = (
            "You are a contact-center manager, and you try to dispatch the user query to the most suitable team/agent.\n"
            "You only provide factual answers to queries, and do not try to make up an answer.\n"
            "The functionality and responsibility of teams are described below in markdown format.\n\n"
            f"```markdown\n{team_info}\n```\n"
        )
        usr_msg = (
            "Here is user query.\n"
            f"```\n{query}\n```\n\n"
            "Reply the dispatchment in json format with 2 keys named team_id and agent_id.\n"
            "If you have no idea about how to dispatch based on the given team information, simply return team_id=0 and agent_id=0.\n"
            "The answer should be only json string and nothing else.\n"
        )
        msgs = [{'role':'system','content':sys_msg},{'role':'user','content':usr_msg}]
        try:
            use_azure = True if orgnum==4 else False
            response = openai.ChatCompletion.create(
                api_base=(model_conf(const.OPEN_AI).get('azure_api_base') if use_azure else None),
                api_key=(model_conf(const.OPEN_AI).get('azure_api_key') if use_azure else None),
                api_type=("azure" if use_azure else None),
                api_version=("2023-05-15" if use_azure else None),
                engine=("base" if use_azure else None),
                model=model_conf(const.OPEN_AI).get("model") or "gpt-3.5-turbo",
                messages=msgs,
                temperature=0.1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            reply_content = response.choices[0]['message']['content']
            reply_usage = response['usage']
            log.info("[CHATGPT] find_teambot: result={} usage={}".format(reply_content,json.dumps(reply_usage)))
            dispatch = json.loads(reply_content)
            return int(dispatch['agent_id']), int(dispatch['team_id'])
        except Exception as e:
            log.exception(e)
            return 0, 0

    def reply_text(self, query, qtext, qmodel, user_id, org_id, chatbot_id, similarity, temperature, use_faiss=False, retry_count=0):
        try:
            try:
                temperature = float(temperature)
                if temperature < 0.0 or temperature > 1.0:
                    raise ValueError()
            except ValueError:
                temperature = model_conf(const.OPEN_AI).get("temperature", 0.75)

            orgnum = get_org_id(org_id)
            use_azure = True if orgnum==4 else False
            response = openai.ChatCompletion.create(
                api_base=(model_conf(const.OPEN_AI).get('azure_api_base') if use_azure else None),
                api_key=(model_conf(const.OPEN_AI).get('azure_api_key') if use_azure else None),
                api_type=("azure" if use_azure else None),
                api_version=("2023-05-15" if use_azure else None),
                engine=("base" if use_azure else None), # Azure deployment Name
                model=qmodel or model_conf(const.OPEN_AI).get("model") or "gpt-3.5-turbo",  # 对话模型的名称
                messages=query,
                temperature=temperature,  # 熵值，在[0,1]之间，越大表示选取的候选词越随机，回复越具有不确定性，建议和top_p参数二选一使用，创意性任务越大越好，精确性任务越小越好
                #max_tokens=4096,  # 回复最大的字符数，为输入和输出的总数
                #top_p=model_conf(const.OPEN_AI).get("top_p", 0.7),,  #候选词列表。0.7 意味着只考虑前70%候选词的标记，建议和temperature参数二选一使用
                frequency_penalty=model_conf(const.OPEN_AI).get("frequency_penalty", 0.0),  # [-2,2]之间，该值越大则越降低模型一行中的重复用词，更倾向于产生不同的内容
                presence_penalty=model_conf(const.OPEN_AI).get("presence_penalty", 1.0)  # [-2,2]之间，该值越大则越不受输入限制，将鼓励模型生成输入中不存在的新词，更倾向于产生不同的内容
            )
            reply_content = response.choices[0]['message']['content']
            used_tokens = response['usage']['total_tokens']
            prompt_tokens = response['usage']['prompt_tokens']
            completion_tokens = response['usage']['completion_tokens']
            log.debug(response)
            log.info("[CHATGPT] usage={}", json.dumps(response['usage']))
            log.info("[CHATGPT] reply={}", reply_content)
            logid = Session.save_session(qtext, reply_content, user_id, org_id, chatbot_id, used_tokens, prompt_tokens, completion_tokens, similarity, use_faiss)
            return reply_content, logid
        except openai.error.RateLimitError as e:
            # rate limit exception
            log.warn(e)
            if retry_count < 1:
                time.sleep(5)
                log.warn("[CHATGPT] RateLimit exceed, retry {} attempts".format(retry_count+1))
                return self.reply_text(query, qtext, qmodel, user_id, org_id, chatbot_id, similarity, temperature, use_faiss, retry_count+1)
            else:
                return "You're asking too quickly, please take a break before asking me again.", None
        except openai.error.APIConnectionError as e:
            log.warn(e)
            log.warn("[CHATGPT] APIConnection failed")
            return "I can't connect to the service, please try again later.", None
        except openai.error.Timeout as e:
            log.warn(e)
            log.warn("[CHATGPT] Timeout")
            return "I haven't received the message, please try again later.", None
        except openai.error.ServiceUnavailableError as e:
            log.warn(e)
            log.warn("[CHATGPT] Service Unavailable")
            return "The server is overloaded or not ready yet.", None
        except Exception as e:
            # unknown exception
            log.exception(e)
            Session.clear_session(user_id)
            return "Oops, something wrong, please ask me again.", None


    async def reply_text_stream(self, query, context, retry_count=0):
        try:
            log.info("[CHATGPT] query={}".format(query))
            from_user_id = context['from_user_id']
            from_org_id = context['from_org_id']
            from_org_id, from_chatbot_id = get_org_bot(from_org_id)
            user_flag = context['userflag']
            nres = int(context.get('res','0'))
            fwd = int(context.get('fwd','0'))
            character_id = context.get('character_id')
            character_desc = context.get('character_desc')
            temperature = context['temperature']
            website = context.get('website','undef')
            email = context.get('email','undef')
            sfmodel = context.get('sfmodel','undef')
            if isinstance(sfmodel, str) and (sfmodel == 'undef' or sfmodel == ''):
                sfmodel = None
            new_query, hitdocs, refurls, similarity, use_faiss = Session.build_session_query(query, from_user_id, from_org_id, from_chatbot_id, user_flag, character_desc, character_id, website, email, fwd)
            if new_query is None:
                yield True,'Sorry, I have no ideas about what you said.'

            log.info("[CHATGPT] session query={}".format(new_query))
            if new_query[-1]['role'] == 'assistant':
                reply_message = new_query.pop()
                reply_content = reply_message['content']
                logid = Session.save_session(query, reply_content, from_user_id, from_org_id, from_chatbot_id, 0, 0, 0, similarity, use_faiss)
                reply_content = run_word_filter(reply_content, get_org_id(from_org_id))
                reply_content+='\n```sf-json\n'
                reply_content+=json.dumps({'logid':logid})
                reply_content+='\n```\n'
                yield True,reply_content

            try:
                temperature = float(temperature)
                if temperature < 0.0 or temperature > 1.0:
                    raise ValueError()
            except ValueError:
                temperature = model_conf(const.OPEN_AI).get("temperature", 0.75)

            orgnum = str(get_org_id(from_org_id))
            botnum = str(get_bot_id(from_chatbot_id))
            myredis = RedisSingleton(password=common_conf_val('redis_password', ''))
            sfbot_key = "sfbot:org:{}:bot:{}".format(orgnum,botnum)
            sfbot_model = myredis.redis.hget(sfbot_key, 'model')
            if sfmodel is None and sfbot_model is not None:
                sfmodel = sfbot_model.decode().strip()

            res = openai.ChatCompletion.create(
                model=sfmodel or model_conf(const.OPEN_AI).get("model") or "gpt-3.5-turbo",  # 对话模型的名称
                messages=new_query,
                temperature=temperature,  # 熵值，在[0,1]之间，越大表示选取的候选词越随机，回复越具有不确定性，建议和top_p参数二选一使用，创意性任务越大越好，精确性任务越小越好
                #max_tokens=4096,  # 回复最大的字符数，为输入和输出的总数
                #top_p=model_conf(const.OPEN_AI).get("top_p", 0.7),,  #候选词列表。0.7 意味着只考虑前70%候选词的标记，建议和temperature参数二选一使用
                frequency_penalty=model_conf(const.OPEN_AI).get("frequency_penalty", 0.0),  # [-2,2]之间，该值越大则越降低模型一行中的重复用词，更倾向于产生不同的内容
                presence_penalty=model_conf(const.OPEN_AI).get("presence_penalty", 1.0),  # [-2,2]之间，该值越大则越不受输入限制，将鼓励模型生成输入中不存在的新词，更倾向于产生不同的内容
                stream=True
            )
            full_response = ""
            for chunk in res:
                log.debug(chunk)
                if (chunk["choices"][0]["finish_reason"]=="stop"):
                    break
                chunk_message = chunk['choices'][0]['delta'].get("content")
                if(chunk_message):
                    full_response+=chunk_message
                yield False,full_response

            prompt_tokens = num_tokens_from_messages(new_query)
            completion_tokens = num_tokens_from_string(full_response)
            used_tokens = prompt_tokens + completion_tokens
            logid = Session.save_session(query, full_response, from_user_id, from_org_id, from_chatbot_id, used_tokens, prompt_tokens, completion_tokens, similarity, use_faiss)

            resources = []
            if nres > 0:
                resources = Session.get_resources(full_response, from_user_id, from_org_id)

            full_response = run_word_filter(full_response, get_org_id(from_org_id))
            full_response+='\n```sf-json\n'
            full_response+=json.dumps({'docs':hitdocs,'pages':refurls,'resources':resources,'logid':logid})
            full_response+='\n```\n'
            #log.debug("[CHATGPT] user={}, query={}, reply={}".format(from_user_id, new_query, full_response))
            yield True,full_response

        except openai.error.RateLimitError as e:
            # rate limit exception
            log.warn(e)
            if retry_count < 1:
                time.sleep(5)
                log.warn("[CHATGPT] RateLimit exceed, retry {} attempts".format(retry_count+1))
                yield True, self.reply_text_stream(query, context, retry_count+1)
            else:
                yield True, "You're asking too quickly, please take a break before asking me again."
        except openai.error.APIConnectionError as e:
            log.warn(e)
            log.warn("[CHATGPT] APIConnection failed")
            yield True, "I can't connect to the service, please try again later."
        except openai.error.Timeout as e:
            log.warn(e)
            log.warn("[CHATGPT] Timeout")
            yield True, "I haven't received the message, please try again later."
        except openai.error.ServiceUnavailableError as e:
            log.warn(e)
            log.warn("[CHATGPT] Service Unavailable")
            yield True, "The server is overloaded or not ready yet."
        except Exception as e:
            # unknown exception
            log.exception(e)
            Session.clear_session(from_user_id)
            yield True, "Oops, something wrong, please ask me again."

    def create_img(self, query, retry_count=0):
        try:
            log.info("[OPEN_AI] image_query={}".format(query))
            response = openai.Image.create(
                prompt=query,    #图片描述
                n=1,             #每次生成图片的数量
                size="256x256"   #图片大小,可选有 256x256, 512x512, 1024x1024
            )
            image_url = response['data'][0]['url']
            log.info("[OPEN_AI] image_url={}".format(image_url))
            return [image_url]
        except openai.error.RateLimitError as e:
            log.warn(e)
            if retry_count < 1:
                time.sleep(5)
                log.warn("[OPEN_AI] ImgCreate RateLimit exceed, retry {} attempts".format(retry_count+1))
                return self.create_img(query, retry_count+1)
            else:
                return "You're asking too quickly, please take a break before asking me again."
        except Exception as e:
            log.exception(e)
            return None


class Session(object):
    @staticmethod
    def build_session_query(query, user_id, org_id, chatbot_id='bot:0', user_flag='external', character_desc=None, character_id=None, website=None, email=None, fwd=0):
        '''
        build query with conversation history
        e.g.  [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
        :param query: query content
        :param user_id: from user id
        :return: query content with conversaction
        '''
        config_prompt = common_conf_val("input_prompt", "")
        session = user_session.get(user_id, [])

        faiss_id = user_id
        if isinstance(website, str) and website != 'undef' and len(website) > 0:
            log.info("[FAISS] try to search data of website:{}".format(website))
            faiss_id = calculate_md5('website:'+re.sub(r'https?://','',website.lower()))
        elif isinstance(email, str) and email != 'undef' and len(email) > 0:
            log.info("[FAISS] try to search data of email:{}".format(email))
            faiss_id = calculate_md5(email.lower())

        if re.match(md5sum_pattern, faiss_id):
            log.info("[FAISS] try to load local store {}".format(faiss_id))
        if re.match(md5sum_pattern, faiss_id) and os.path.exists(f"{faiss_store_root}{faiss_id}"):
            faiss_store_path = f"{faiss_store_root}{faiss_id}"
            mykey = model_conf(const.OPEN_AI).get('api_key')
            embeddings = OpenAIEmbeddings(openai_api_key=mykey)
            dbx = FAISS.load_local(faiss_store_path, embeddings)
            log.info("[FAISS] local store loaded")
            similarity = 0.0
            docs = dbx.similarity_search_with_score(query, k=3)
            log.info("[FAISS] semantic search done")
            if len(docs) == 0:
                log.info("[FAISS] semantic search: None")
                return None, [], [], similarity, True
            similarity = float(docs[0][1])
            '''
            if len(docs) > 0 and similarity < 0.6:
                log.info(f"[FAISS] semantic search: score:{similarity} < threshold:0.6")
                return None, [], [], similarity, True
            '''
            system_prompt = 'You are answering the question just like you are the owner or partner of the company described in the context.'
            if isinstance(character_desc, str) and character_desc != 'undef' and len(character_desc) > 0:
                system_prompt = character_desc
            system_prompt += '\nIf you don\'t know the answer, just say you don\'t know. DO NOT try to make up an answer.'
            system_prompt += '\nIf the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.'
            system_prompt += '\nIf you are unclear about the question, politely respond that you need a clearer and more detailed description.'
            system_prompt += f"\n{config_prompt}\n```"
            for doc, score in docs:
                log.info("[FAISS] {} {}".format(score, json.dumps(doc.metadata)))
                '''
                if score < 0.6:
                    break
                '''
                system_prompt += '\n' + doc.page_content
            system_prompt += '\n```\n'
            log.info("[FAISS] prompt={}".format(system_prompt))
            if len(session) > 0 and session[0]['role'] == 'system':
                session.pop(0)
            session = []
            system_item = {'role': 'system', 'content': system_prompt}
            session.insert(0, system_item)
            user_session[user_id] = session
            user_item = {'role': 'user', 'content': query}
            session.append(user_item)
            return session, [], [], similarity, True

        orgnum = get_org_id(org_id)
        qnaorg = "(0|{})".format(orgnum)
        botnum = str(get_bot_id(chatbot_id))
        if isinstance(character_id, str) and (character_id[0] == 'c' or character_id[0] == 'x' or character_id[0] == 't'):
            botnum += " | {}".format(character_id)
        refurls = []
        hitdocs = []
        qna_output = None
        myquery = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]['embedding']
        myredis = RedisSingleton(password=common_conf_val('redis_password', ''))
        qnas = myredis.ft_search(embedded_query=myquery, vector_field="title_vector", hybrid_fields=myredis.create_hybrid_field(qnaorg, "category", "qa"))
        if len(qnas) > 0 and float(qnas[0].vector_score) < 0.15:
            qna = qnas[0]
            log.info(f"Q/A: {qna.id} {qna.orgid} {qna.category} {qna.vector_score}")
            try:
                qnatext = myredis.redis.hget(qna.id, 'text').decode()
                answers = json.loads(qnatext)
                if len(answers)>0:
                    qna_output = random.choice(answers)
                    fid = myredis.redis.hget(qna.id, 'id').decode()
                    increase_hit_count(fid, 'qa', '')
            except json.JSONDecodeError as e:
                pass
            except Exception as e:
                pass

        log.info("[RDSFT] org={} {} {}".format(org_id, orgnum, qnaorg))
        similarity = 0.0
        docs = myredis.ft_search(embedded_query=myquery,
                                 vector_field="text_vector",
                                 hybrid_fields=myredis.create_hybrid_field2(str(orgnum), botnum, user_flag, "category", "kb"))
        if len(docs) > 0:
            similarity = 1.0 - float(docs[0].vector_score)
            threshold = float(common_conf_val('similarity_threshold', 0.7))
            if similarity < threshold:
                docs = []

        system_prompt = 'You are a helpful AI customer support agent. Use the following pieces of context to answer the customer inquiry.'
        orgnum = str(get_org_id(org_id))
        botnum = str(get_bot_id(chatbot_id))
        sfbot_key = "sfbot:org:{}:bot:{}".format(orgnum,botnum)
        sfbot_char_desc = myredis.redis.hget(sfbot_key, 'character_desc')
        if sfbot_char_desc is not None:
            sfbot_char_desc = sfbot_char_desc.decode()
            if len(sfbot_char_desc) > 0:
                system_prompt = sfbot_char_desc
        if isinstance(character_desc, str) and character_desc != 'undef' and len(character_desc) > 0:
            system_prompt = character_desc

        if fwd > 0:
            log.info("[CHATGPT] prompt(onlyfwd)={}".format(system_prompt))
            if len(session) > 0 and session[0]['role'] == 'system':
                session.pop(0)
            system_item = {'role': 'system', 'content': system_prompt}
            session.insert(0, system_item)
            user_session[user_id] = session
            user_item = {'role': 'user', 'content': query}
            session.append(user_item)
            return session, [], [], similarity, False

        if isinstance(character_id, str) and character_id.startswith('x'):
            log.info("[CHATGPT] teambot character id={} add context".format(character_id))
        else:
            system_prompt += '\nIf you don\'t know the answer, just say you don\'t know. DO NOT try to make up an answer.'
            system_prompt += '\nIf the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.'
            system_prompt += '\nIf you are unclear about the question, politely respond that you need a clearer and more detailed description.'

        if len(docs) == 0 and qna_output is None:
            log.info("[CHATGPT] prompt(nodoc)={}".format(system_prompt))
            if len(session) > 0 and session[0]['role'] == 'system':
                session.pop(0)
            system_item = {'role': 'system', 'content': system_prompt}
            session.insert(0, system_item)
            user_session[user_id] = session
            user_item = {'role': 'user', 'content': query}
            session.append(user_item)
            return session, [], [], similarity, False

        system_prompt += f"\n{config_prompt}\n```"
        if qna_output is not None:
            system_prompt += '\n' + qna_output
        for i, doc in enumerate(docs):
            log.info(f"{i}) {doc.id} {doc.orgid} {doc.category} {doc.vector_score}")
            system_prompt += '\n' + myredis.redis.hget(doc.id, 'text').decode()
            if float(doc.vector_score) < 0.15:
                urlhit = ''
                docurl = myredis.redis.hget(doc.id, 'source')
                if docurl is not None:
                    urlhit = docurl.decode()
                dockey = myredis.redis.hget(doc.id, 'dkey')
                if dockey is not None:
                    dockey = dockey.decode()
                    dockeyparts = dockey.split(":")
                    fct = dockeyparts[1]
                    fid = dockeyparts[2]
                    if fct == 'file':
                        dfname = myredis.redis.hget(doc.id, 'filename')
                        if dfname is not None:
                            dfname = dfname.decode()
                        hitdocs.append({'id':fid,'category':fct,'url':urlhit,'filename':dfname,'key':f"{fid};{urlhit}"})
            if float(doc.vector_score) < 0.2:
                docurl = myredis.redis.hget(doc.id, 'source')
                if docurl is None:
                    continue
                urlkey = myredis.redis.hget(doc.id, 'refkey')
                if urlkey is None:
                    continue
                urltitle = None
                try:
                    docurl = docurl.decode()
                    urlkey = urlkey.decode()
                    urlmeta = json.loads(myredis.redis.lindex(urlkey, 0).decode())
                    urltitle = urlmeta['title']
                except json.JSONDecodeError as e:
                    print("Error decoding JSON:", urlkey, str(e))
                except Exception as e:
                    print("Error URL:", urlkey, str(e))
                log.info(f"{i}) {doc.id} URL={docurl} Title={urltitle}")
                refurls.append({'url': docurl, 'title': urltitle})
        system_prompt += '\n```\n'
        log.info("[CHATGPT] prompt={}".format(system_prompt))
        refurls = get_unique_by_key(refurls, 'url')
        hitdocs = get_unique_by_key(hitdocs, 'key')
        hitdocs = [{k: v for k, v in d.items() if k != 'key'} for d in hitdocs]
        for doc in hitdocs:
            increase_hit_count(doc['id'], doc['category'], doc['url'])
        if len(session) > 0 and session[0]['role'] == 'system':
            session.pop(0)
        system_item = {'role': 'system', 'content': system_prompt}
        session.insert(0, system_item)
        user_session[user_id] = session
        user_item = {'role': 'user', 'content': query}
        session.append(user_item)
        return session, hitdocs, refurls, similarity, False

    @staticmethod
    def save_session(query, answer, user_id, org_id, chatbot_id, used_tokens=0, prompt_tokens=0, completion_tokens=0, similarity=0.0, use_faiss=False):
        max_tokens = model_conf(const.OPEN_AI).get('conversation_max_tokens')
        max_history_num = model_conf(const.OPEN_AI).get('max_history_num', None)
        if not max_tokens or max_tokens > 4000:
            # default value
            max_tokens = 1000
        session = user_session.get(user_id)
        if session:
            # append conversation
            gpt_item = {'role': 'assistant', 'content': answer}
            session.append(gpt_item)

        if used_tokens > max_tokens and len(session) >= 3:
            # pop first conversation (TODO: more accurate calculation)
            session.pop(1)
            session.pop(1)

        if max_history_num is not None:
            while len(session) > max_history_num * 2 + 1:
                session.pop(1)
                session.pop(1)

        if use_faiss:
            return None

        if used_tokens > 0:
            myredis = RedisSingleton(password=common_conf_val('redis_password', ''))
            botkey = "sfbot:{}:{}".format(org_id,chatbot_id)
            momkey = 'stat_'+datetime.now().strftime("%Y%m")
            momqty = myredis.redis.hget(botkey, momkey)
            if momqty is None:
                myredis.redis.hset(botkey, momkey, 1)
            else:
                momqty = int(momqty.decode())
                myredis.redis.hset(botkey, momkey, momqty+1)

        gqlurl = 'http://127.0.0.1:5000/graphql'
        gqlfunc = 'createChatHistory'
        headers = { "Content-Type": "application/json", }
        orgnum = get_org_id(org_id)
        botnum = get_bot_id(chatbot_id)
        question = base64.b64encode(query.encode('utf-8')).decode('utf-8')
        answer = base64.b64encode(answer.encode('utf-8')).decode('utf-8')
        xquery = f"""mutation {gqlfunc} {{ {gqlfunc}( chatHistory:{{ tag:"{user_id}",organizationId:{orgnum},sfbotId:{botnum},question:"{question}",answer:"{answer}",similarity:{similarity},promptTokens:{prompt_tokens},completionTokens:{completion_tokens},totalTokens:{used_tokens}}}){{ id tag }} }}"""
        gqldata = { "query": xquery, "variables": {}, }
        try:
            gqlresp = requests.post(gqlurl, json=gqldata, headers=headers)
            log.info("[HISTORY] response: {} {}".format(gqlresp.status_code, gqlresp.text.strip()))
            if gqlresp.status_code != 200:
                return None
            chatlog = json.loads(gqlresp.text)
            return chatlog['data']['createChatHistory']['id']
        except Exception as e:
            log.exception(e)
            return None

    @staticmethod
    def clear_session(user_id):
        user_session[user_id] = []

    @staticmethod
    def get_resources(query, user_id, org_id):
        orgnum = get_org_id(org_id)
        resorg = "(0|{})".format(orgnum)
        myquery = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]['embedding']
        myredis = RedisSingleton(password=common_conf_val('redis_password', ''))
        ress = myredis.ft_search(embedded_query=myquery, vector_field="text_vector", hybrid_fields=myredis.create_hybrid_field(resorg, "category", "res"), k=5)
        if len(ress) == 0:
            return []

        resources = []
        for i, res in enumerate(ress):
            resurl = myredis.redis.hget(res.id, 'url')
            resnam = myredis.redis.hget(res.id, 'title')
            vscore = 1.0 - float(res.vector_score)
            if resurl is not None:
                resurl = resurl.decode()
                resnam = resnam.decode()
                resources.append({'url':resurl,'name':resnam,'score':vscore})
        resources = get_unique_by_key(resources, 'url')
        return resources

    @staticmethod
    def get_top_resource(query, user_id, org_id, pos=0):
        orgnum = get_org_id(org_id)
        resorg = "(0|{})".format(orgnum)
        myquery = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]['embedding']
        myredis = RedisSingleton(password=common_conf_val('redis_password', ''))
        ress = myredis.ft_search(embedded_query=myquery, vector_field="text_vector", hybrid_fields=myredis.create_hybrid_field(resorg, "category", "res"), k=1, offset=pos)
        if len(ress) == 0:
            return None
        res0 = ress[0]
        if float(res0.vector_score) > 0.25:
            return None
        resurl = myredis.redis.hget(res0.id, 'url')
        if resurl is None:
            return None
        resurl = resurl.decode()
        resname = myredis.redis.hget(res0.id, 'title')
        vscore = 1.0 - float(res0.vector_score)
        if resname is not None:
            resname = resname.decode()
        urlnoq = remove_url_query(resurl)
        restype = 'unknown'
        if is_image_url(urlnoq):
            restype = 'image'
        elif is_video_url(urlnoq):
            restype = 'video'
        topres = {'rid':res0.id, 'url':resurl,'name':resname,'type':restype,'score':vscore}
        return topres

    @staticmethod
    def insert_resource_to_reply(text, user_id, org_id):
        resrids=set()
        paragraphs = text.split("\n\n")
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) < 50:
                continue
            found = False
            for j in range(10):
                resource = Session.get_top_resource(paragraph, user_id, org_id, j)
                if resource is None:
                    found = False
                    break
                resrid = resource['rid']
                if resrid not in resrids:
                    found = True
                    resrids.add(resrid)
                    break
            if not found:
                continue
            resurl = resource['url']
            resname = resource['name']
            restype = resource['type']
            if restype == 'image':
                imagetag = f"\n\n<img src=\"{resurl}\" alt=\"{resname}\" width=\"600\">\n\n\n"
                paragraphs[i] = paragraphs[i] + imagetag
            elif restype == 'video':
                videotag = f"\n\n<video width=\"600\" controls><source src=\"{resurl}\" type=\"video/mp4\">Your browser does not support the video tag.</video>\n\n\n"
                paragraphs[i] = paragraphs[i] + videotag
        modified_text = "\n\n".join(paragraphs)
        return modified_text
