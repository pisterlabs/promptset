# -*- coding: utf-8 -*-
import openai
import json
import os
import requests
from datetime import datetime
from openai import OpenAIError
from retrying import retry

from config.config import gitlab_server_url, gitlab_private_token, openai_api_key, openai_baseurl, openai_model_name, feishu_webhook, review_prompt, review_file_count, review_file_suffix, review_file_contain
from service.content_handle import filter_diff_content
from utils.LogHandler import log

"""
传入project_id和project_commit_id
ChatGPT代码补丁审查
"""

# 配置openai
openai_api_key = openai_api_key
gitlab_private_token = gitlab_private_token
gitlab_server_url = gitlab_server_url
headers = {
    "PRIVATE-TOKEN": gitlab_private_token,
}


def send_to_feishu_webhook(commit_url, title, branch, user_name, time):
    # 过滤掉branch中的"refs/heads/"
    filtered_branch = branch.replace("refs/heads/", "")

    # 裁剪时间到年月日，时分秒
    # 假设时间格式为 "2023-12-22T16:01:45+08:00"
    trimmed_time = time.split("+")[0]  # 这将只保留日期和时间部分，不包括时区

    # 创建飞书消息卡片内容
    feishu_webhook_message = {
        "msg_type": "interactive",
        "card": {
            "config": {
                "wide_screen_mode": True
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**分支:** {filtered_branch}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**用户:** {user_name}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**修改内容:** {title}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**提交时间:** {trimmed_time}"
                    }
                },
                {
                    "tag": "action",
                    "actions": [
                        {
                            "tag": "button",
                            "text": {
                                "tag": "plain_text",
                                "content": "查看Commit提交"
                            },
                            "url": commit_url,
                            "type": "primary"
                        }
                    ]
                }
            ]
        }
    }

    # 发送请求到飞书Webhook
    headers = {'Content-Type': 'application/json'}
    response = requests.post(feishu_webhook, headers=headers, json=feishu_webhook_message)
    return response.status_code, response.text


@retry(stop_max_attempt_number=3, wait_fixed=2000)
def post_comments(id, commit_id, content, commit_url, title, branch, user_name, time):
    data = {
        'note': content
    }
    comments_url = f'{gitlab_server_url}/api/v4/projects/{id}/repository/commits/{commit_id}/comments'
    response = requests.post(comments_url, headers=headers, json=data)
    log.debug(f"请求结果: {response.json}")
    if response.status_code == 201:
        comment_data = response.json()
        # 处理创建的评论数据
        log.info(f"创建评论成功，评论id: {comment_data}")
    else:
        log.error(f"请求失败，状态码: {response.status_code}")


def wait_and_retry(exception):
    return isinstance(exception, OpenAIError)

@retry(retry_on_exception=wait_and_retry, stop_max_attempt_number=3, wait_fixed=60000)
def generate_review_note(change):
    content = filter_diff_content(change['diff'])
    openai.api_key = openai_api_key
    openai.api_base = openai_baseurl
    log.info(f"review content: {content}")
    messages = [
        {
            "role": "system",
            "content": review_prompt
        },
        {
            "role": "user",
            "content": f"请根据上述标准审查以下代码变更，并提供您的评分和反馈：{content}"
        }
    ]

    response = openai.ChatCompletion.create(
        model=openai_model_name,
        messages=messages,
    )
    new_path = change['new_path']
    log.info(f'对 {new_path} review中...')
    response_content = response['choices'][0]['message']['content'].replace('\n\n', '\n')
    total_tokens = response['usage']['total_tokens']
    review_note = f'### `{new_path}`' + '\n\n'
    review_note += f'({total_tokens} tokens) {"AI review 意见如下:"}' + '\n\n'
    review_note += response_content
    log.info(f'对 {new_path} review结束')
    return review_note

def chat_review(project_id, project_commit_id, content, commit_url, title, branch, user_name, time, loop_limit):
    log.info('开始code review')
    count = 0  # 记录循环次数
    for change in content:
        if count >= loop_limit:
            log.info(f'已达到循环次数上限 {loop_limit}，结束循环')
            break

        log.info(f"单项目的commit内容： {change}")
        
        file_name = os.path.basename(change['new_path'])
        log.info(f"file_name： {file_name}")
        if any(ext in change['new_path'] for ext in review_file_suffix):
            if file_name in review_file_contain or file_name.endswith('.g.dart'):
                log.info(f"跳过： {file_name}")
                pass
            else:
                log.info(f"没跳过： {file_name}")
                try:
                    review_note = generate_review_note(change)
                    log.info(f'对 {change["new_path"]}，review结果如下：{review_note}')
                    post_comments(project_id, project_commit_id, review_note, commit_url, title, branch, user_name, time)
                    count += 1
                except Exception as e:
                    log.error(f'出现异常，异常信息：{e}')
        else:
            log.error(f'格式不正确，对 {change["new_path"]}，不需要review')

    send_to_feishu_webhook(commit_url, title, branch, user_name, time)


@retry(stop_max_attempt_number=3, wait_fixed=2000)
def review_code(project_id, project_commit_id, commit_list_url, title, branch, user_name, time):
    if not isinstance(commit_list_url, list) or len(commit_list_url) != 1:
        return
    
    commit_url = commit_list_url[0]
    for commit_id in project_commit_id:
        url = f'{gitlab_server_url}/api/v4/projects/{project_id}/repository/commits/{commit_id}/diff'
        log.info(f"开始请求gitlab的{url}   ,commit: {commit_id}的diff内容")

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            content = response.json()
            # 开始处理请求的类容
            log.info(f"开始处理All请求的类容: {content}")
            chat_review(project_id, commit_id, content, commit_url, title, branch, user_name, time, review_file_count)
        else:
            log.error(f"请求gitlab的{url}commit失败，状态码：{response.status_code}")
            raise Exception(f"请求gitlab的{url}commit失败，状态码：{response.status_code}")

if __name__ == '__main__':
    project_id = 787
    project_commit_id = ['ac98654c27a669bf88ce6d261d371a259c19dfcc']
    commit_list_url = ['']
    title = ""
    branch = ""
    user_name = ""
    time = ""

    log.info(f"项目id: {project_id}，commit_id: {project_commit_id} 开始进行ChatGPT代码补丁审查")
    review_code(project_id, project_commit_id, commit_list_url, title, branch, user_name, time)
