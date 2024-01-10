import os
from datetime import datetime
from typing import Dict

from openai import OpenAI

import json
from serpapi import GoogleSearch

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M")

# botに使ってほしいfunction
functions = [
  {
    "name": "web_search",
    "description": "If you need information that cannot be answered normally, search the web for information. Be sure to use this feature if the user's question contains words like 'search the web'",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Find information from users. Split by word to search for",
        },
      },
      "required": ["query"],
    },
  },
]

def web_search(query: str) -> Dict[str, str]:
  search = GoogleSearch({
    "engine": "yahoo",
    "p": query,
    "api_key": os.environ.get('SERP_API_KEY')
  })
  result = search.get_dict()
  top3_result = result["organic_results"][:3]

  res = {
    "snippet":"",
    "link":""
  }

  for i in top3_result:
    res["snippet"] += i["snippet"] + ","
    res["link"] += i["link"] + " "

  return res

def send_prompt(messages, uid):

  if any(item.get("role") != "system" for item in messages):
    messages = [{"role": "system", "content": os.environ.get('CHARACTER_PROMPT')},
                {"role": "system", "content": "User comments are in the format Name:UID: Comment"},
                {"role": "system", "content": f"You are UID:{uid}"},
                {"role": "user", "content": "name:257827101397352450: hi!"},
                {"role": "assistant", "content": "こんにちはにゃ！"}
                ] + messages

  client = OpenAI(
    api_key = os.environ.get('OPEN_AI_API_KEY')
  )

  try:
    response = client.chat.completions.create(
      model="gpt-4-vision-preview",
      # model="gpt-3.5-turbo",
      # model="gpt-4-0613",
      messages=messages,
      #functions=functions,
      #function_call="auto",
      max_tokens=300,
      timeout=30,
    )

    #message = response['choices'][0]['message']
    message = response.choices[0].message
    messages.append(message)

    # modelがfunctionの呼び出しを求めている
    # if message.get("function_call"):
    #   function_name = message["function_call"]["name"]
    #   arguments=json.loads(message["function_call"]["arguments"])
    #
    #   # modelの求めるfunctionを呼ぶ
    #   match function_name:
    #     case "web_search":
    #       function_response = web_search(
    #         query=arguments.get("query"),
    #       )
    #       content = function_response["snippet"] + ' You can read more at: ' + function_response["link"]
    #
    #
    #   # modelに結果を送る
    #   second_response = client.chat.completions.create(
    #     model="gpt-3.5-turbo-0613",
    #     messages=messages + [{
    #       "role": "function",
    #       "name": function_name,
    #       "content": content,
    #     }]
    #   )
    #   message = second_response['choices'][0]['message']
    #   messages.append(message)

    print(messages)
    return messages
  except:
    raise
