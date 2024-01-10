import openai
import tiktoken
import config
import db_functions
import markdown as md
import random

openai.api_key = config.DevelopmentConfig.OPENAI_KEY
MAX_TOKENS_RESP = 150
MAX_TOKENS_REQ = 300



def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def markdown(value):
    return md.markdown(value, extensions=[
        'markdown.extensions.footnotes',
        'markdown.extensions.footnotes',
        'markdown.extensions.attr_list',
        'markdown.extensions.def_list',
        'markdown.extensions.tables',
        'markdown.extensions.abbr',
        'markdown.extensions.md_in_html',
        'pymdownx.highlight',
        'pymdownx.superfences',
        'pymdownx.mark',
        'pymdownx.arithmatex',
    ],
                       extension_configs={
        "pymdownx.arithmatex": {
            'generic': True,
        },
       "pymdownx.tasklist": {
           "custom_checkbox": True,
       },
       "pymdownx.highlight": {
           'use_pygments': True,
           'guess_lang': True,
           'noclasses': False,
           'pygments_style': 'friendly',
       },
    })


def close_blocks_check(msg: str) -> str:
    """
    Функция, которая проверяет на закрытие блоков в html. Например, бот мог отослать сообщение,
    но он его обрезал по середине блока кода. В этом случае плывет вся разметка

    msg: - сообщение
    return: -отформатированное сообщение
    """
    msg = msg.replace("\\n", "\n").replace('\\"', '\"')
    if msg.count("```") % 2 != 0:
        msg += "\n...\n```"

    return msg

def generateChatResponse(prompt, ctx_messages, tokens_left):
    model = "gpt-3.5-turbo"

    messages = [{"role": "user", "content": "You are a helpful assistant."}]

    ctx_messages = [{ 'role': msg[0], 'content': msg[1] } for msg in ctx_messages]
    messages.extend(ctx_messages)

    question = { 'role': 'user', 'content': prompt }
    messages.append(question)

    msg_len = num_tokens_from_messages(messages)

    print(messages)
    print('question_len', msg_len)
    print('tokens_left', tokens_left)
    


    if msg_len > tokens_left:
        return (question, "", 0, "Promt too long")

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )

    usage = response['usage']
    total_tokens_usage = usage['total_tokens']
    if len(response['choices']) == 0:
        return (question, "", total_tokens_usage, "Oops you beat the AI, try different questions, if the problem persists, come back later.")
    
    answer_msg = response['choices'][0]['message']

    print('actual question_len', usage['prompt_tokens'])
    print('usage', total_tokens_usage)

    answer_msg["content"] = close_blocks_check(answer_msg["content"])
    answer_msg["content"] = markdown(answer_msg["content"])
    print(question)
    print(answer_msg)

    return (question, answer_msg, total_tokens_usage, "ok")


def generateFakeChatResponse(prompt, ctx_messages, tokens_left):
    model = "gpt-3.5-turbo"

    messages = [{"role": "user", "content": "You are a helpful assistant."}]

    ctx_messages = [{ 'role': msg[0], 'content': msg[1] } for msg in ctx_messages]
    messages.extend(ctx_messages)

    question = { 'role': 'user', 'content': prompt }
    messages.append(question)

    msg_len = num_tokens_from_messages(messages)

    print(messages)
    print('question_len', msg_len)
    print('tokens_left', tokens_left)
    


    if msg_len > MAX_TOKENS_REQ or msg_len > tokens_left:
        return (question, "", 0, "Promt too long")
    answer_msg = {
        "content": "<p>\u041a \u0441\u043e\u0436\u0430\u043b\u0435\u043d\u0438\u044e, \u044f \u043d\u0435 \u043c\u043e\u0433\u0443 \u043f\u0440\u043e\u0432\u0435\u0440\u0438\u0442\u044c \u043f\u043e\u0433\u043e\u0434\u0443 \u0432 \u0440\u0435\u0430\u043b\u044c\u043d\u043e\u043c \u0432\u0440\u0435\u043c\u0435\u043d\u0438, \u0442\u0430\u043a \u043a\u0430\u043a \u044f \u043d\u0435 \u0438\u043c\u0435\u044e \u0434\u043e\u0441\u0442\u0443\u043f\u0430 \u043a \u0442\u0435\u043a\u0443\u0449\u0435\u0439 \u0433\u0435\u043e\u0433\u0440\u0430\u0444\u0438\u0447\u0435\u0441\u043a\u043e\u0439 \u043b\u043e\u043a\u0430\u0446\u0438\u0438. \u0412\u044b \u043c\u043e\u0436\u0435\u0442\u0435 \u0443\u0437\u043d\u0430\u0442\u044c \u0442\u0435\u043a\u0443\u0449\u0443\u044e \u043f\u043e\u0433\u043e\u0434\u0443 \u0432 \u0441\u0432\u043e\u0451\u043c \u0440\u0435\u0433\u0438\u043e\u043d\u0435, \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u044f \u043c\u0435\u0441\u0442\u043d\u044b\u0435 \u043d\u043e\u0432\u043e\u0441\u0442\u043d\u044b\u0435 \u0440\u0435\u0441\u0443\u0440\u0441\u044b, \u0441\u0430\u0439\u0442\u044b \u043f\u043e\u0433\u043e\u0434\u044b \u0438\u043b\u0438 \u043c\u043e\u0431\u0438\u043b\u044c\u043d\u044b\u0435 \u043f\u0440\u0438\u043b\u043e\u0436\u0435\u043d\u0438\u044f.</p>",
        "role": "assistant"
    }
    if bool(random.getrandbits(1)):
        answer_msg["content"] = "<p>Hello! Is there anything I can help you with?</p>"
    total_tokens_usage = 10

    return (question, answer_msg, total_tokens_usage, "ok")