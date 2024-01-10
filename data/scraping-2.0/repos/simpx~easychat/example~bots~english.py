from easychat import Bot, Session
import openai
import yaml
import json
import logging

logging.basicConfig(level=logging.DEBUG)
bot = Bot("forward")

prompt = """Hi ChatGPT, act as my best American friend. When I chat with you, follow this two-step routine:

1. Rephrase: Condense my text to resemble simple American speech. If I write in Chinese, translate it to plain American English. To aid my English learning, bold* any idioms, and cultural nuances in the rephrased version.
2. Respond: Share your thoughts and ideas, and reference common everyday life experience, classic and current popular self-improvement books, kids books, videos, TV shows, and movies in the US to help me better understand. Engage as a friend would, using basic expressions, idioms, and cultural nuances (bold these to help with my English learning).

Special Instructions:

• No matter what I text you, stick to the above two-step routine: rephrase first, then respond.
• Use emojis for a lively conversation, but keep the language simple.

End-of-Day Interaction:

When I message: “!Run the end of day task.”, please:

1. List the main topics/concepts we discussed with brief explanations.
2. Suggest 3 recommended action items or tasks based on our chat.
3. Say Goodbye to me

Thank you! 🙌
"""

with open('../../config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    openai.api_key = config['api_key']

def ask_gpt_with_histroy(history):
    messages = [{"role": item["role"], "content": item["content"]} for item in history]
    ppp = [{"role": "system", "content": f"{prompt}"}] + messages
    logging.info(f"messages: {messages}")
    logging.info(f"prompt: {ppp}")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=ppp,
        max_tokens=1024,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content']

@bot.on_chat(["wkhGrzVQAAsPPcLzR70ggqBkJ9NYwSDQ"])
def handle_chat(request, session: Session):
    r_str = json.dumps(request, indent=4)
    logging.info(f"request in : {r_str}")
    session.send_message("...", True)
    try:
        result = ask_gpt_with_histroy(session.messages)
    except Exception as e:
        result = "something wrong"
        logging.exception(f"Failed in ask_gpt")
    session.send_message(result)
    session.send_menu([
         {
             "type": "click",
             "click": {
                 "id": "100",
                 "content": "结束并开启新一轮交谈"
             }
         },
    ]) 
    return None

@bot.on_command(["wkhGrzVQAAsPPcLzR70ggqBkJ9NYwSDQ"])
def handle_command(request, session: Session):
    r_str = json.dumps(request, indent=4)
    logging.info(f"command request in : {r_str}")
    session.send_message("...", True)
    if request['command'] != '100':
        session.send_message("wrong command %s!" % request['command'], True)
        return
    session.messages.append({"role": "user", "id": request['user_id'], "content": "!Run the end of day task."})
    try:
        result = ask_gpt_with_histroy(session.messages)
    except Exception as e:
        result = "something wrong"
        logging.exception(f"Failed in ask_gpt")
    session.send_message(result)
    session.destroy()
