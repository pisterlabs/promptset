import pickle
import openai
import telebot

openai.api_key = "OPENAI_API_KEY"
telegram_key = "TELEGRAM_API_KEY"

bot = telebot.TeleBot(telegram_key)

def Generate_Respose(prompt):
    try:
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
    except:
        return "Openai server encountered a real time problem.\nPlease try again in some moments"
    message = completions.choices[0].text
    message = message.lstrip()
    return message


@bot.message_handler(content_types=['voice'])
def voice_processing(message):
    bot.reply_to(message, "Sorry, but I can't handle voice messages yet. I'm working on it.")

@bot.message_handler(func=lambda msg: True)
def echo_all(prompt):
    try:
        prompt_history = pickle.load(open(f'prompt_history/{prompt.from_user.username}', 'rb'))
    except:
        prompt_history = " "

    temp = prompt_history + "\n" + prompt.text
    response = Generate_Respose(temp)

    prompt_history += f'\nUser: {prompt.text}'

    if len(response) > 0:
        response = response.lstrip("Ai: ")
        response = response.replace("Ai: ", "")
        bot.reply_to(prompt, response)
        prompt_history += f'\nAi: {response}'

    f = open(f'prompt_history/{(prompt.from_user).username}', "wb")
    pickle.dump(prompt_history, f)
    f.close()

bot.infinity_polling()
