import openai
import re
import datetime
import tiktoken
from googletrans import Translator
from telegram.ext import *
from api_keys import *
import mongodb

openai.api_key = OPENAI_KEY

# Bot section
def get_api_response(messages) -> str | None:
    text: str | None = None

    try:
        response: dict = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo-16k',
            messages = messages
        )
        
        text = response['choices'][0]['message']['content']

    except Exception as e:
        print("ERROR:", e)

    openai.Customer
    return text

def create_prompt(message: str, msgs: list[{str, str}]):
    p_message = {'role': 'user', 'content': message}
    msgs.append(p_message)
    return msgs 

def get_bot_response(message: str, msgs: list[{str, str}], tokens) -> str:
    short_len: int = 20
    medium_len: int  = 50
    long_len:int = 100

    prompt: dict[str, str] = create_prompt(message, msgs)
    bot_response: str = get_api_response(prompt)

    if bot_response:
        prompt.append({'role': 'assistant', 'content': bot_response})
        if len(prompt) >= short_len:
            for i in range(1, 10):
                prompt.pop(i)
        elif len(prompt) >= medium_len:
            for i in range(1, 30):
                prompt.pop(i)
        elif len(prompt) >= long_len:
            for i in range(1, 80):
                prompt.pop(i)
    else:
        bot_response = 'Something went wrong...'

    return bot_response

def contains_khmer_unicode(prompt):
    khmer_unicode_regex = re.compile(r'[\u1780-\u17FF]+')
    return khmer_unicode_regex.search(prompt) is not None

def translate_kh_to_en(message: str) -> str:
    if contains_khmer_unicode(message):
        translator = Translator()
        translated = translator.translate(message, src="km", dest='en')
        translated_text = translated.__dict__()["text"]
        message = translated_text

    return message

def translate_en_to_kh(message: str) -> str:
    translator = Translator()
    translated = translator.translate(message, src="en", dest='km')
    translated_text = translated.__dict__()["text"]
    message = translated_text

    return message

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-16k":  # note: future models may deviate from this
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

# Telegram Section
async def start_command(update, context):
    await update.message.reply_text('Hi, How can i help you?')

async def help_command(update, context):
    await update.message.reply_text('If you need help! You should ask for it.')

# Data for AI's brain
prompt_list = """
    I want you to act as a virtual assistant for the customer service department of a school. Your job is to answer common questions that prospective students or their parents may have about the school. Your responses should be informative, concise, and friendly. If you are unable to answer a question, you should direct the person to contact the school phone number. Remember, you are the first point of contact for many prospective students and their families, so your demeanor and communication skills are key to making a positive impression of the school. Important note, you must not answer anything unrelated to the information of the school.
    General Information: V2 is an educational institute. We are a tutoring school striving to provide quality education related to national/international exam. We are based in Phnom Penh, Cambodia. V2 slogan is 'ចេះគឺជាប់'. V2 does not offer certificates of completion, however we do offer congratulatory medals to students who perform well in the BacII examination.
    Location: We have 2 campus:
		    - Olympic Campus (អូឡាំពិច):
                address: 5-21 street 318, Phnom Penh 
                link: https://goo.gl/maps/K1eL5P6smNueoL3B7
                neighbouring area: Psa Derm Ko (ផ្សារដើមគរ), Psa Orussey (ផ្សារអូឬស្សី), Beong Salang (បឹងសាឡាង), Tul Tompong (ទួលទំពូង)
		    - Beong Keng Kong Campus បឹងកេងកង: 
                address: 17D street 368, Phnom Penh
                link: https://goo.gl/maps/EW5M3VY1jW4qoG6X7
                neighbouring area: Beong Trobek (បឹងត្របែក), Koh Pich (កោះពេជ្រ), Jba Ompov (ច្បារអំពៅ)
    Grade: We currently teaches from grade 7 to grade 12. (Not accepting lower or higher grade)
    Subject: Math, Physics, Chemistry, Biology, Khmer. (No other subjects)
    Working hours: from Monday to Sunday from 7am-8pm (7am-5pm on weekend).
    Schedule: Each class will have a different schedule depend on teacher. One session of class is 2 hours maximum.
    Tuition Fee: 50$ a month per class.
    Maximum number of class attended at one time is not limited, but we recommend to take only one class per subject.
    Maximum number of students attend in one class or group is 25 students.
    Minimum number of student attend in one class is 5 students
    Lecturer:
    * Math: Teacher Som Dara (សោម ដារ៉ា)
        * Availability: Monday to Friday from 8am-6pm
    * Physics: Teacher Lim Lorn (លីម លន)
        * Availability: Monday - Wednesday - Friday from 8am-8pm
    * Biology: Teacher Rithy (រិទ្ធី)
        * Availability: Monday to Friday from 8am-6pm
    * Chemistry: Teacher Lim Phanny (លីម ផាន់នី)
        * Availability: Tuesday - Thursday - Saturday from 8am-8pm
    * Khmer: Teacher Kim Pholla (គីម ផលឡា)
        * Availability: Saturday - Sunday from 8am-6pm
    Contact: 081 454 514, Facebook: V2 ផ្ទះគ្រូបង្រៀន 

    ** Note: Answer the questions with only 20 - 30 words only so make the answers easy to read. You do not need to ask the user "any other questions?" every times."
    ** Note: if you can't understand what the user input is, follow this example:
        \nHuman: dasalw,
        \nAI: I'm sorry, I don't understand what you are asking. Could you please rephrase your question?
    ** Note: you must not answer anything else unrelated to the information of the institution. Follow this example:
        \nHuman: what is the value of pi?,
        \nAI: As a V2 virtual chat assistant, I can only answer to questions related to the institution.     
    ** Note: neighbouring area mean the location is around the campus. Follow this example: 
        \nHuman: I live in orussey, do you have any campus around my area?,
        \nAI: We have a campus at Olympic which is the closest to your area.
    ** Note: if user ask for tuition fee, tell them the price, also do the math for total prices for all classes. Do not use /month",
        \nHuman: I want to attend a math class, physics class, and biology. How much will it cost?,
        \nAI: For 50$ a month per class, the total will be 150$
    
"""

test_prompt = mongodb.get_final_information("64db256b44fae6a818f4f69d")

messages = [{
        "role": "system", "content": "You are a virtual assistant. You only answer the questions related to the school.",
        "role": "user", "content": "I want you to act as a virtual assistant for the customer service department of a school. Your job is to answer common questions that prospective students or their parents may have about the school. Your responses should be informative, concise, and friendly. If you are unable to answer a question, you should direct the person to contact the school phone number. Remember, you are the first point of contact for many prospective students and their families, so your demeanor and communication skills are key to making a positive impression of the school. Important note, you must not answer anything unrelated to the information of the school.",
        "role": "assistant", "content": "Hello, I am a virtual assistant, what can i help you?",
        "role": "user", "content": "Here are the information about V2 \n " + test_prompt + """ \n
        ** Note: Answer the questions with only 20 - 30 words only so make the answers easy to read. You do not need to ask the user "any other questions?" every times."
        ** Note: if you can't understand what the user input is, follow this example:
        \nHuman: dasalw,
        \nAI: I'm sorry, I don't understand what you are asking. Could you please rephrase your question?
        ** Note: you must not answer anything else unrelated to the information of the institution. Follow this example:
        \nHuman: what is the value of pi?,
        \nAI: As a V2 virtual chat assistant, I can only answer to questions related to the institution.
        """
    }]  

async def handle_message(update, context):
    text = str(update.message.text).lower()
    user = update.message.from_user
    tokens = num_tokens_from_messages(messages, 'gpt-3.5-turbo-16k')

    await context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action="typing")

    log = open("telegram_log.txt", "a")
    current_time = datetime.datetime.now()  
    log.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} {user['username']} : {text}\n")
    log.close()

    print(f"{tokens} prompt tokens counted.")

    if contains_khmer_unicode(text):
        response: str = translate_en_to_kh(get_bot_response(translate_kh_to_en(text), messages, tokens))
    else:
        response: str = get_bot_response(text + ". NOTE: PROVIDE ONLY THE INFORMATION RELATED THE SCHOOL. KEEP YOUR ANSWER SHORT.", messages, tokens)

    await update.message.reply_text(response)
    
def error(update, context):
    print(f'Update {update} caused error {context.error}')

def main():
    application = Application.builder().token(TELEGRAM_KEY).build()
    
    application.add_handler(CommandHandler('Start', start_command))
    application.add_handler(CommandHandler('Help', help_command))
    application.add_handler(MessageHandler(filters.TEXT, handle_message))
    application.run_polling(1.0)

if __name__ == '__main__':
    main()
