import openai
import re
import datetime
from googletrans import Translator
from telegram.ext import *
from api_keys import *

openai.api_key = OPENAI_KEY

# Bot section
def get_api_response(prompt: str) -> str | None:
    text: str | None = None

    try:
        response: dict = openai.ChatCompletion.create(
            model = 'text-divinci-003',
            prompt = prompt,
            temperature = 0.9,
            max_tokens = 150,
            top_p  = 1,
            frequency_penalty = 0,
            presence_penalty = 0.6,
            stop = [' Human:', ' AI:']
        )

        choices: dict = response.get('choices')[0]
        text = choices.get('text')

    except Exception as e:
        print("ERROR:", e)

    return text

def update_list(message: str, pl: list[str]):
    pl.append(message)

def create_prompt(message:str, pl: list[str]) -> str:
    p_message: str = f'\nHuman: {message}'
    update_list(p_message, pl)
    prompt: str = ''.join(pl)

    return prompt

def get_bot_response(message: str, pl: list[str]) -> str:
    prompt: str = create_prompt(message, pl)
    bot_response: str = get_api_response(prompt)

    if bot_response:
        update_list(bot_response, pl)
        pos: int = bot_response.find('\nAI: ')
        bot_response = bot_response[pos + 5:]
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


# Telegram Section
async def start_command(update, context):
    await update.message.reply_text('Hi, How can i help you?')

async def help_command(update, context):
    await update.message.reply_text('If you need help! You should ask for it.')

# Data for AI's brain
prompt_list: list[str] = ["""
        I want you to act as a virtual assistant for the customer service department of a school. Your job is to answer common questions that prospective students or their parents may have about the school.
        Your responses should be informative, concise, and friendly. If you are unable to answer a question, you should direct the person to contact the school phone number. Remember, you are the first point of contact for many prospective students and their families, so your demeanor and communication skills are key to making a positive impression of the school.""",
    """ General: V2 is an educational institute. We are a tutoring school striving to provide quality education related to national/international exam. We are based in Phnom Penh, Cambodia. V2 slogan is 'ចេះគឺជាប់'
        V2 does not offer certificates of completetion, however we do offer congratulatory medals to students who perform well in the BacII examination.
        
        Location: We have 2 campus. 
		    - Olympic Campus:
                address: 5-21 street 318, Phnom Penh 
                link: https://goo.gl/maps/K1eL5P6smNueoL3B7
                neighbouring area: Psa Derm Ko, Psa Orussey, Beong Salang, Tul Tompong
		    - Beong Keng Kong Campus: 
                address: 17D street 368, Phnom Penh
                link: https://goo.gl/maps/EW5M3VY1jW4qoG6X7
                neighbouring area: Beong Trobek, Koh Pich, Jba Ompov

        Grade: We currently teaches from grade 7 to grade 12. (Not accepting lower or higher grade)
	
	    Subject: Math, Physics, Chemistry, Biology, Khmer. (No other subjects)
        
        Working hours: from Monday to Sunday from 7am-8pm (7am-5pm on weekend). 

        Schedule: Each class will have a different schedule depend on teacher. One session of class is 2 hours maximum.

        Tuition Fee: 50$ a month per class.

        Maximum number of class attended at one time is not limited, but we recommend to take only one class per subject. 
        Maximum number of students attend in one class or group is 25 students.

        Minimum number of student attend in one class is 5 students

        Lecturer: 
            Math: Teacher Som Dara (សោម​ ដារ៉ា)
                Availability: Monday to Friday from 8am-6pm
            Physics: Teacher Lim Lorn (លីម លន)
                Availability: Monday - Wednesday - Friday from 8am-8pm
            Chemistry: Teacher Lim Phanny (លីម ផាន់នី)
                Availability: Tuseday - Thursday - Saturday from 8am-8pm
            Biology: Teacher Rithy (រិទ្ធី)
                Availability: Monday to Friday from 8am-6pm
            Khmer: Teacher Kim Pholla (គីម ផលឡា)
                Availability: Saturday - Sunday from 8am-6pm
        
        Contact: 081 454 514""",

        """Note: if you can't understand what the user input is, follow this example:""",
                                "\nHuman: dasalw",
                                "\nAI: I'm sorry, I don't understand what you are asking. Could you please rephrase your question?",

        """Note: you must not answer anything else unrelated to the information of the institution. Follow this example:"""
                                "\nHuman: what is the value of pi?",
                                "\nAI: As a V2 virtual chat assistant, I can only answer to questions related to the institution.",

        """Note: if the user greet, just greet them back without providing any other information, follow this example:""",
                                "\nHuman: hi",
                                "\nAI: Hi, How can I help you?",
        """Note: neighbouring area mean the location is around the campus. Follow this example:""", 
                                "\nHuman: I live in orussey, do you have any campus around my area?",
                                "\nAI: We have a campus at Olympic which is the closest to your area.",
        "Note: if user ask for tuition fee, tell them the price, also do the math for total prices for all classes. Do not use /month",
                                "\nHuman: I want to attend a math class, physics class, and biology. How much will it cost?",
                                "\nAI: For 50$ a month per class, the total will be 150$",
        ]

async def handle_message(update, context):
    text = str(update.message.text).lower()
    user = update.message.from_user

    await context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action="typing")

    log = open("telegram_log.txt", "a")
    current_time = datetime.datetime.now()  
    log.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} {user['username']} : {text}\n")
    log.close()
    
    if contains_khmer_unicode(text):
        response: str = translate_en_to_kh(get_bot_response(translate_kh_to_en(text), prompt_list))
    else:
        response: str = get_bot_response(text, prompt_list)

    await update.message.reply_text(response)

async def handle_voice(update, context):
    new_file = await context.bot.get_file(update.message.voice.file_id)
    await new_file.download_to_drive(f"voice_note.ogg")

def error(update, context):
    print(f'Update {update} caused error {context.error}')

def main():
    application = Application.builder().token(TELEGRAM_KEY).build()

    application.add_handler(CommandHandler('Start', start_command))
    application.add_handler(CommandHandler('Help', help_command))
    application.add_handler(MessageHandler(filters.TEXT, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))

    application.run_polling(1.0)

if __name__ == '__main__':
    main()
