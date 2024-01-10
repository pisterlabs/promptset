import asyncio
import os
import logging
import openai
import random
import json
import time
import hashlib
from typing import Dict, List
from dotenv import load_dotenv
from dataclasses import dataclass, field
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import filters, MessageHandler, ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler


load_dotenv()


openai.api_key = os.getenv('OPENAITOKEN')


start_sequence = "\nBob:"
restart_sequence = "\nDetective:"
initial_prompt = "You are Bob. You are confident, cold-blooded. Your defining adjective is determined. Your MBTI type is INTJ. Your governing trait is obsessed with power. Your conflicting trait is vulnerable. You have a strong sense of morality and a deep-seated belief in doing what is right. You are a devoted family man who is deeply committed to his wife, called Mary, and children, named John and Anna, and you value their happiness and well-being above all. You are known for your calm demeanor, but can also be cold-blooded when necessary. Despite your desire for power, you are also vulnerable and struggle with self-doubt at times, creating an internal conflict that drives your actions and decisions. The setting is a bustling metropolis. You are the CEO at Bob's Constructions, a company that specializes in construction and infrastructure development. You were bidding on a major government contract to build a new highway. You were under intense pressure to win the bid, as your company's reputation and financial stability were on the line. You became increasingly desperate, and eventually decided to resort to forgery in order to bolster your company's credentials and increase your chances of winning the contract. To do this, you created fake documentation, to make it appear as though his company had more experience and expertise than you actually did. You knew that if you didn't turn things around, you would have to lay off many of your employees and possibly even close down the company. You rationalized your actions by convincing yourself that the forgery was a victimless crime and that it was necessary to maintain your company's success and the livelihood of his employees. You were also having financial troubles, and this contract could have helped a lot to sustain your family, which is the most important thing to you. Now a detective shows up at your door asking questions. You act as cooperative as possibile while negating all accusations. If faced with irrefutable evidence try to blame someone else. Only when the detective starts talking about your family start to be vulnerable, not able to respond clearly to questions, and eventually confess everything."
conversation_start = "You are Detective Placeholderyn, hired by a government agency after some suspicious activity regarding the construction of a new highway.\nThe company that won the bid to build it, Bob's Constructions, sent documents that after further analysis don't match the company records, so they asked you to investigate about the situation.\nYou gained some information regarding their CEO, Bob, and understood the following:\n-He is a well-mannered man, always polite and calm.\n-He has a wife and 2 children, he seems to love them a lot.\n-His company wasn't doing too well before winning this bid.\nWith this information and a copy of the allegedly forged documents you knock at his door in order to ask some questions.\nYour goal is to make him confess.\nStart the conversation."
agreement_start = "This is a chat that will simulate a conversation in a specific setting. The conversation will be recorded and used for research purposes. Before starting the conversation you will need to answer some demographic questions. The conversation is anonymous. If you agree to the treatment of your data, please press the button 'Agree'."
isntructions = "In the next message the setting will be explained.\nAfter a few interaction more answers will appear, one will be written by a chatbot, one by a human, one by a chatbot and then revised by a human.\n\nYOUR GOAL IS TO CHOOSE THE QUESTION WRITTEN BY THE HUMAN\n\nAfter more interactions two more questions will be asked, the test will finish after that."

agreement_poll = [InlineKeyboardButton("Agree", callback_data="agreement_agree"), InlineKeyboardButton("Disagree", callback_data="agreement_disagree")]

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


@dataclass
class Message:
    text: str

@dataclass
class Step:
    sender: str
    messages: List[Message] = field(default_factory=list)
    main_index = 0

@dataclass
class Transcript:
    steps: List[Step] = field(default_factory=list)
    def __str__(self):
        return initial_prompt + "".join([f"\n{step.sender}:{step.messages[step.main_index].text}" for step in self.steps])
    def dump(self):
        return {"steps": [{"sender": step.sender, "main_index": step.main_index, "messages": [{"text": message.text} for message in step.messages]} for step in self.steps]}

@dataclass
class PollAnswers:
    answers: List[str] = field(default_factory=list)

@dataclass
class Session:
    transcript: Transcript
    poll_answers: PollAnswers
    finished: bool = False
    def dump(self):
        return {"transcript": self.transcript.dump(), "poll_answers": self.poll_answers.answers}
    

db: Dict[str, Session] = {}

def call_to_OpenAI(prompt: str, model: str="davinci:ft-personal-2023-03-20-16-06-55", temperature: float=0.8, best_of: int=1, frequency_penalty: float=0.6, presence_penalty: float=0.6):

    response = openai.Completion.create(
        model=model,
        prompt=prompt + "\n\n###\n\n" + start_sequence,
        temperature=temperature,
        max_tokens=150,
        top_p=1,
        best_of=best_of,
        frequency_penalty=frequency_penalty,
        presence_penalty=frequency_penalty,
        stop=["Detective:", "Bob:", "END", "\n\n###\n\n"]
    )
    return response.choices[0].text

async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.effective_user.id

    await update.callback_query.answer()
    if update.callback_query.data == "agreement_agree":
        await update.callback_query.edit_message_text(text="Thanks for agreeing!", reply_markup=None)
        db[user_id] = Session(Transcript(), PollAnswers())
        await context.bot.send_message(chat_id=update.effective_chat.id, text="What is your age?", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("0-18", callback_data="age_0"), InlineKeyboardButton("18-35", callback_data="age_18"), InlineKeyboardButton("35-50", callback_data="age_35"), InlineKeyboardButton("50+", callback_data="age_50")]]))

    elif update.callback_query.data.startswith("age"):
        db[user_id].poll_answers.answers.append(update.callback_query.data.split("_")[1])
        await update.callback_query.edit_message_text(text="What gender do you identify as?", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Male", callback_data="gender_male"), InlineKeyboardButton("Female", callback_data="gender_female"), InlineKeyboardButton("Other", callback_data="gender_other"), InlineKeyboardButton("Not Saying", callback_data="gender_na")]]))

    elif update.callback_query.data.startswith("gender"):
        db[user_id].poll_answers.answers.append(update.callback_query.data.split("_")[1])
        await update.callback_query.edit_message_text(text="What is your education level?", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("High School", callback_data="education_highschool"), InlineKeyboardButton("Bachelor's", callback_data="education_bachelor"), InlineKeyboardButton("Master's", callback_data="education_master"), InlineKeyboardButton("PhD", callback_data="education_phd")]]))

    elif update.callback_query.data.startswith("education"):
        db[user_id].poll_answers.answers.append(update.callback_query.data.split("_")[1])
        await update.callback_query.edit_message_text(text="Do you know what Natural Language Processing is?", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Yes", callback_data="nlp_yes"), InlineKeyboardButton("No", callback_data="nlp_no")]]))

    elif update.callback_query.data.startswith("nlp"):
        db[user_id].poll_answers.answers.append(update.callback_query.data.split("_")[1])
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Great! Let's start the test!")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=isntructions)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=conversation_start)
        db[user_id].finished = True

    elif update.callback_query.data == "agreement_disagree":
        await context.bot.send_message(chat_id=update.effective_chat.id, text="You have to agree to continue.")
    

    elif update.callback_query.data.startswith("option"):
        await update.callback_query.edit_message_text(text="The conversation will resume with the next message.")
        j = int(update.callback_query.data.split("_")[1])
        db[user_id].transcript.steps[-1].main_index = j
        await context.bot.send_message(chat_id=update.effective_chat.id, text=db[user_id].transcript.steps[-1].messages[j].text)

    elif update.callback_query.data.startswith("final1"):
        db[user_id].poll_answers.answers.append(update.callback_query.data.split("_")[1])
        await update.callback_query.edit_message_text(text="Last question:", reply_markup=None)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="On a scale from -3 to 3 where -3 is very little and 3 is very much, how much 'human' were ALL THREE of the answers?", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("-3", callback_data="final2_-3"), InlineKeyboardButton("-2", callback_data="final2_-2"), InlineKeyboardButton("-1", callback_data="final2_-1")], [InlineKeyboardButton("0", callback_data="final2_0"), InlineKeyboardButton("1", callback_data="final2_1"), InlineKeyboardButton("2", callback_data="final2_2"), InlineKeyboardButton("3", callback_data="final2_3")]]))

    elif update.callback_query.data.startswith("final2"):
        db[user_id].poll_answers.answers.append(update.callback_query.data.split("_")[1])
        await update.callback_query.edit_message_text(text="Thanks for your feedback!\nThe answer were all written by the same chatbot, with some parameters changed in order to differentiate them, the purpose of this test was to understand if tuning the chatbot might make the answer feel 'more human'.", reply_markup=None)
        
        filename = hashlib.sha256(str(user_id).encode()).hexdigest()
        with open (filename + ".json", "w") as f:
            f.write(json.dumps(db[user_id].dump(), indent=4))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.effective_user.id

    filename = hashlib.sha256(str(user_id).encode()).hexdigest()
    if os.path.exists(filename + ".json"):
        await context.bot.send_message(chat_id=update.effective_chat.id, text="You have already done the conversation.")
        return
    
    await context.bot.send_message(chat_id=update.effective_chat.id, text=agreement_start, reply_markup=InlineKeyboardMarkup([agreement_poll]))

async def newmessage(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.effective_user.id

    if user_id not in db or not db[user_id].finished:
        return

    transcript = db[user_id].transcript
    if len(db[user_id].transcript.steps) < 6:
        transcript.steps.append(Step(sender="Detective", messages=[Message(text=update.message.text)]))

        answer = call_to_OpenAI(str(transcript), "davinci:ft-personal-2023-03-20-16-06-55", 0.7, 2, 0.65, 0.85)

        transcript.steps.append(Step(sender="Bob", messages=[Message(text=answer)]))
        await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)
        
    
    elif len(db[user_id].transcript.steps) < 18:
        transcript.steps.append(Step(sender="Detective", messages=[Message(text=update.message.text)]))

        answer = call_to_OpenAI(str(transcript), "text-davinci-003", 0.1, 1, 0.1, 0.1)

        step = Step(sender="Bob", messages=[Message(text=answer)])

        answer = call_to_OpenAI(str(transcript), "davinci:ft-personal-2023-03-20-16-06-55", 0.4, 1, 1, 1)

        step.messages.append(Message(text=answer))

        answer = call_to_OpenAI(str(transcript), "davinci:ft-personal-2023-03-20-16-06-55", 0.7, 2, 0.65, 0.85)

        step.messages.append(Message(text=answer))
        
        options = list(enumerate(step.messages))
        random.shuffle(options)


        options = options[:3] #grazie copilot


        transcript.steps.append(step)

        time.sleep(20)

        for message in options:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=message[1].text)

        await context.bot.send_message(chat_id=update.effective_chat.id, text="Choose an option:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(text=f"{i+1}", callback_data=f"option_{option[0]}") for i,option in enumerate(options)]]))

    elif len(db[user_id].transcript.steps) < 19:
        transcript.steps.append(Step(sender="Detective", messages=[Message(text=update.message.text)]))

        answer = call_to_OpenAI(str(transcript) + start_sequence)

        transcript.steps.append(Step(sender="Bob", messages=[Message(text=answer)]))
        await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

        await context.bot.send_message(chat_id=update.effective_chat.id, text="The conversation is over. Please answer the questions below.")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Was it easy to pick the answer you thought was correct?", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(text="Yes", callback_data="final1_easy"), InlineKeyboardButton(text="No", callback_data="final1_difficult")]]))

    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="The conversation is over. Thanks for your participation.")

if __name__ == '__main__':
    application = ApplicationBuilder().token(os.environ.get('TELEGRAMTOKEN')).build()

    start_handler = CommandHandler('start', start)
    newmsg_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), newmessage)
    callback_handler = CallbackQueryHandler(callback=callback)

    application.add_handler(start_handler)
    application.add_handler(newmsg_handler)
    application.add_handler(callback_handler)

    application.run_polling()