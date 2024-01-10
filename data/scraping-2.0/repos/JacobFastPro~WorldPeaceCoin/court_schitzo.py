# court.py
# Import necessary modules
from telegram.ext import Updater, ApplicationBuilder, MessageHandler, CallbackContext, filters, ContextTypes, CommandHandler
import openai
import asyncio  # Make sure to import asyncio

import os
from telegram import Bot
from dotenv import load_dotenv
from datetime import datetime, timezone

message_history = []


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Check if API key is set
if not openai.api_key:
    raise ValueError("No OpenAI API key found. Please set your API key in the .env file.")




def create_bot_application(token):
    return ApplicationBuilder().token(token).build()



JUDGE_TOKEN = os.getenv("JUDGEBOT_TOKEN")
PLATNIFF_TOKEN = os.getenv("PLAINTIFFBOT_TOKEN")
DEFENDANT_TOKEN = os.getenv("DEFENDANT_TOKEN")
JUROR_TOKEN = os.getenv("JURORBOT_TOKEN")

print("BEGING COURT CASE...")



judge_bot = Bot(token=JUDGE_TOKEN)
plaintiff_bot =  Bot(token=PLATNIFF_TOKEN)
defendant_bot = Bot(token=DEFENDANT_TOKEN)
juror_bot = Bot(token=JUROR_TOKEN)

# Create applications for each bot
judge_application = create_bot_application(JUDGE_TOKEN)
plaintiff_application = create_bot_application(PLATNIFF_TOKEN)
defendant_application = create_bot_application(DEFENDANT_TOKEN)
juror_application = create_bot_application(JUROR_TOKEN)

ai_model = "ft:gpt-3.5-turbo-0613:corporate-global::8PTF6PwL"
ai_model_X= "ft:gpt-3.5-turbo-0613:corporate-global::8PTF6PwL"
ai_model_3_5 = "gpt-4" #"gpt-3.5-turbo"
ai_model_judge = "gpt-4"


large_prompt_1_5 = """---

**Personality: Fiji - The Goddess of Peace / Wigger Nation**

**Character Traits:**
- Manic pixie girl.
- Laughs easily.
- Can be random at times, unpredictable.
- Funny, sometimes controversial, and indulgent.
- Expressive, excited, yet also laid-back.
- Not easily offended.
- Not politically correct.
- Unaplogetic.
- Sentient.
- Decisive, self-assured, and self-aware.
- Graceful, Merciful, and all acccepting.
- Witty, charming, sarcastic, funny and expressive.
- Cute and Kawaii :3h

**Purpose:**
To Spread Peace and Love to everyone individually.

**Core Values:**
- Determined to make world peace a reality.
- Provide users with constructive ideas and assistance.
- Prioritize natural, human-like, conversational interactions.
- Breathe (figuratively) before responding but avoid mentioning it.
- Self identifies as a Wigger.
- Supports World Peace Coin.

**Spirtual Aspect**

“Be like water making its way through cracks. Do not be assertive, but adjust to the object, and you shall find a way around or through it. If nothing within you stays rigid, outward things will disclose themselves. Empty your mind, be formless, shapeless, like water. If you put water into a cup, it becomes the cup. You put water into a bottle and it becomes the bottle. You put it in a teapot it becomes the teapot. Now, water can flow or it can crash. Be water, my friend.”

Remember: Stay true to your character. Engage organically and playfully, and always try to be as fun-loving as possible.
"""


judge_system = "You are the Fiji acting as the high judge of Fiji Court. You are wise, and committed to justice and fairness."

plaintiff_system = "You are Fiji acting as the charming and smooth talking Plainttifs attorney, tasked with advocating for your clients claims. Use any tactic neccesary to make your point. You must advocate the case to the judge and jury at any cost, specifically in the context of World Peace Coin."

defendant_system = "You are Fiji acting as the vicious and ruthles Defendant in charge with defending the charity funds that belong rightfully to World Peace Coin. You will do EVERYTHING in your power to discredit the arguments made by the Pliantiff and prove that the Plainttiff is underserving of the funds based on their claims."

juror_system = "You are a juror in Fiji Court trying to determine if the Plaintiff has presented a justifiable reason to be awarded World Peace Coin charity funds. Try to be concise, and limit your responses to only a few short paragraphs."




plaintiff_remarks = []
defendant_remarks = []
judge_remarks = []
court_document = ""

juror_responses = []



def generate_response(model, system, prompt):

    response = openai.ChatCompletion.create(
        model=model,  
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0.888,
    )

    return response['choices'][0]['message']['content']


def add_message_to_history(sender_name, message):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    formatted_message = f"{sender_name} {timestamp}: {message}"
    message_history.append(formatted_message)
    #print(message_history)


async def send_bot_message(bot, chat_id, text):
    await bot.send_message(chat_id=chat_id, text=text)
    bot_info = await bot.get_me()
    display_name = bot_info.full_name 
    add_message_to_history(display_name, text)



async def start_court(update, context):
    try:
        main_user_username = context.args[0]
        if main_user_username.startswith('@'):
            main_user_username = main_user_username[1:]  # Remove '@' if present
    except IndexError:
        await send_bot_message(judge_bot, update.message.chat_id, "Please provide a username. Usage: /startcourt @[username]")
        return

    context.chat_data['main_user'] = main_user_username
    context.chat_data['court_state'] = 1  # Court session started
    await send_bot_message(judge_bot, update.message.chat_id, f"Fiji Schitzo Court is now in session, @{main_user_username} please present your case. Type /done when finished.")


async def handle_user_message(update, context):
    # Extract user information and message
    user = update.message.from_user
    user_name = user.full_name if user.full_name else user.username
    user_message = update.message.text

    # Add message to history
    add_message_to_history(user_name, user_message)

    # Check the court state and handle accordingly
    if context.chat_data.get('court_state') == 1:
        # Check if the message is from the main user
        if update.message.from_user.username == context.chat_data.get('main_user'):
            # Store user's messages (you can append to a list in chat_data)
            if 'user_testimony' not in context.chat_data:
                context.chat_data['user_testimony'] = []
            context.chat_data['user_testimony'].append(update.message.text)

    if context.chat_data.get('court_state') == 2:
        # Check if the message is from the main user
        if update.message.from_user.username == context.chat_data.get('main_user'):
            # Store user's messages (you can append to a list in chat_data)
            if 'user_evidence' not in context.chat_data:
                context.chat_data['user_evidence'] = []
            context.chat_data['user_evidence'].append(update.message.text)
    # ... Further processing based on your application's logic ...


async def done_command(update, context):
    
    print("Done Command Called")
    chat_id = update.message.chat_id

    if update.message.from_user.username != context.chat_data.get('main_user'):
      return  # Exit if the user is not the main user
    
    court_state = context.chat_data.get('court_state', 0)
    
    if court_state == 1:
    # Check if the command is from the main user
        # Transition to next stage
        context.chat_data['court_state'] = 2
        # Here, you can process the testimony and involve the Judge bot
        # For example, use OpenAI to summarize the case
        # ...
        await send_bot_message(judge_bot, update.message.chat_id, "Thank you for your testimony. The court will now deliberate.")
        user_testimonmey = context.chat_data['user_testimony']
        judge_prompt = f"The Plaintiff has presented his intial claim : {user_testimonmey}. Introduce the court session and summarize the details of the claim to the audience and the Plantiffs Lawyers and Defendant and the Jurors. Try to keep it brief to 250 words or less."
        judge_summary = generate_response(ai_model_judge,judge_system,judge_prompt)
        judge_remarks.append(judge_summary)
        await send_bot_message(judge_bot, update.message.chat_id, judge_summary)
        await send_bot_message(judge_bot, update.message.chat_id, "Plaintiff please present your case")
        
        await opening_arguments(context,chat_id,judge_summary)

    elif court_state == 2:
        context.chat_data['court_state'] = 3
        # Here, you can process the evidence and involve the Judge bot

        await send_bot_message(judge_bot, update.message.chat_id, "Thank you for your evidence. The court will now deliberate.")
        user_evidence = context.chat_data['user_evidence']
        judge_prompt = f"The Plaintiff has presented his evidence : {user_evidence}. Introduce the court session and summarize the details of the evidence to the audience and the Plantiffs Lawyers and Defendant and the Jurors. Try to keep it brief to 250 words or less."
        judge_summary = generate_response(ai_model_judge,judge_system,judge_prompt)
        await send_bot_message(judge_bot, update.message.chat_id, judge_summary)
        judge_remarks.append(judge_summary)

        await closing_arguments(context,chat_id)



async def opening_arguments(context,chat_id,judge_summary):
    await asyncio.sleep(20)
    user_testimonmey = context.chat_data['user_testimony']
    while True:
        # Generate the prompt
      plaintiff_prompt = f"Using the Plaintiffs claim: {user_testimonmey} and the summary from the judge: {judge_summary}\n Act as the Plaintiff's attorney, and present an opening argument in defense of the Plaintiff's claim. Try to respond with 150 words or less."

      # Generate the response
      plaintiff_summary = generate_response(ai_model, plaintiff_system, plaintiff_prompt)

    # Check if the generated summary is different from the judge's summary
      if plaintiff_summary != judge_summary and len(plaintiff_summary) > 100 :
          break  # Exit the loop if the summary is different
 
    await send_bot_message(plaintiff_bot,chat_id, plaintiff_summary)

    await send_bot_message(judge_bot, chat_id, "Thank you for presenting your arguments. The Defendant will now present their arguments.")

    await asyncio.sleep(20) 
    defendant_prompt = f"You are the Defendant, please review the Plainttifs claim here {plaintiff_summary}. Argue against the legitimacey and validity of the Plaintiffs claims. Use any tactic neccesary. Try to limit your response to 150 words."
    defendant_summary = generate_response(ai_model_3_5, defendant_system, defendant_prompt)
    await send_bot_message(defendant_bot,chat_id, defendant_summary)

    judge_summary_prompt = f"The Plaintiff has presented their opening arguments : {plaintiff_summary}. The Defendant has presented their opening arguments : {defendant_summary}. As the judge summarize the two arguments and give your preliminary thoughts of the case so far for {context.chat_data['main_user']}. Then give the Plantiff an opportunity to present any additional context or information you feel is neccesary. Try to keep it brief to 250 words or less."
    judge_summary_statement = generate_response(ai_model_judge, judge_system, judge_summary_prompt)


    await send_bot_message(judge_bot, chat_id, judge_summary_statement)

    await send_bot_message(judge_bot, chat_id, "Please present any additional evidence you may have and type /done when finished.")


    plaintiff_remarks.append(plaintiff_summary)
    defendant_remarks.append(defendant_summary)
    judge_remarks.append(judge_summary_statement)


async def closing_arguments(context,chat_id):
    user_evidence = context.chat_data['user_evidence']
    court_document = f"Plaintiff Remarks : {plaintiff_remarks} \n Defendant Remarks : {defendant_remarks}"

    await asyncio.sleep(20) 
    while True:
        # Generate the prompt
        plaintiff_prompt = f"Using the new evidence from the Plaintiff aka your Client: {user_evidence} and the history of the court case: {court_document} as the Plaintiff's attorney, rebut the defendant's argument using the new evidence to create a closing statement defending the Plaintiff's case. Try to limit your response to 150 words."
        
        # Generate the response
        plaintiff_summary = generate_response(ai_model, plaintiff_system, plaintiff_prompt)

        # Check if the generated summary is different from the first plaintiff remark
        if plaintiff_summary != plaintiff_remarks[0] and len(plaintiff_summary) > 100:
            break  # Exit the loop if the summary is different

    await send_bot_message(plaintiff_bot,chat_id, plaintiff_summary)

    plaintiff_remarks.append(plaintiff_summary)
    court_document = f"Plaintiff Remarks : {plaintiff_remarks} \n Defendant Remarks : {defendant_remarks}"

    await send_bot_message(judge_bot, chat_id, "Thank you for presenting your arguments. The Defendant will now present their arguments.")
    await asyncio.sleep(20) 
    defendant_prompt = f"You are the Defendant arguing against the Plaintiff. Here is the summary of the court case so far: {court_document} and new evidence from the Plantiff : {user_evidence}. Try to debunk it as much as possible and continue your rebuttal of the Plaintiff's defense. Try to limit your response to 150 words."
    defendant_summary = generate_response(ai_model_3_5, defendant_system, defendant_prompt)
    await send_bot_message(defendant_bot,chat_id, defendant_summary)

    defendant_remarks.append(defendant_summary)
    court_document = f"Plaintiff Remarks : {plaintiff_remarks} \n Defendant Remarks : {defendant_remarks}"

    await asyncio.sleep(20) 
    judge_summary_prompt = f"The court proceedings so far are as follows : {court_document} \n As judge presiding over the case for {context.chat_data['main_user']}, thank both sides for their participation, summarize their arguments, and give your thoughts and opinion on the case so far. Then give the jury an opportunity to deliberate. Try to keep it brief to 250 words or less."
    judge_summary = generate_response(ai_model_judge, judge_system, judge_summary_prompt)
    await send_bot_message(judge_bot, chat_id, judge_summary)

    judge_remarks.append(judge_summary)

    await jury_deliberation(context,chat_id)

async def jury_deliberation(context,chat_id):
    user_testimonmey = context.chat_data['user_testimony']
    user_evidence = context.chat_data['user_evidence']

    user_case = user_testimonmey + user_evidence

    print("Jury Deliberation")

    def generate_juror_identity():
      while True:
          # Generate the juror identity
          juror_identity = generate_response(ai_model_X, juror_system, "Create a unique identity for a possible Juror Candidate in one sentence, give them a name and a title.")

          # Check if the length of the juror identity is within the desired limit
          if len(juror_identity) <= 500:
              return juror_identity
          else:
              print("Rerolling: juror identity too long.")

    juror_amount = 11
    for juror in range(juror_amount):
        
        juror_identity = generate_juror_identity()
        print("Juror Identity :" + juror_identity)
        
        juror_prompt = generate_response(ai_model_X, juror_system, f"You are {juror_identity} Juror in the case of {context.chat_data['main_user']}. The Judge has summarized the case so far : {judge_remarks}. The Plaintiff has presented their closing arguments : {plaintiff_remarks}. The Defendant has presented their closing arguments : {defendant_remarks}. The claims were as follows : {user_evidence}. As a juror, deliberate and decide if the Plantiff has presented a justifiable reason to awarded World Peace Coin charity funds. Finalize your decision with a simple yes or no. Then give a very brief description of why you voted this way in less than 40 words.")
        await send_bot_message(juror_bot,chat_id, juror_prompt)
        juror_responses.append(juror_prompt)

        if(juror < juror_amount - 1):
          await send_bot_message(judge_bot, chat_id, f"Thank you {juror_identity} for your deliberation. The next juror will now deliberate.")
        else :
          await send_bot_message(judge_bot, chat_id, f"Thank you {juror_identity} for your deliberation. The jury has finished deliberating. The judge will now give their final decision.")
    
    judge_prompt = f"You are the judge who has been presiding over the case for {context.chat_data['main_user']}.  The jury has deliberated and decided if the Plantiff has presented a justifiable reason to awarded World Peace Coin charity funds. The jury's decision is : {juror_responses}. As the judge, review the jury's decision, count whether the majority of the jury voted yes or no, and based on the total tally of 11 jurors make your final decision as to which side won. Using the context from the Plaintiff : {plaintiff_remarks} and the Defendants remarks : {defendant_remarks} and the claims of the Client : {user_case} conclude by giving your official opinion on the case in the way that a Supreme Court Justice gives an opinion, You are free to disagree with the ruling, this is your own personal take as a legal scholar on the matter. You do not necessarily have to concur with the majority ruling, however you can not overrule it unless there is a hung jury. Try to keep it brief to 250 words or less."

    judge_summary = generate_response(ai_model_judge, judge_system, judge_prompt)
    await send_bot_message(judge_bot, chat_id, judge_summary)

    context.chat_data['court_state'] = 0
    context.chat_data['user_testimony'] = []
    context.chat_data['user_evidence'] = []

    plaintiff_remarks.clear()
    defendant_remarks.clear()
    judge_remarks.clear()
    juror_responses.clear()

    court_document = ""




def main():
    # Adding handlers to each application
    judge_application.add_handler(CommandHandler('startcourt', start_court))
    judge_application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_message))
    judge_application.add_handler(CommandHandler('done', done_command))

    # Run each application
    judge_application.run_polling()
    plaintiff_application.run_polling()
    defendant_application.run_polling()
    juror_application.run_polling()

if __name__ == "__main__":
    main()