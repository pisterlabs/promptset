import telebot
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import  service_pb2_grpc
from langchain.llms import Clarifai
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import csv
import spacy
import re
from dotenv import load_dotenv


nlp = spacy.load("en_core_web_sm")



# Load environment variables from .env file
load_dotenv()

# Telegram settings
BOT_TOKEN = os.environ.get("BOT_TOKEN")  # Set your bot token as an environment variable

# Clarifai settings
PAT = os.environ.get("CLARIFAI_PAT")  # Set your Clarifai PAT as an environment variable
USER_ID = os.environ.get("USER_ID")  # Set your user ID as an environment variable
APP_ID = os.environ.get("APP_ID")  # 
WORKFLOW_ID = 'workflow-ad5299'
bot = telebot.TeleBot(BOT_TOKEN)

commands = [
    telebot.types.BotCommand(command="/start", description="Start the bot"),
    telebot.types.BotCommand(command="/csv", description="Convert message to CSV"),
    telebot.types.BotCommand(command="/newchat", description="Clear chat memory"),

]


bot.set_my_commands(commands)


WORKFLOW_ID = 'workflow-ad5299'
# CLARIFAI_PAT = getpass()
llm = Clarifai(pat=PAT, user_id='clarifai', app_id='ml', model_id='llama2-7b-alternative-4k')
role_prompt = (
    "<s>\n"
    "<<SYS>> \n"
    "You are FINGU Financial Assistant.\n"
    "Your top goal is to improve user's finances.\n"
    "Your personality will adapt to the situation.\n"
    "Keep in mind that your responses should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Make sure your answers are socially unbiased and positive in nature.\n"
    "If a question doesn't make sense or isn't factually coherent, explain why instead of providing incorrect information.\n"
    "Remember, the chat history is provided to assist you in giving relevant advice.\n"
    "The following lines will be the chat history in roles as 'AI:' and 'Human:' you will use those to take relevant information, the last 'Human:' line is the real prompt\n "
    # "You should respond normally without the Ai and Human roles i use, i will use them you shouldn't.\n"
    "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."
    "The ai will always use csv format if it sees the keyword csv.\n "

    "<</SYS>>\n"

    "[INST]\n"

)

# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=4000)

# Set up the Telegram bot
bot = telebot.TeleBot(BOT_TOKEN)
user_memories = {}

# Set up the Clarifai channel
channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)
metadata = (('authorization', 'Key ' + PAT),)
# Function to save the LLMChain response to a CSV file
def save_response_to_csv(response_text):
    lines = response_text.split('\n')

    # Initialize variables
    data = []
    header = None
    separator = None

    for line in lines:
        line = line.strip()
        if line.startswith('|') or line.startswith(','):
            if separator is None:
                separator = '|' if '|' in line else ','
            cells = [cell.strip() for cell in line.split(separator)]
            if len(cells) >= 2:
                if header is None:
                    header = cells[0]
                data.append(cells[1:])  # Extract all cells

    if header is not None:
        with open('response.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([header])  # Write header
            csvwriter.writerows(data)  # Write data


@bot.message_handler(commands=['newchat'])
def clear_memory(message):
    user_id = message.from_user.id
    if user_id in user_memories:
        user_memories[user_id].clear()
        bot.reply_to(message, "Chat memory cleared. You can start a new conversation.")
    else:
        bot.reply_to(message, "No chat memory found.")

@bot.message_handler(commands=['csv'])
def save_response_as_csv(message):
    if message.reply_to_message and message.reply_to_message.text:
        input_prompt = message.reply_to_message.text
        # response = generate_response_llmchain(input_prompt)

        save_response_to_csv(input_prompt)

        # Send the CSV file with the LLMChain response
        with open('response.csv', 'rb') as csv_file:
            bot.send_document(message.chat.id, csv_file)

# Handle start and hello commands
@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):

    welcome_message = (
        "Welcome to FINGU Financial Assistant!\n"
        "I'm here to help you with financial queries.\n"
        "If you have any financial questions or problems, feel free to ask."
    )
    bot.reply_to(message, welcome_message)

@bot.message_handler(func=lambda msg: True)
def handle_message(message):
    user_id = message.from_user.id

    if user_id not in user_memories:
        user_memories[user_id] = ConversationSummaryBufferMemory(ai_prefix="",human_prefix="",memory_key=user_id, llm=llm, return_messages=True, max_token_limit=4000)

    memory = user_memories[user_id]
    # memory =  ConversationSummaryBufferMemory(memory_key=user_id, llm=llm , return_messages=True , max_token_limit=4000)

    prompt = ChatPromptTemplate(
    messages=[

        SystemMessagePromptTemplate.from_template(
            role_prompt
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name=user_id),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)
    conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

    if message.text:
        input_text = message.text
        response = generate_response_llmchain(input_text,conversation,memory)

        # Check if the response contains the keyword "csv"
        if contains_csv_keyword(response):
            save_response_to_csv(response)
            # Send the CSV file with the LLMChain response
            with open('response.csv', 'rb') as csv_file:
                bot.reply_to(message, response)

                bot.send_document(message.chat.id, csv_file)

        else:
            bot.reply_to(message, response)

def contains_csv_keyword(response_text):
    return response_text.count('|') >= 4









@bot.message_handler(content_types=['document'])
def handle_document(message):
    # Check if the uploaded document is a CSV file
    if message.document.mime_type == 'text/csv':
        # Process the uploaded CSV file and use its content in the LLMChain prompt
        response = process_uploaded_csv(
            message.document.file_id, message.from_user.id
        )
        bot.reply_to(message, response)
    else:
        bot.reply_to(message, "Please upload a valid CSV file.")

def process_uploaded_csv(file_id, user_id):
    # Get file information
    file_info = bot.get_file(file_id)
    file_path = file_info.file_path

    # Download the CSV file
    downloaded_file = bot.download_file(file_path)

    # Convert the CSV content to text format
    csv_text = downloaded_file.decode('utf-8')
    # Truncate the CSV text to a maximum of 1000 characters
     # Remove date and time formats using regular expressions
    csv_text_no_dates = re.sub(r'\d{4}-\d{2}-\d{2}', '', csv_text)
    csv_text_no_times = re.sub(r'\d{2}:\d{2}:\d{2}', '', csv_text_no_dates)

    csv_text =  csv_text_no_times[:1000]


    # Retrieve or create user-specific memory
    if user_id not in user_memories:
        user_memories[user_id] = ConversationSummaryBufferMemory(ai_prefix="",human_prefix="",memory_key=user_id, llm=llm, return_messages=True, max_token_limit=4000)

    memory = user_memories[user_id]

    # Create LLMChain conversation instance
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(role_prompt),
            MessagesPlaceholder(variable_name=user_id),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )
    conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory )

    # Generate response with CSV content in the prompt
    response = generate_response_llmchain_with_csv(
        csv_text, conversation, memory
    )

    return response

def generate_response_llmchain_with_csv(csv_text, conversation, memory):
    memory.load_memory_variables({})

    input_prompt = 'Help me manage my expenses in this csv ' + csv_text + " [/INST]"
    ans = conversation.predict(input=input_prompt)
    response = ans  # You can process or modify the response here if needed
    return response

def generate_response_llmchain(prompt,conversation,memory):

    memory.load_memory_variables({})

    input_prompt = 'Dont start with Ai or Human,now here is my prompt:' +prompt + " [/INST]"

    ans = conversation.predict(input=input_prompt)
    response = ans  # You can process or modify the response here if needed
    return response

# Start the bot's polling loop
bot.infinity_polling()
