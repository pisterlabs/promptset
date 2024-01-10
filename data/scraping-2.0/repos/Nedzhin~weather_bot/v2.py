import logging
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
import requests
import json
import constants

from openai import OpenAI


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# setting up the OpenAI API key
client = OpenAI(
  api_key=constants.OPEN_API_KEY,
)


async def get_current_weather(location):
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"
    API_KEY = constants.OPENWEATHERMAP_KEY
    url= BASE_URL + 'appid=' + API_KEY +'&q=' + location

    def kelvin_to_celsius(kelvin):
        celsius = kelvin - 273.15

        return celsius

    response = requests.get(url)
    response = response.json()
    print(response)
    temp_kelvin = response['main']['temp']
    #temp_kelvin_feels_like = response['main']['feels_like']

    temp_celsius = kelvin_to_celsius(temp_kelvin)
    #temp_celsius_feels_like = kelvin_to_celsius(temp_kelvin_feels_like)

    weather_info = {
        'location': location,
        'temperature': temp_celsius,
        'unit': 'celsius'
    }
    return json.dumps(weather_info)

#Function to generate the OpenAi response to the Doctors entered information
async def generate_openai_response(question: str) -> str:

    messages = [{"role": "user", "content": question}]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or choose another model
        messages=messages,
        tools= tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        # Step 3: call the function
        
        available_functions = {
            "get_current_weather": get_current_weather,
        }  
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = await function_to_call(
                location=function_args.get("location")
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
            second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            )  # get a new response from the model where it can see the function response
            return str(second_response.choices[0].message.content)
    
    return str(response_message.content)



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the conversation and introduces itself."""

    await context.bot.send_message(chat_id=update.effective_chat.id,
        text="""Hi! I am weather bot. Ask any question in this field. I will try to answer! Also, I can talk on general topics.
      Send /cancel to stop talking to me.""")
     


async def bot_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Returns the reply to user after getting reply from server."""
    user = update.message.from_user
    logger.info("Question from User: %s", update.message.text)
    if update.message.text != '':
        
        llm_reply = await generate_openai_response(update.message.text)
      

    else:
        return 

    await update.message.reply_text(llm_reply)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    await update.message.reply_text(
        "Bye! I hope we can talk again some day."
    )




def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(constants.TELEGRAM_BOT_TOKEN).build()

    # Add conversation handler with the states GENDER, PHOTO, LOCATION and BIO
    application.add_handler(CommandHandler("start", start))
    #application.add_handler(CommandHandler("modify", modify_global_variable))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,bot_reply))
    application.add_handler(CommandHandler("cancel", cancel))

    application.run_polling()

import asyncio
if __name__ == "__main__":
    asyncio.run(main())