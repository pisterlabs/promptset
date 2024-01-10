
import random

from openai import OpenAI
from vkbottle.bot import Bot, Message
from vkbottle.dispatch.rules.base import FromUserRule
from toxik import toxik_bot

client = OpenAI()


# Замени на свой токен и ID сообщества или приложения
TOKEN = ""

# Создаем экземпляр бота
bot = Bot(TOKEN)

# функция запуска бота
def main() -> None:
  bot.run_forever()

# Обработка входящих сообщений
@bot.on.chat_message(FromUserRule())
async def talk(message: Message) -> None:
    mess = message.text
    chance_of_exe = 0.3
    random_num = random.random()

    if random_num < chance_of_exe:
        await message.reply(toxik_bot(mess))
        
    else:
        pass
    
if __name__ == "__main__":
    main()
