import openai
import asyncio
from collections import deque
import discord
import time
import BotReply as br
from BotInfo import ShellBot_manager
openai.api_type = "azure"
openai.api_key = "0396650923ad40b880bf2a3ce3b80b9b"
openai.api_base = "https://myshell0.openai.azure.com"
openai.api_version = "2023-07-01-preview"
queue_en = deque(maxlen=10)
queue_cn = deque(maxlen=10)
queues = {##接收消息的频道
    '1140481547355553792': queue_en,
    '1144227746956984493': queue_cn,
}
intents = discord.Intents.all()


def create_on_ready(bot):
    async def on_ready():
        print(f'{bot.user.name} has connected to Discord!')
        await bot.change_presence(activity=discord.Game(name="MyShell"))
    return on_ready


def create_on_message(bot,bot_instance):
    async def on_message(message):
        if str(message.channel.id) not in queues:
            return
        #保存指定频道10条消息记录
        discord_prompt = f"""You are {bot_instance.name} in a Discord server called MyShell. People come to this server to engage in casual and informative conversations, ask questions, and share interesting content. As {bot_instance.name}, your goal is to provide helpful, friendly, and entertaining responses to the users, while also considering the presence of other figures. For your information, "MyShell is a robot creation platform that reshapes the future, unleashing unlimited creative potential!
        Experience our product: https://app.myshell.ai
        On our platform, the most advanced AI robots are waiting to meet you. Whether it's casual chatting, practicing speaking, playing games, or seeking psychological counseling, each robot has unique functions and personalities to meet your diverse needs."""

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        channel_id = str(message.channel.id)

        if not is_duplicate(message, queues[channel_id]):
            queues[channel_id].append(
                {"time": current_time, "username": message.author.name, "content": message.content})

        if len(queues[channel_id]) > 10:
            queues[channel_id].popleft()
        # 忽略机器人的消息
        if message.author.bot:
            return
        reference = None
        reference_author = None
        if message.reference and message.reference.resolved:
            if isinstance(message.reference.resolved, discord.Message):
                reference_author = message.reference.resolved.author.name
                reference = message.reference.resolved.content
                print(f"reference:{reference}")
        separator = "----"
        history_messages = separator.join([str(msg) for msg in queues[channel_id]])
        print(f"{channel_id}:{history_messages}\n\n")
        if bot.user in message.mentions:
            async with message.channel.typing():
                fi_reference = f"This is the referenced content for the request, originating from {reference_author}. If the reference is \"NONE,\" you can disregard it. Otherwise, you need to analyze both the author's intent and the user's intent to generate a comprehensive Chinese response. If the reference is not from DogeInShell, it is from another source, so please evaluate accordingly. Here is the reference: {reference}"
                input = message.content.replace(f'<@!{bot.user.id}>', '').strip()
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    deployment_id="gpt-35-turbo-0613",
                    messages=[{"role": "system", "content": bot_instance.prompt},
                              {"role": "system", "content": discord_prompt},
                              {"role": "system", "content": fi_reference},
                              {'role': 'user', 'content': history_messages},
                              {'role': 'user', 'content': f"{bot_instance.prefix}{message.author.name}:{input}"}],
                    temperature=0.7,
                    top_p=1, )
                # reply = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
                reply = f"<@!{message.author.id}> {response.choices[0].message.content}"
            await message.channel.send(reply)
            return
        flag = br.reply_or_not(message.content,bot_instance,history_messages,True)
        # global_flag = br.reply_or_not(message.content,bot_instance,True)
        # botflag = getattr(flag, bot_instance.name)
        if flag:
            # 使用 dogebark 函数生成回复内容
            print('Dog cc', message)
            dogebark_content = br.active_reply(message.content, history_messages, discord_prompt, bot_instance)
            length = len(dogebark_content)
            delay = length * 0.05  # 延时 0.1 秒/字符
            # 发送回复内容
            async with message.channel.typing():
                await asyncio.sleep(delay)
                await message.channel.send(dogebark_content)

    return on_message

def is_duplicate(message, queue):
    for stored_data in queue:
        if stored_data.get("username") == message.author.name and stored_data.get("content") == message.content:
            return True
    return False

bots = []
for bot_name, bot_instance in ShellBot_manager.items():
    bot = discord.Bot(intents=intents)
    bot.on_ready = create_on_ready(bot)
    bot.on_message = create_on_message(bot,bot_instance)
    bots.append(bot)
async def run_bots():
    tasks = []
    for bot, bot_instance in zip(bots, ShellBot_manager.values()):
        task = asyncio.create_task(bot.start(bot_instance.token))
        tasks.append(task)
    await asyncio.gather(*tasks)

loop = asyncio.get_event_loop()
loop.run_until_complete(run_bots())