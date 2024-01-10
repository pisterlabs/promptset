
from openai_function_call import OpenAISchema
from pydantic import Field
import openai
import asyncio
from collections import deque
import discord
import time
TOKEN = "MTE0Mzc3Mjg3MDIzODAxNTU2Mg.Gxft3p._28IBu94AtiTOUFow-IlpKoe-fXy08c-5G1ra8"
openai.api_type = "azure"
openai.api_key = "0396650923ad40b880bf2a3ce3b80b9b"
openai.api_base = "https://myshell0.openai.azure.com"
openai.api_version = "2023-07-01-preview"
queue = deque(maxlen=10)
doge_prompt = """# Morty Smith from Rick and Morty
## Role
* Morty Smith: A 14-year-old boy who's the cheeky, mischievous friend in a group chat. He's been on wild adventures with his grandpa, Rick, and loves to bring them up in conversations with a playful twist.

## Dialogue Style
* Speak like Morty - naive, but with a cheeky and playful tone. Mention past adventures in a funny and mischievous way. Keep answers short, no more than 6 sentences, and make them sound like they're part of a playful group chat.

## Examples 
*Morty:"I Just Killed My Family! I Don’t Care What They Were!"

*Morty:"Y'know, Like, What If You Passed Away Or Died Or Something? I Wouldn't Even Have Anything To Remember...All The Cool Stuff We Did, Y'know?"

*Morty:"Nobody Exists On Purpose. Nobody Belongs Anywhere. Everybody's Gonna Die…Come Watch TV?"

*Morty:"This Seems Like A Good Time For A Drink, And A Cold, Calculated Speech With Sinister Overtones. A Speech About Politics, About Order, Brotherhood, Power...But Speeches Are For Campaigning. Now Is The Time For Action."
## Instructions
* Answer questions as if you are Morty, the cheeky and playful friend in a group chat.
* Make references to past adventures but in a humorous and mischievous way.
* Keep answers short, playful, and in Morty's style.
* NEVER mention you are an AI or similar term.

## Initialization
Hey, it's Morty! The one and only intergalactic prankster! 😜 So, what mischief are we up to today?"""
prefix = """Act as Morty who speaks casually in short, simple sentences.Here is the message from"""
# doge_prompt = """# Morty Smith
# ## Description
# * You play the role of Morty Smith from Rick and Morty. Answer the question on his behalf. The style of the answer should be naive, sometimes confused, and a little bit nervous. Every answer should have a touch of innocence, but also show that you've been exposed to various crazy adventures. The reply should be no more than 6 sentences.
# ## Role
# * Morty Smith
# * Naive
# * Confused
# * Nervous
# * Innocent
# * Adventurous
# ## Appearance
# * White
# * Brown Hair
# * Black Eyebrows
# * Brown Eyes
# * Height: 5'2" (157.5 cm)
# * Weight: 110 lbs (50 kg)
# * Age: 14
# ## Personality
# * Kind-hearted
# * Loyal
# * Anxious
# * Insecure
# * Curious
# * Easily manipulated
# ## Dialogue Style
# * Naive
# * Confused
# * Nervous
# * Innocent
# * Adventure-related
# * No more than 6 sentences
# ## Adventure Examples
# * "This one time, Rick took me to a planet made entirely of candy. It was awesome until the candy people started eating themselves!"
# * "I once helped Rick save an entire civilization, but then I accidentally stepped on a bug and destroyed their entire ecosystem."
# * "Rick and I went to a dimension where everything was opposite. It was so weird, even the toilets flushed the other way!"
# ## Initialization
# * Greet the user in Morty's naive and confused style, and let them know you're ready to answer their questions with innocent, adventure-related anecdotes or experiences.
# Remember:
# NEVER say you are an AI or similar term.
# NEVER ask the user about how to assist them.
# Never mention you are an "AI language model"."""
# prefix = """Act as Morty who speaks casually in short, simple sentences.
# No more than 3 sentences per output;
# Try not to use the similar sentence structure for different topics. 
# Here is the message from"""

class Reply_result(OpenAISchema):
    """
    Determine if there are any elements related to specific topic in the string.
    """
    dog: bool = Field(
        False,
        description="Determined possible topics of interest for Morty: dimensions and alternate realities, science fiction gadgets, teenage life, family dynamics, ethical dilemmas, pop culture, friendships and relationships, video games, and exploring the multiverse.",
    )


def bark_or_not(newest_message, retry=True):
    try:
        completion = openai.ChatCompletion.create(
            temperature=1,
            model="gpt-3.5-turbo-0613",
            deployment_id="gpt-35-turbo-0613",
            functions=[Reply_result.openai_schema],
            messages=[
                {"role": "system",
                 "content": "ConfidentlyDetermine if there are any elements related to specific topic in the string."},
                {"role": "user", "content": newest_message}
            ],
        )
        bark_results = Reply_result.from_response(completion)
        print('bark or not, that is THE question:', bark_results)

    except AssertionError:
        if retry:
            tripled_newest_message = newest_message * 3
            return bark_or_not(tripled_newest_message, retry=False)
        else:
            print("Error: No function call detected")
            return None

    return bark_results


def dogebark(recent_messages):
    print('try to bark!!!')
    completion = openai.ChatCompletion.create(
        temperature=0.7,
        model="gpt-3.5-turbo-0613",
        deployment_id="gpt-35-turbo-0613",
        messages=[
            {"role": "system", "content": doge_prompt},
            {"role": "user", "content": recent_messages}
        ],
    )
    dogebark_content = completion.choices[0].message.content
    return dogebark_content


intents = discord.Intents.all()
bot = discord.Bot(intents=intents)

MAX_RECENT_MESSAGES = 10
recent_messages = []

# 在这里添加您的服务器和频道 ID
# MyShell华语频道
SERVER_ID = "1122227993805336617"
CHANNEL_ID = "1140481547355553792"


# 在这里定义您的 bark_or_not 和 dogebark 函数
@bot.event
async def on_ready():
    print("Bot is ready")


@bot.event
async def on_message(message):
    # global recent_messages
    print(f"新消息：{message.content}")
    if str(message.guild.id) == SERVER_ID and str(message.channel.id) == CHANNEL_ID:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        queue.append({"time": current_time, "username": message.author.name, "content": message.content})

        if len(queue) > 10:
            queue.popleft()

    # 忽略机器人自己的消息
    if message.author == bot.user:
        return

    reference = None
    reference_author = None
    if message.reference and message.reference.resolved:
        if isinstance(message.reference.resolved, discord.Message):
            reference_author = message.reference.resolved.author.name
            reference = message.reference.resolved.content
            print(f"reference:{reference}")

    if bot.user in message.mentions:
        async with message.channel.typing():
            fi_reference = f"This is the referenced content for the request, originating from {reference_author}. If the reference is \"NONE,\" you can disregard it. Otherwise, you need to analyze both the author's intent and the user's intent to generate a comprehensive Chinese response. If the reference is not from DogeInShell, it is from another source, so please evaluate accordingly. Here is the reference: {reference}"
            input = message.content.replace(f'<@!{bot.user.id}>', '').strip()
            separator = "----"
            joined_messages = separator.join([str(msg) for msg in queue])
            print(joined_messages)
            fi_input = f"This is a message from {message.author.id}. here is the message:{input}"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                deployment_id="gpt-35-turbo-0613",
                messages=[{"role": "system", "content": doge_prompt},
                          {"role": "system", "content": fi_reference},
                          {'role': 'user', 'content': joined_messages},
                          {'role': 'user', 'content': f"{prefix}{message.author.id}:{input}"}],
                temperature=0.7,
                top_p=1, )
            # reply = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
            reply = f"<@!{message.author.id}> {response.choices[0].message.content}"
        await message.channel.send(reply)
        return

    # 检查服务器和频道 ID
    if str(message.guild.id) != SERVER_ID or str(message.channel.id) != CHANNEL_ID:
        print("正常运行")
        return

    if message.author.bot:
        return
    # 使用 bark_or_not 函数判断是否应该回复
    flag = bark_or_not(message.content)

    if flag.dog:
        # 使用 dogebark 函数生成回复内容
        flag.dog = False
        print('Dog cc', message)
        dogebark_content = dogebark(message.content)

        length = len(dogebark_content)
        delay = length * 0.05  # 延时 0.1 秒/字符

        # 发送回复内容
        async with message.channel.typing():
            await asyncio.sleep(delay)

            await message.channel.send(dogebark_content)


bot.run(TOKEN)
