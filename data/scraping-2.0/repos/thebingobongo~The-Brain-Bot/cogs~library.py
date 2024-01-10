import discord
from discord.ext import commands
from databaselayer import hasEnough
import openai
import requests
import random
import json
from datetime import datetime, date
import pytz

def getAnswer(question):
    text = "The Brain is a chatbot that reluctantly answers questions.\nYou: How many pounds are in a kilogram?\nThe Brain: This again? There are 2.2 pounds in a kilogram. Please make a note of this.\nYou: What does HTML stand for?\nThe Brain: Was Google too busy? Hypertext Markup Language. The T is for try to ask better questions in the future.\nYou: When did the first airplane fly?\nThe Brain: On December 17, 1903, Wilbur and Orville Wright made the first flights. I wish they’d come and take me away.\nYou:" + str(
        question) + "\nThe Brain:"
    response = openai.Completion.create(
        engine="curie",
        prompt=text,
        temperature=0.6,
        max_tokens=60,
        top_p=0.3,
        frequency_penalty=0.75,
        presence_penalty=0.5,
        stop=["\n"]
    )
    answer = response["choices"][0]["text"]
    return answer


def getAnswer2(question):
    text = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: " + str(
        question) + "\n AI:",

    response = openai.Completion.create(
        engine="curie-instruct-beta",
        prompt=text,
        temperature=0.75,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n", " Human:", " AI:"]
    )
    answer = response["choices"][0]["text"]
    return answer


def getAnswer3(question):
    text = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: " + str(
        question) + "\n AI:",

    response = openai.Completion.create(
        engine="davinci-instruct-beta",
        prompt=text,
        temperature=0.75,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n", " Human:", " AI:"]
    )
    answer = response["choices"][0]["text"]
    return answer


def getJoke():
    response = requests.get('https://joke.deno.dev/type/general/1', timeout=30)
    joke = json.loads(response.text)
    text = joke[0]['setup'] + " \n" + joke[0]['punchline']
    embed = discord.Embed(title=text, color=0x00ffff)
    return embed


def getProgrammingJoke():
    response = requests.get('https://joke.deno.dev/type/programming/1', timeout=30)
    joke = json.loads(response.text)
    text = joke[0]['setup'] + " \n" + joke[0]['punchline']
    embed = discord.Embed(title=text, color=0x00ffff)
    return embed


def getKnockKnock():
    response = requests.get('https://joke.deno.dev/type/knock-knock/1', timeout=30)
    joke = json.loads(response.text)
    text = joke[0]['setup'] + " \n" + joke[0]['punchline']
    embed = discord.Embed(title=text, color=0x00ffff)
    return embed


def getInsult():
    response = requests.get('https://evilinsult.com/generate_insult.php?lang=en&type=json', timeout=30)
    insult = json.loads(response.text)
    text = insult['insult']
    while 'testicles' in text or 'Suck' in text or 'chromosomes' in text or 'orangutans' in text or 'abortion' in text or \
            'Tampon' in text or 'reindeer!' in text or 'motherfucker.&quot;\r\n--&gt;' in text or 'jerk off' in text \
            or "amp&" in text or 'booble' in text or 'walt' in text or 'dick,' in text or 'twatface' in text:
        response = requests.get('https://evilinsult.com/generate_insult.php?lang=en&type=json')
        insult = json.loads(response.text)
        text = insult['insult']
    embed = discord.Embed(title=text, color=0x00ffff)
    return embed


def getQuote():
    id = random.randint(1, 583)
    response = requests.get('https://philosophyapi.herokuapp.com/api/ideas/' + str(id), timeout=30)
    json_data = json.loads(response.text)
    quote = '"' + json_data['quote'] + '" \n                           -' + json_data['author']
    while len(quote) >= 256:
        quote = '"' + json_data['quote'] + '" \n                           -' + json_data['author']
    embed = discord.Embed(title=quote, color=0x00ffff)
    return embed


def getAdvice():
    response = requests.get('https://api.adviceslip.com/advice', timeout=30)
    advice = json.loads(response.text)
    text = advice['slip']['advice']
    embed = discord.Embed(title=text, color=0x00ffff)
    return embed


def getSearch(searchterm):
    response = requests.get('http://philosophyapi.herokuapp.com/api/ideas/?search=' + str(searchterm), timeout=30)
    json_data = json.loads(response.text)
    if json_data['count'] != 0:
        quotelist = json_data['results']
        if len(quotelist) == 1:
            quote = '"' + quotelist[0]['quote'] + '" \n                           -' + quotelist[0]['author']
            while len(quote) >= 256:
                quote = '"' + quotelist[0]['quote'] + '" \n                           -' + quotelist[0]['author']
        else:
            rand = random.randint(0, len(quotelist) - 1)
            quote = '"' + quotelist[rand]['quote'] + '" \n                           -' + quotelist[rand]['author']
            while len(quote) >= 256:
                rand = random.randint(0, len(quotelist) - 1)
                quote = '"' + quotelist[rand]['quote'] + '" \n                           -' + quotelist[rand]['author']
    else:
        quote = "Couldn't find a quote with that search. Try another search term."
    embed = discord.Embed(title=quote, color=0x00ffff)
    return embed


def getSearchPhilosopher(philosopher):
    response = requests.get('https://philosophyapi.herokuapp.com/api/philosophers/?search=' + str(philosopher), timeout=30)
    json_data = json.loads(response.text)
    if json_data['count'] != 0:
        quotelist = json_data['results'][0]['ideas']
        name = json_data['results'][0]['name']
        rand = random.randint(1, len(quotelist) - 1)
        quote = '"' + quotelist[rand] + '"\n                         -' + name
    else:
        quote = "Search Term not found. Try a different philosopher."

    embed = discord.Embed(title=quote, color=0x00ffff)
    return embed


def getMathFact():
    response = requests.get('http://numbersapi.com/random/math', timeout=30)
    embed = discord.Embed(title=response.text, color=0x00ffff)
    return embed


def getDateFact():
    today = date.today()
    day = today.day
    month = today.month
    response = requests.get('http://numbersapi.com/' + str(month) + '/' + str(day) + '/date', timeout=30)
    embed = discord.Embed(title=response.text, color=0x00ffff)
    return embed


def getDefinition(search):
    response = requests.get('https://api.dictionaryapi.dev/api/v2/entries/en_US/' + str(search), timeout=30)
    text = json.loads(response.text)
    if len(text) == 3:
        return ["Word not found. Try again."]
    else:
        word = text[0]['word']
        def_list = []
        definitions = text[0]['meanings']
        for i in range(len(definitions)):
            type = definitions[i]['partOfSpeech']
            definition = definitions[i]['definitions'][0]['definition']
            def_list.append([type, definition])
        return [word, def_list]

def getZodiac(ctx, sign):
    link = f"https://aztro.sameerkumar.website/?sign={sign}&day=today"
    response = requests.post(link)
    res = json.loads(response.text)
    embed = discord.Embed(title=f"Daily horoscope for {sign.capitalize()}", colour=ctx.author.color)
    embed.add_field(name="Compatibillity", value=res["compatibility"], inline=True)
    embed.add_field(name="Mood", value=res["mood"], inline=True)
    embed.add_field(name="Lucky Number", value=res["lucky_number"], inline=True)
    embed.add_field(name="Horoscope", value=res["description"], inline=False)
    embed.set_footer(text=res['current_date'])
    return embed


def getUrbanDefinition(word):
    word = word.strip()
    newword = word.replace(" ", "-")
    link = f"https://api.urbandictionary.com/v0/define?term={newword}"
    response = requests.get(link)
    res = json.loads(response.text)
    if res['list'] == []:
        return None
    definition = res['list'][0]['definition']
    example = res['list'][0]['example']
    i=1
    while len(definition) > 256 or len(example) > 1024:
        try:
            definition = res['list'][i]['definition']
            example = res['list'][i]['example']
        except:
            return "Something broke. i cant fix it."
        i+=1
    definition = [char for char in definition if (char != '[' and char != ']' )]
    definition = "".join(definition)
    example = [char for char in example if (char != '[' and char != ']' )]
    example = "".join(example)
    embed = discord.Embed(title=f"Urban definition for {word}",color=0xff0000)
    embed.add_field(name=definition,value=f"\nExample:\n{example}")
    return embed



def getTimeZones(ctx):
    est = get_EST()
    pst = get_PST()
    cst = get_CST()
    cet = get_CET()
    gmt = get_GMT()
    pkt = get_PKT()
    d = get_Today()

    embed = discord.Embed(title="Times in different parts of the world!", color=ctx.author.color)
    embed.add_field(name="Eastern Standard Time | GMT-5", value=est, inline=True)
    embed.add_field(name="Pacific Standard Time | GMT-8", value=pst, inline=True)
    embed.add_field(name="Greenwich Mean Time or UTC", value=gmt, inline=True)
    embed.add_field(name="Central Standard Time | GMT-6", value=cst, inline=True)
    embed.add_field(name="Central European Time | GMT+1", value=cet, inline=True)
    embed.add_field(name="Pakistan Standard Time | GMT+5", value=pkt, inline=True)
    embed.set_footer(text=f"Today is {d}. All times are in 24 hour format.")

    return embed


def get_time(timezone):
    timezone = pytz.timezone(timezone)
    timezone_date_and_time = datetime.now(timezone)
    return timezone_date_and_time.strftime("%H:%M:%S")

def get_EST():
    return get_time('EST')


def get_PST():
    return get_time('PST8PDT')


def get_GMT():
    return get_time('GMT')


def get_CET():
    return get_time("CET")


def get_EET():
    return get_time('EET')


def get_PKT():
    return get_time('Etc/GMT-5')


def get_CST():
    return get_time("Etc/GMT+6")


def get_Today():
    today = date.today()
    d = today.strftime("%b %d, %Y")
    return d


class ApprovalButton(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
     
    @discord.ui.button(label="✅", style=discord.ButtonStyle.green)
    async def checkmark(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.value = True
        self.stop()
        embed = interaction.message.embeds[0]
        embed.title = f"APPROVED by {interaction.user}"
        embed.color = 0x00ff00
        await interaction.message.edit(embed=embed)
        return

    @discord.ui.button(label="❌", style=discord.ButtonStyle.red)
    async def cancel(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.value = False
        self.stop()
        embed = interaction.message.embeds[0]
        embed.title = f"DECLINED by {interaction.user}"
        embed.color = 0xff0000
        await interaction.message.edit(embed=embed)
        return


class Library(commands.Cog):

    def __init__(self, client):
        self.client = client

    async def get_approval(self, type, author, text):
        def check(m):
            print(m.component, m.components)

        approval_channel = self.client.get_channel(908283306599145475)
        view = ApprovalButton()
        embed = discord.Embed(title=f"APPROVAL REQUEST", colour=0x03fce3)
        embed.add_field(name=type, value=text)
        embed.set_footer(text=f"Sent by {author}")
        msg = await approval_channel.send(embed=embed, view=view)
        await view.wait()
        await msg.edit(view=None)
        return view.value

async def check_input(channel, amount):
    try:
        amount = int(amount)
    except:
        await channel.send("There was an error, try again.")
        return False
    if amount <= 0:
        await channel.send("Can't do negative numbers.")
        return False
    elif amount < 50:
        await channel.send("Need to bet at least 50 Brain Cells.")
        return False
    elif not hasEnough(channel.author.id, amount):
        await channel.send("You do not have enough Brain Cells.")
        return False
    return True


def setup(client):
    client.add_cog(Library(client))
