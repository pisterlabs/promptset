from discord.ext import commands
import json
import openai
import random
from datetime import datetime
from io import BytesIO
import discord
from PIL import Image, ImageFont, ImageDraw

class Message(commands.Cog):
    def __init__(self, client):
        self.client = client


    @commands.Cog.listener()
    async def on_member_join(self, ctx):
        # 1600 800
        member_name = ctx.name
        card = Image.open("welcome_base.jpg")
        draw = ImageDraw.Draw(card)
        font = ImageFont.truetype("RussoOne-Regular.ttf", size=150)
        text = "Witamy Cię!"
        member_name = str(ctx.name)
        draw.text((1600,300), text, (0,0,0), anchor="ms", font=font)
        draw.text((1600,500), member_name, (255,0,0) , anchor="ms", font=font)
        asset = ctx.display_avatar
        data = BytesIO(await asset.read())
        pfp = Image.open(data)
        pfp = pfp.resize((553,553))
        card.paste(pfp,(210,200))
        card.save("welcome.jpg")
        dater = str(datetime.now())
        now = dater[:-10]
        with open("./bot_database/channel_config.json", "r") as f:
            channels = json.load(f)
        welcome = self.client.get_channel(int(channels["welcome"]))
        with open("./bot_database/members.json", "r") as r:
            members = json.load(r)
        if str(ctx.id) in members:
            again=discord.Embed(title=f"Witamy cie ponownie {ctx.mention} na maszym serwerze",description=f"Zajrzyj na #general!")
            await welcome.send(embed=again)
            await welcome.send(file= discord.File("welcome.jpg"))
        if str(ctx.id) not in members:
            indf = ctx.id
            members.update({indf:{}})
            members[indf].update({"joining_date":now})
            with open("./bot_database/members.json", "w") as t:
                json.dump(members, t, indent=1)
            joined= discord.Embed(title=f"Witamy cie {ctx.mention} na maszym serwerze",description=f"Zajrzyj na #general!")
            await welcome.send(embed=joined)    
                
    @commands.Cog.listener()
    async def on_message(self,ctx):
        if ctx.channel.id in [1049410779738284083,1050502781032738909] and ctx.author.name != "FairBOT":
            openai.api_key = "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"
            
            message = await ctx.channel.send("Chwileczkę przetwarzam twoje polecenie...")
            response = openai.Completion.create(model="text-davinci-003",prompt=ctx.content,temperature=0.7,max_tokens=3521,top_p=1,frequency_penalty=0,presence_penalty=0)
            info = dict(response)
            print(type(info))
            await message.edit(content=(info["choices"][0]["text"]))
        if ctx.content=="@everyone":
            words = ["Co mnie oznaczasz ty kurwo jebana","Wyjebać ci kopa za tego pinga","Rozjebie ci dupe zaraz jak sie nie uspokoisz", "O ty kurwo", "Morda pod fiutem", "Zaraz cię zgwałce"]
            await ctx.reply(random.choice(words))
        if ctx.author.name not in ["Grzanka", "FairBOT", "FairSecurity"]:
            username = ctx.author.name
            with open("./bot_database/messages.json", "r") as f:
                user_dic = json.load(f)
                if username in user_dic:
                    messages_number = user_dic[username]
                    messages_number = messages_number + 1
                    with open("messages.json", "w") as f:
                        user_dic.update({username:messages_number})
                        json.dump(user_dic, f, indent=1)
                else:
                    with open("messages.json", "w") as f:
                        user_dic.update({username:1})
                        json.dump(user_dic, f, indent=1)
        await self.client.process_commands(ctx)
        if ctx.content=="bajo jajo":
            await ctx.channel.send("Bajo jajo ty chuju jebany na inowrocławskiej jesteś, ja ci zaraz dam bajo jajo kurwa, zaraz cię ściągnę i ci chuju do dupy dokopię kurwa bajo jajo pierdolone")
     
async def setup(client):
    await client.add_cog(Message(client))