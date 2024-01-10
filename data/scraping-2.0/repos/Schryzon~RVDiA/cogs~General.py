import base64
import re
import os
import discord
import openai
import requests
import aiohttp
from datetime import datetime
from sys import version as pyver
from os import getenv
from cogs.Event import Event
from discord import app_commands
from discord.ext import commands
from time import time
from PIL import Image
from io import BytesIO
from discord.ui import View, Button, button
from scripts.main import heading, Url_Buttons, has_pfp, AIClient
from scripts.main import event_available, titlecase, check_blacklist, check_vote
    
day_of_week = {
    '1':"Senin",
    '2':"Selasa",
    '3':"Rabu",
    '4':"Kamis",
    '5':"Jumat",
    '6':"Sabtu",
    '0':"Minggu"
}

class Regenerate_Answer_Button(View):
    def __init__(self, last_question:str):
        super().__init__(timeout=None)
        self.last_question = last_question
        vote_me = Button(
            label='Suka RVDiA? Vote!', 
            emoji='<:rvdia:1140812479883128862>',
            style=discord.ButtonStyle.green, 
            url='https://top.gg/bot/957471338577166417/vote'
            )
        self.add_item(vote_me)

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        await self.message.edit(view=self)

    @button(label="Jawab Ulang", custom_id='regenerate', style=discord.ButtonStyle.blurple, emoji='🔁')
    async def regenerate(self, interaction:discord.Interaction, button:Button):
        await interaction.response.defer()
        await interaction.channel.typing()
        message = self.last_question
        currentTime = datetime.now()
        date = currentTime.strftime("%d/%m/%Y")
        hour = currentTime.strftime("%H:%M:%S")
        openai.api_key = os.getenv('openaikey')
        result = await AIClient.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=1.2,
            messages=[
            {"role":'system', 'content':getenv('rolesys')+f' You are currently talking to {interaction.user}'},
            {"role":'assistant', 'content':f"The current date is {date} at {hour} UTC+8"},
            {"role": "user", "content": message}
            ]
        )
        
        if len(message) > 256:
            message = message[:253] + '...' #Adding ... from 253rd character, ignoring other characters.

        embed = discord.Embed(
            title=' '.join((titlecase(word) for word in message.split(' '))), 
            color=interaction.user.color, 
            timestamp=interaction.message.created_at
            )
        embed.description = result.choices[0].message.content # Might improve for >4096 chrs
        embed.set_author(name=interaction.user)
        embed.set_footer(text='Jika ada yang ingin ditanyakan, bisa langsung direply!')
        return await interaction.message.edit(embed=embed, view=self)

class General(commands.Cog):
    """
    Kumpulan command umum.
    """
    def __init__(self, bot):
        self.bot = bot

    @commands.hybrid_group(name='rvdia')
    @check_blacklist()
    async def rvdia_command(self, ctx:commands.Context) -> None:
        """
        Kumpulan command khusus untuk RVDIA. [GROUP]
        """
        await self.rvdia(ctx)
        pass

    @commands.hybrid_group(name='user')
    @check_blacklist()
    async def user_command(self, ctx:commands.Context, member:discord.Member=None) -> None:
        """
        Kumpulan command khusus untuk mengetahui info pengguna. [GROUP]
        """
        member = member or ctx.author
        await self.userinfo(ctx, member=member)
        pass

    @commands.hybrid_group(name='avatar')
    @check_blacklist()
    async def avatar_command(self, ctx:commands.Context, *, member:discord.User=None) -> None:
        """
        Kumpulan command khusus yang berkaitan dengan avatar pengguna. [GROUP]
        """
        member = member or ctx.author
        await self.avatar(ctx, global_user=member)
        pass

    @commands.hybrid_command(description='Mengulangi apapun yang kamu katakan!')
    @app_commands.describe(
        teks='Apa yang kamu ingin aku katakan?',
        attachment='Lampirkan gambar, kalau mau.'
        )
    @check_blacklist()
    async def say(self, ctx:commands.Context, attachment:discord.Attachment=None, *, teks:str=None):
        """
        Mengulangi apapun yang kamu katakan!
        """
        async with ctx.typing():
            if attachment:
                await attachment.save(attachment.filename)
                file = discord.File(attachment.filename)
                if teks:
                    await ctx.send(teks, file=file)
                else:
                    await ctx.send(file=file)
                
                os.remove(attachment.filename) # Haiyaa
            
            else:
                await ctx.send(teks) if teks else await ctx.send("Aku gak tau harus berkata apa ¯\_(ツ)_/¯")

    @rvdia_command.command(name="about", aliases=['intro', 'bot', 'botinfo'])
    @check_blacklist()
    async def rvdia(self, ctx:commands.Context) -> None:
        """
        Memperlihatkan segalanya tentang aku!
        """
        async with ctx.typing():
            m = 0
            for k in self.bot.guilds:
                m += k.member_count -1
            embed = discord.Embed(title="Tentang RVDIA", color=self.bot.color)
            embed.set_thumbnail(url=self.bot.user.avatar.url)
            embed.set_image(url=getenv('banner') if not self.bot.event_mode else getenv('bannerevent'))
            embed.add_field(name = "Versi", value = f"{self.bot.__version__}", inline=False)
            embed.add_field(name = "Mode", value = f"Event Mode" if self.bot.event_mode else "Standard Mode", inline=False)
            embed.add_field(name = "Pencipta", value = f"<@877008612021661726> (Jayananda)", inline=False)
            embed.add_field(name = "Prefix", value = '@RVDIA | / (slash)')
            embed.add_field(name = "Bahasa Pemrograman", value=f"Python ({pyver[:6]})\ndiscord.py ({discord.__version__})", inline=False)
            embed.add_field(name = "Nyala Sejak", value = f"<t:{round(self.bot.runtime)}>\n(<t:{round(self.bot.runtime)}:R>)", inline = False)
            embed.add_field(name = "Jumlah Server", value = f"{len(self.bot.guilds)} Server")
            embed.add_field(name = "Jumlah Pengguna", value = f"{m} Pengguna")
            embed.add_field(name = "Jumlah Command", value = f"Semua: `{len(self.bot.commands)}`\nGlobal: `{self.bot.synced[1]}`", inline=False)
            embed.set_footer(text="Jangan lupa tambahkan aku ke servermu! ❤️")
            await ctx.send(embed=embed, view=Url_Buttons())
    
    @rvdia_command.command(name="ping",
        description = "Menampilkan latency ke Discord API."
        )
    @check_blacklist()
    async def ping(self, ctx:commands.Context) -> None:
        """
        Menampilkan latency ke Discord API
        """
        start_typing = time()
        await ctx.typing()
        end_typing = time()
        delta_typing = end_typing - start_typing
        embed= discord.Embed(title= "Ping--Pong!", color=self.bot.color, timestamp=ctx.message.created_at)
        embed.description = f"**Discord API:** `{round(self.bot.latency*1000)} ms`\n**Typing:** `{round(delta_typing/1000, 2)} ms`"
        await ctx.reply(embed=embed)

    @user_command.command(description="Memperlihatkan avatar pengguna Discord.")
    @app_commands.rename(global_user='pengguna')
    @app_commands.describe(global_user='Pengguna yang ingin diambil foto profilnya')
    @has_pfp()
    @check_blacklist()
    async def avatar(self, ctx, *, global_user: discord.User = None):
        """
        Memperlihatkan avatar pengguna Discord.
        Support: (ID, @Mention, username, name#tag)
        """
        async with ctx.typing():
            global_user = global_user or ctx.author

            if global_user.avatar is None:
                return await ctx.reply(f'{global_user} tidak memiliki foto profil!')
            png = global_user.avatar.with_format("png").url
            jpg = global_user.avatar.with_format("jpg").url
            webp = global_user.avatar.with_format("webp").url

            embed=discord.Embed(title=f"Avatar {global_user}", url = global_user.avatar.with_format("png").url, color= 0xff4df0)

            if global_user.avatar.is_animated():
                gif = global_user.avatar.with_format("gif").url
                embed.set_image(url = global_user.avatar.with_format("gif").url)
                embed.description = f"[png]({png}) | [jpg]({jpg}) | [webp]({webp}) | [gif]({gif})"

            else:
                embed.description = f"[png]({png}) | [jpg]({jpg}) | [webp]({webp})"
                embed.set_image(url = global_user.avatar.with_format("png").url)
            embed.set_footer(text=f"{ctx.author}", icon_url=ctx.author.avatar.url)
            await ctx.reply(embed=embed)

    @user_command.command(name='info', aliases = ['whois'], description="Lihat info tentang seseorang di server ini.")
    @app_commands.rename(member='pengguna')
    @app_commands.describe(
        member = 'Siapa yang ingin diketahui infonya?'
    )
    @check_blacklist()
    async def userinfo(self, ctx:commands.Context, *, member:discord.Member = None):
        """
        Lihat info tentang seseorang di server ini.
        Support: (ID, @Mention, username, name#tag)
        """
        async with ctx.typing():
            member = member or ctx.author
            avatar_url = member.display_avatar.url # Avoids returning None
            bot = member.bot

            if bot == True:
                avatar_url = "https://emoji.gg/assets/emoji/bottag.png"
            roles = [role.mention for role in member.roles][::-1][:-1] or ["None"]
            if roles[0] == "None":
                role_length = 0
            else:
                role_length = len(roles)
            nick = member.display_name
            if nick == member.name:
                nick = "None"

            perm_list = [perm[0] for perm in member.guild_permissions if perm[1]]
            perm_len = len(perm_list)
            lel = [kol.replace('_', ' ') for kol in perm_list]
            lol = [what.title() for what in lel]

            embed=discord.Embed(title=member, color=member.colour, timestamp=ctx.message.created_at)
            embed.set_author(name="User Info:")
            embed.set_thumbnail(url=avatar_url)
            embed.add_field(name="Nama Panggilan", value=nick, inline=False)
            embed.add_field(name="Akun Dibuat", value=member.created_at.strftime("%a, %d %B %Y"))
            embed.add_field(name="Bergabung Pada", value=member.joined_at.strftime("%a, %d %B %Y"))
            embed.add_field(name="Role tertinggi", value=member.top_role.mention, inline=False)
            if role_length > 10:
                embed.add_field(name=f"Roles [{str(role_length)}]", value=" ".join(roles[:10]) + "\n(__10 role pertama__)", inline=False)
            else:
                embed.add_field(name=f"Roles [{str(role_length)}]", value=" ".join(roles), inline=False)
            embed.add_field(name=f"Permissions [{str(perm_len)}]", value="`"+", ".join(lol)+"`", inline=False)
            owner = await self.bot.fetch_user(ctx.guild.owner_id)
            ack = None
            match member.id: # First use of match case wowwwww
                case self.bot.owner_id:
                    ack = "Pencipta Bot"
                case self.bot.user.id:
                    ack = "The One True Love"

            if ack == None:
                if member.bot == True:
                    ack = "Server Bot"
                elif owner.id == member.id:
                    ack = "Pemilik Server"
                elif member.guild_permissions.administrator == True:
                    ack = "Server Admin"
                else:
                    ack = "Anggota Server"

            embed.add_field(name = "Dikenal Sebagai", value = ack)
            embed.set_footer(text=f"ID: {member.id}", icon_url=avatar_url)
            await ctx.reply(embed=embed)


    @avatar_command.command(aliases=['grayscale'], description="Ubah foto profil menjadi grayscale (hitam putih).")
    @app_commands.rename(user='pengguna')
    @app_commands.describe(user='Foto profil siapa yang ingin diedit?')
    @has_pfp()
    @check_blacklist()
    async def greyscale(self, ctx, *, user:discord.User = None):
        """Ubah foto profil menjadi grayscale."""
        user = user or ctx.author
        avatar = user.display_avatar.with_format("png").url
        async with aiohttp.ClientSession() as session:
            async with session.get(f'https://some-random-api.com/canvas/filter/greyscale?avatar={avatar}') as data:
                image = BytesIO(await data.read())
                await session.close()
                await ctx.reply(file=discord.File(image, 'Grayscale.png'))

    @avatar_command.command(description="Ubah foto profil menjadi inverted (warna terbalik).")
    @app_commands.rename(user='pengguna')
    @app_commands.describe(user='Foto profil siapa yang ingin diedit?')
    @has_pfp()
    @check_blacklist()
    async def invert(self, ctx, *, user:discord.User = None):
        """Ubah foto profil menjadi inverted."""
        user = user or ctx.author
        avatar = user.display_avatar.with_format("png").url
        async with aiohttp.ClientSession() as session:
            async with session.get(f'https://some-random-api.com/canvas/filter/invert?avatar={avatar}') as data:
                image = BytesIO(await data.read())
                await session.close()
                await ctx.reply(file=discord.File(image, 'Inverted.png'))

    @avatar_command.command(description="Meng-crop lingkaran pada foto profilmu!")
    @app_commands.rename(user='pengguna')
    @app_commands.describe(user='Foto profil siapa yang ingin diedit?')
    @has_pfp()
    @check_blacklist()
    async def circle(self, ctx, *, user:discord.User = None):
        """Meng-crop lingkaran pada foto profilmu!"""
        user = user or ctx.author
        avatar = user.display_avatar.with_format("png").url
        async with aiohttp.ClientSession() as session:
            async with session.get(f'https://some-random-api.com/canvas/misc/circle?avatar={avatar}') as data:
                image = BytesIO(await data.read())
                await session.close()
                await ctx.reply(file=discord.File(image, 'Circled.png'))

    @avatar_command.command(description="Membuat foto profilmu menjadi buram!")
    @app_commands.rename(user='pengguna')
    @app_commands.describe(user='Foto profil siapa yang ingin diedit?')
    @has_pfp()
    @check_blacklist()
    async def blur(self, ctx, *, user:discord.User = None):
        """Membuat foto profilmu menjadi buram!"""
        user = user or ctx.author
        avatar = user.display_avatar.with_format("png").url
        async with aiohttp.ClientSession() as session:
            async with session.get(f'https://some-random-api.com/canvas/misc/blur?avatar={avatar}') as data:
                image = BytesIO(await data.read())
                await session.close()
                await ctx.reply(file=discord.File(image, 'Blurred.png'))


class Utilities(commands.Cog):
    """
    Kategori command berupa alat-alat dan fitur bermanfaat.
    """
    def __init__(self, bot):
        self.bot = bot

    @commands.hybrid_command(aliases = ['cuaca'], description="Lihat info tentang cuaca di suatu kota atau daerah!")
    @app_commands.rename(location='lokasi')
    @app_commands.describe(
        location = 'Lokasi mana yang ingin diketahui cuacanya?'
    )
    @check_blacklist()
    async def weather(self, ctx, *, location:str):
        """
        Lihat info tentang keadaan cuaca di suatu kota atau daerah!
        """
        async with ctx.typing():
            try:
                # Need to decode geocode consisting of latitude and longitude
                data = requests.get(f'http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={getenv("openweatherkey")}').json()
                geocode = [data[0]['lat'], data[0]['lon']]
                result = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={geocode[0]}&lon={geocode[1]}&lang=id&units=metric&appid={getenv('openweatherkey')}").json()
                icon = f"http://openweathermap.org/img/wn/{result['weather'][0]['icon']}@4x.png"
                embed = discord.Embed(title=f"Cuaca di {result['name']}", description=f"__{result['weather'][0]['description'].title()}__")
                embed.color = 0x00ffff
                embed.set_thumbnail(url=icon)
                temp = result['main']
                embed.add_field(
                        name=f"Suhu ({temp['temp']}°C)",
                        value = 
                        f"**Terasa seperti:** ``{temp['feels_like']}°C``\n**Minimum:** ``{temp['temp_min']}°C``\n**Maksimum:** ``{temp['temp_max']}°C``\n"+
                        f"**Tekanan Atmosfir:** ``{temp['pressure']} hPa``\n**Kelembaban:** ``{temp['humidity']}%``\n**Persentase Awan:** ``{result['clouds']['all']}%``",
                        inline=False
                        )
                wind = result['wind']
                embed.add_field(
                    name = "Angin",
                    value = f"""**Kecepatan:** ``{wind['speed']} m/s``\n**Arah:** ``{wind['deg']}° ({heading(wind['deg'])})``
                    """, inline=False
                )
                embed.add_field(
                    name="Sunrise",
                    value=f"<t:{result['sys']['sunrise']}:R>", inline=False
                )
                embed.add_field(
                    name="Sunset",
                    value=f"<t:{result['sys']['sunset']}:R>"
                )
                embed.set_footer(text=f"{ctx.author}", icon_url=ctx.author.avatar.url)
                await ctx.send(embed=embed)

            except(IndexError):
                await ctx.send('Aku tidak bisa menemukan lokasi itu!')

    @commands.hybrid_command(description="Lihat info tentang waktu di suatu kota atau daerah!")
    @app_commands.describe(location='Daerah mana yang ingin kamu ketahui?')
    @app_commands.rename(location='lokasi')
    @check_blacklist()
    async def time(self, ctx:commands.Context, *, location:str): # Does not conflict with the package "time"
        """
        Lihat info tentang waktu di suatu kota atau daerah!
        """
        async with ctx.typing():
            check_timezone = requests.get(f'http://worldtimeapi.org/api/timezone').json()
            area = []
            for elements in check_timezone:
                match = elements.split("/") # Split karena formatnya Continent/Area
                if location.title() in match:
                    area = match

            if area == []:
                return await ctx.send('Aku tidak bisa menemukan daerah itu!\nLihat list daerah yang ada [click disini!](http://www.worldtimeapi.org/api/timezone)\nContoh: `r-time Makassar`')
            
            req_data = "/".join(area)
            data = requests.get(f'http://worldtimeapi.org/api/timezone/{req_data}').json()
            day = str(data['day_of_week'])
            day = day_of_week[day]

            local_datetimestr = data['datetime']
            utc_datetimestr = data['utc_datetime']
            local_datetimeobj = datetime.fromisoformat(local_datetimestr)
            utc_datetimeobj = datetime.fromisoformat(utc_datetimestr)

            local_time = local_datetimeobj.strftime('%H:%M:%S')
            utc_time = utc_datetimeobj.strftime('%H:%M:%S')

            embed = discord.Embed(title=f"Waktu di {area[1]}", description=f"UTC{data['utc_offset']}", color=0x00ffff)
            embed.add_field(name="Akronim Timezone", value=data['abbreviation'], inline=False)
            embed.add_field(name="Perbandingan Waktu:",
                            value=f"Waktu Lokal: {local_time}\nWaktu UTC: {utc_time}\nWaktu Anda: <t:{ctx.message.created_at}:T>",
                            inline=False
                            )
            embed.add_field(name="Hari di Lokasi", value=f"{day} (Hari ke-{data['day_of_year']})", inline=False)
            await ctx.send(embed=embed)

    @commands.hybrid_command(
            aliases = ['ask', 'chatbot', 'tanya'],
            description = 'Tanyakan atau perhintahkan aku untuk melakukan sesuatu!'
        )
    @app_commands.rename(message='pesan')
    @app_commands.describe(message='Apa yang ingin kamu tanyakan?')
    @commands.cooldown(type=commands.BucketType.user, per=2, rate=1)
    @check_blacklist()
    async def chat(self, ctx:commands.Context, *, message:str):
        """
        Tanyakan atau perhintahkan aku untuk melakukan sesuatu!
        """
        async with ctx.typing():
            currentTime = datetime.now()
            date = currentTime.strftime("%d/%m/%Y")
            hour = currentTime.strftime("%H:%M:%S")
            result = await AIClient.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=1.2,
                messages=[
                {"role":'system', 'content':getenv('rolesys')+f' You are currently talking to {ctx.author}'},
                {"role":'assistant', 'content':f"The current date is {date} at {hour} UTC+8"},
                {"role": "user", "content": message}
                ]
            )
            
            if len(message) > 256:
               message = message[:253] + '...' #Adding ... from 253rd character, ignoring other characters.

            embed = discord.Embed(
                title=' '.join((titlecase(word) for word in message.split(' '))), 
                color=ctx.author.color, 
                timestamp=ctx.message.created_at
                )
            embed.description = result.choices[0].message.content # Might improve for >4096 chrs
            embed.set_author(name=ctx.author)
            embed.set_footer(text='Jika ada yang ingin ditanyakan, bisa langsung direply!')
            regenerate_button = Regenerate_Answer_Button(message)
            return await ctx.reply(embed=embed, view=regenerate_button)

    @commands.hybrid_command(
            aliases = ['image', 'create'],
            description = 'Ciptakan sebuah karya seni!'
        )
    @app_commands.describe(prompt='Apa yang ingin diciptakan?')
    @commands.cooldown(type=commands.BucketType.user, per=2, rate=1)
    @check_blacklist()
    async def generate(self, ctx:commands.Context, *, prompt:str):
        """
        Ciptakan sebuah karya seni dua dimensi dengan perintah!
        """
        async with ctx.typing():
            start=time()
            result = await AIClient.images.generate(
                model='dall-e-3',
                prompt=prompt,
                size='1024x1024',
                response_format='b64_json',
                n=1
            )
            b64_data = result.data[0].b64_json; end=time() # Finished generating and gained data
            decoded_data = base64.b64decode(b64_data)
            image=open('generated.png', 'wb')
            image.write(decoded_data)
            image.close()
            required_time=end-start

            embed = discord.Embed(title='Karya Diciptakan', color=ctx.author.colour, timestamp=ctx.message.created_at)
            embed.description = f'Prompt: `{prompt}`\nWaktu dibutuhkan: **`{round(required_time, 2)} detik`**'
            file = discord.File("generated.png")
            embed.set_image(url= "attachment://generated.png")
        
        await ctx.reply(file=file, embed=embed)
        os.remove('./generated.png')

    def crop_to_square(self, img_path):
        """
        Converts ANY aspect ratio to 1:1
        Thanks, RVDIA!
        """
        with Image.open(img_path) as img:
            width, height = img.size
            size = min(width, height)
            left = (width - size) // 2
            top = (height - size) // 2
            right = (width + size) // 2
            bottom = (height + size) // 2
            cropped = img.crop((left, top, right, bottom))
            cropped.save(img_path[2:])

    @commands.hybrid_command(
        aliases=['edit', 'imageedit'],
        description='Ciptakan variasi dari gambar yang diberikan!'
        )
    @app_commands.describe(attachment='Lampirkan gambar!')
    @commands.cooldown(type=commands.BucketType.user, per=2, rate=1)
    @check_blacklist()
    async def variation(self, ctx:commands.Context, attachment:discord.Attachment):
        """
        Ciptakan variasi dari gambar yang diberikan!
        """
        async with ctx.typing():
            attachment = attachment or ctx.message.attachments[0]
            
            await attachment.save(attachment.filename)
            self.crop_to_square(f'./{attachment.filename}')
            selected_image=attachment.filename

            special_supported = ['.jpg', '.JPEG', '.jpeg']
            if any(attachment.filename.endswith(suffix) for suffix in special_supported):
                image = Image.open(attachment.filename)
                image.save(f'{attachment.filename[:-3]}.png' if attachment.filename.endswith('.jpg') else f'{attachment.filename[:-4]}.png')
                selected_image = f'{attachment.filename[:-3]}.png' if attachment.filename.endswith('.jpg') else f'{attachment.filename[:-4]}.png'

            start=time()
            result = await AIClient.images.create_variation(
                image = open(selected_image, 'rb'),
                model='dall-e-2',
                size='1024x1024',
                response_format = 'b64_json',
                n=1
            )
            os.remove(f'./{selected_image}') # No longer need file
            b64_data = result.data[0].b64_json; end=time()
            decoded_data = base64.b64decode(b64_data)
            image=open('variation.png', 'wb')
            image.write(decoded_data)
            image.close()
            required_time=end-start

            embed = discord.Embed(title='Variasi Diciptakan', color=ctx.author.colour, timestamp=ctx.message.created_at)
            embed.description = f'Waktu dibutuhkan: **`{round(required_time, 2)} detik`**'
            file = discord.File("variation.png")
            embed.set_image(url= "attachment://variation.png")
            embed.set_footer(text='Kesalahan pada gambar? Mungkin karena gambar aslinya tidak 1:1!')

        await ctx.reply(file=file, embed=embed)
        os.remove('./variation.png')

    @commands.hybrid_command(description="Memperlihatkan warna dari nilai hexadecimal.")
    @app_commands.describe(hex='Kode hexadecimal (Contoh: FF0000).')
    @has_pfp()
    @check_blacklist()
    async def hex(self, ctx:commands.Context, hex:str):
        """Memperlihatkan warna dari nilai hexadecimal."""
        if "#" in hex:
            hex = hex.split('#')[1]

        async def validate_hex(hex_str:str):
            pattern = r'^[0-9A-Fa-f]+$'  # Regular expression pattern for hexadecimal string
            if not re.match(pattern, hex_str):
                raise ValueError("Invalid hex!")
            
        try:
            await validate_hex(hex)
        except: # Malas
            return await ctx.reply(f"`{hex}` bukan merupakan kode hexadecimal yang valid!", ephemeral=True)
        
        hex_code = int(hex, 16)
        red = (hex_code >> 16) & 0xff # Bitwise right shift
        green = (hex_code >> 8) & 0xff
        blue = hex_code & 0xff
        async with aiohttp.ClientSession() as session:
            async with session.get(f'https://some-random-api.com/canvas/misc/colorviewer?hex={hex}') as data:
                image = BytesIO(await data.read())
                await session.close()
                await ctx.reply(content=f"Hex: #{hex.upper()}\nRGB: ({red}, {green}, {blue})", file=discord.File(image, f'{hex.upper()}.png'))

    @commands.hybrid_command(description="Memperlihatkan warna dari nilai RGB.")
    @app_commands.describe(
        red='Warna merah (0 - 255)',
        green='Warna hijau (0 - 255)',
        blue='Warna biru (0 - 255)'
        )
    @has_pfp()
    @check_blacklist()
    async def rgb(self, ctx:commands.Context, red:int, green:int, blue:int):
        """Memperlihatkan warna dari nilai RGB."""
        rgb = [red, green, blue]
        if any(color > 255 for color in rgb):
            return await ctx.reply("Salah satu nilai dari warna RGB melebihi 255!\nPastikan nilai RGB valid!", ephemeral=True)
        async with aiohttp.ClientSession() as session:
            initial_connection = await session.get(f'https://some-random-api.com/canvas/misc/hex?rgb={red},{green},{blue}')
            data = await initial_connection.json()
            hex_string = data['hex'].split('#')[1]
        await self.hex(ctx, hex_string) # Cheat

class Support(commands.GroupCog, group_name='support'):
    """
    Kumpulan command khusus untuk memperoleh bantuan dan pemberian saran/kritik.
    """
    def __init__(self, bot):
        self.bot = bot
    
    class Support_Button(View):
        def __init__(self):
            super().__init__(timeout=None)

            support_server = Button(
                label= "Support Server",
                emoji = '<:cyron:1082789553263349851>',
                style = discord.ButtonStyle.blurple,
                url = 'https://discord.gg/QqWCnk6zxw'
            )
            self.add_item(support_server)

    class Donate_Button(View):
        def __init__(self):
            super().__init__(timeout=None)

            donate = Button(
                label= "Saweria Link",
                emoji = '<:rvdia_happy:1121412270220660803>',
                style = discord.ButtonStyle.blurple,
                url = 'https://saweria.co/schryzon'
            )
            self.add_item(donate)

    @app_commands.command(description = 'Mengirimkan link untuk server supportku!')
    async def guild(self, interaction:discord.Interaction):
        """
        Mengirimkan link untuk server supportku!
        """
        await interaction.response.send_message(f"Untuk join serverku agar dapat mengetahui lebih banyak tentang RVDiA, silahkan tekan link di bawah!\nhttps://discord.gg/QqWCnk6zxw\nAtau tekan tombol abu-abu di bawah ini.", view=self.Support_Button())

    @app_commands.command(description = 'Dukung RVDiA melalui Saweria!')
    async def donate(self, interaction:discord.Interaction):
        """
        Dukung RVDiA melalui Saweria!
        """
        await interaction.response.send_message(f"Untuk mendukung RVDiA secara finansial, tekan link di bawah ini!\nhttps://saweria.co/schryzon\nAtau tekan tombol abu-abu di bawah ini. Terima kasih!", view=self.Donate_Button())

    @app_commands.command(description = 'Berikan aku saran untuk perbaikan atau penambahan fitur!')
    @app_commands.rename(text='saran')
    @app_commands.rename(attachment='lampiran')
    @app_commands.describe(text='Apa yang ingin kamu sampaikan?')
    @app_commands.describe(attachment='Apakah ada contoh gambarnya? (Opsional)')
    @check_blacklist()
    async def suggest(self, interaction:discord.Interaction, text:str, attachment:discord.Attachment = None):
        """
        Berikan aku saran untuk perbaikan atau penambahan fitur!
        """
        channel = self.bot.get_channel(1118145279464570921)
        embed = discord.Embed(title="Saran Baru!", color=interaction.user.color, timestamp=interaction.message.created_at)
        embed.set_author(name=f"Dari {interaction.user}")
        if attachment:
            embed.set_image(url = attachment.url)
        embed.description = text
        embed.set_thumbnail(url = interaction.user.display_avatar.url) # New knowledge get!
        await channel.send(embed=embed)
        await interaction.response.send_message(f"Terima kasih atas sarannya!\nSemoga RVDiA akan selalu bisa memenuhi ekspektasimu!")

async def setup(bot:commands.Bot):
    await bot.add_cog(General(bot))
    await bot.add_cog(Utilities(bot))
    await bot.add_cog(Support(bot))