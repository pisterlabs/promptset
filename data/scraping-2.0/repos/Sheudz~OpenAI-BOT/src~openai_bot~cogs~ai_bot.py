import disnake
from disnake.ext import commands

from datetime import timedelta
from os import getenv
from time import time
from re import sub
import openai

role_ban: int = 1054109349628358817
role_admin: int = 1054002956493664268

role_newbie: int = 973871427788873748
role_constant: int = 974602932265811988
role_old: int = 973718824174092288
role_eternalold: int = 1044959103316918354
role_pseudoowner: int = 1044959814096269312

channel_gpt: int = 1054106565663264809


class AiBot(commands.Cog):
    def __init__(self, bot: commands.InteractionBot):
        self.bot = bot
        openai.api_key = getenv("OPENAI_API_KEY")

    @commands.user_command(name="Block")
    async def member_block_user(
        self, inter: disnake.UserCommandInteraction, member: disnake.Member
    ) -> None:
        if isinstance(inter.author, disnake.Member):
            roles: list[int] = [role.id for role in inter.author.roles]
            if role_admin in roles:
                target_roles: list[int] = [role.id for role in member.roles]
                if role_admin not in target_roles:
                    try:
                        if isinstance(inter.guild, disnake.Guild):
                            await member.add_roles(inter.guild.get_role(role_ban))
                            await inter.response.send_message(
                                f"{member.mention}/{member.name} заблокований",
                                ephemeral=True,
                            )
                    except Exception:
                        await inter.response.send_message(
                            f"{member.mention}/{member.name} не був заблокований",
                            ephemeral=True,
                        )
                else:
                    raise commands.CommandError(
                        "Адміністратори не можуть блокувати інших адміністраторів"
                    )
            else:
                raise commands.MissingRole(role_admin)

    @commands.user_command(name="Unblock")
    async def member_unblock_user(
        self, inter: disnake.UserCommandInteraction, member: disnake.Member
    ) -> None:
        if isinstance(inter.author, disnake.Member):
            roles: list[int] = [role.id for role in inter.author.roles]
            if role_admin in roles:
                target_roles: list[int] = [role.id for role in member.roles]
                if role_admin not in target_roles:
                    try:
                        if isinstance(inter.guild, disnake.Guild):
                            await member.remove_roles(inter.guild.get_role(role_ban))
                            await inter.response.send_message(
                                f"{member.mention}/{member.name} разблокований",
                                ephemeral=True,
                            )
                    except Exception:
                        await inter.response.send_message(
                            f"{member.mention}/{member.name} не був заблокований",
                            ephemeral=True,
                        )
                else:
                    raise commands.CommandError(
                        "Адміністратори не можуть розблоковувати інших адміністраторів"
                    )
            else:
                raise commands.MissingRole(role_admin)

    @commands.Cog.listener()
    async def on_message(self, message: disnake.Message) -> None:
        if message.channel.id == channel_gpt:
            if message.author.id != self.bot.user.id:
                if f"<@{self.bot.user.id}>" == message.content[:21]:
                    await message.channel.trigger_typing()
                    response = await openai.ChatCompletion.acreate(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a discord bot powered by ChatGPT",
                            },
                            {"role": "user", "content": message.content[21:]},
                        ],
                        max_tokens=3000,
                    )
                    msg: str = sub(
                        r"\[.*]\((.*)\)", r"\1", response.choices[0].message.content
                    )
                    if len(msg) > 2000:
                        msg_chunks: list[str] = [
                            msg[i : i + 2000] for i in range(0, len(msg), 2000)
                        ]
                        for chunk in msg_chunks:
                            await message.reply(chunk)
                    else:
                        await message.reply(msg)

    @commands.slash_command(
        name="ask", description="ask different OpenAI models a question"
    )
    async def ask_group(self, inter: disnake.ApplicationCommandInteraction) -> None:
        pass

    @ask_group.sub_command(name="babbage", description="ask babbage model a question")
    @commands.cooldown(1, 30, commands.BucketType.user)
    async def ask_babbage(
        self, inter: disnake.ApplicationCommandInteraction, prompt: str
    ) -> None:
        if isinstance(inter.author, disnake.Member):
            if role_ban in [role.id for role in inter.author.roles]:
                await inter.response.send_message(
                    f"Тобі не доступний {self.bot.user.name}", ephemeral=True
                )
            elif inter.channel.id != channel_gpt:
                await inter.response.send_message(
                    "Я можу відповідати на ваші запитання лише у каналі #gpt-chat",
                    ephemeral=True,
                )
            else:
                await inter.response.defer(ephemeral=True)
                try:
                    computation_start: float = time()
                    response = await openai.Completion.acreate(
                        engine="text-babbage-001",
                        prompt=prompt,
                        temperature=0.4,
                        max_tokens=1024,
                        top_p=0.1,
                        frequency_penalty=0.1,
                        presence_penalty=0.1,
                    )
                    elapsedtime = int(round(time() - computation_start))
                    embed = disnake.Embed(
                        title="Відповідь:",
                        description=response["choices"][0]["text"],
                        color=0x5258BD,
                    )
                    embed.add_field(name="Запитання:", value=prompt, inline=False)
                    embed.set_footer(
                        text=f"обробка зайняла {str(timedelta(seconds=elapsedtime))}"
                    )
                    await inter.followup.send(embed=embed)
                except Exception as errmsg:
                    await inter.followup.send(f"{errmsg}")

    @ask_group.sub_command(name="curie", description="ask curie model a question")
    @commands.cooldown(1, 30, commands.BucketType.user)
    async def ask_curie(
        self, inter: disnake.ApplicationCommandInteraction, prompt: str
    ) -> None:
        if isinstance(inter.author, disnake.Member):
            roles: list[int] = [role.id for role in inter.author.roles]
            if role_ban in roles:
                await inter.response.send_message(
                    f"Тобі не доступний {self.bot.user.name}", ephemeral=True
                )
            elif (
                role_newbie not in roles
                and role_constant not in roles
                and role_old not in roles
                and role_eternalold not in roles
                and role_pseudoowner not in roles
            ):
                raise commands.MissingRole(role_newbie)
            elif inter.channel.id != channel_gpt:
                await inter.response.send_message(
                    "Я можу відповідати на ваші запитання лише у каналі #gpt-chat",
                    ephemeral=True,
                )
            else:
                await inter.response.defer(ephemeral=True)
                try:
                    computation_start: float = time()
                    response = await openai.Completion.acreate(
                        engine="text-curie-001",
                        prompt=prompt,
                        temperature=0.4,
                        max_tokens=1024,
                        top_p=0.1,
                        frequency_penalty=0.1,
                        presence_penalty=0.1,
                    )
                    elapsedtime = int(round(time() - computation_start))
                    embed = disnake.Embed(
                        title="Відповідь:",
                        description=response["choices"][0]["text"],
                        color=0x5258BD,
                    )
                    embed.add_field(name="Запитання:", value=prompt, inline=False)
                    embed.set_footer(
                        text=f"обробка зайняла {str(timedelta(seconds=elapsedtime))}"
                    )
                    await inter.followup.send(embed=embed)
                except Exception as errmsg:
                    await inter.followup.send(f"{errmsg}")

    @ask_group.sub_command(name="davinci", description="ask davinci model a question")
    @commands.cooldown(1, 30, commands.BucketType.user)
    async def ask_davinci(
        self, inter: disnake.ApplicationCommandInteraction, prompt: str
    ) -> None:
        if isinstance(inter.author, disnake.Member):
            roles: list[int] = [role.id for role in inter.author.roles]
            if role_ban in roles:
                await inter.response.send_message(
                    f"Тобі не доступний {self.bot.user.name}", ephemeral=True
                )
            elif (
                role_constant not in roles
                and role_old not in roles
                and role_eternalold not in roles
                and role_pseudoowner not in roles
            ):
                raise commands.MissingRole(role_constant)
            elif inter.channel.id != channel_gpt:
                await inter.response.send_message(
                    "Я можу відповідати на ваші запитання лише у каналі #gpt-chat",
                    ephemeral=True,
                )
            else:
                await inter.response.defer(ephemeral=True)
                try:
                    computation_start: float = time()
                    response = await openai.Completion.acreate(
                        engine="text-davinci-003",
                        prompt=prompt,
                        temperature=0.4,
                        max_tokens=1024,
                        top_p=0.1,
                        frequency_penalty=0.1,
                        presence_penalty=0.1,
                    )
                    elapsedtime = int(round(time() - computation_start))
                    embed = disnake.Embed(
                        title="Відповідь:",
                        description=response["choices"][0]["text"],
                        color=0x5258BD,
                    )
                    embed.add_field(name="Запитання:", value=prompt, inline=False)
                    embed.set_footer(
                        text=f"обробка зайняла {str(timedelta(seconds=elapsedtime))}"
                    )
                    await inter.followup.send(embed=embed)
                except Exception as errmsg:
                    await inter.followup.send(f"{errmsg}")

    @commands.slash_command(name="image", description="image generation commands")
    async def image_group(self, inter: disnake.ApplicationCommandInteraction) -> None:
        pass

    @image_group.sub_command(name="generate", description="generate image")
    @commands.cooldown(1, 70, commands.BucketType.user)
    async def image_generate(
        self, inter: disnake.ApplicationCommandInteraction, prompt: str
    ) -> None:
        if isinstance(inter.author, disnake.Member):
            roles: list[int] = [role.id for role in inter.author.roles]
            if role_ban in roles:
                await inter.response.send_message(
                    f"Тобі не доступний {self.bot.user.name}", ephemeral=True
                )
            elif (
                role_newbie not in roles
                and role_constant not in roles
                and role_old not in roles
                and role_eternalold not in roles
                and role_pseudoowner not in roles
            ):
                raise commands.MissingRole(role_newbie)
            elif inter.channel.id != channel_gpt:
                await inter.response.send_message(
                    "Я можу відповідати на ваші запитання лише у каналі #gpt-chat",
                    ephemeral=True,
                )
            else:
                await inter.response.defer(ephemeral=True)
                try:
                    computation_start: float = time()
                    response = await openai.Image.acreate(prompt=prompt, n=1, size="1024x1024")
                    image_url = response["data"][0]["url"]
                    elapsedtime = int(round(time() - computation_start))
                    embed = disnake.Embed(
                        title="Згенероване зображення: " + prompt, color=0x5258BD
                    )
                    embed.set_image(url=image_url)
                    embed.set_footer(
                        text=f"обробка зайняла {str(timedelta(seconds=elapsedtime))}"
                    )
                    await inter.followup.send(embed=embed)
                except Exception as errmsg:
                    await inter.followup.send(f"{errmsg}")


def setup(bot: commands.InteractionBot) -> None:
    bot.add_cog(AiBot(bot))
