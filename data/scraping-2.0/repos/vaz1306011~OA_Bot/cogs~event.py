import os
import random
import re
import sqlite3

import discord
import openai
from discord import app_commands
from discord.app_commands import MissingPermissions
from discord.ext import commands

from core.check import is_exception_content
from core.classes import Cog_Extension
from core.data import PRESENCE


class Event(Cog_Extension):
    def __init__(self, bot: commands.Bot):
        super().__init__(bot)
        self.conn_dom = sqlite3.connect("./data/on_message_ignore.db")

    @commands.Cog.listener()
    async def on_ready(self):
        from datetime import datetime

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] - OA_Bot上線")

        activity = discord.Activity(
            type=PRESENCE["type"], name=PRESENCE["name"], url=PRESENCE["url"]
        )

        await self.bot.change_presence(status=PRESENCE["status"], activity=activity)

        openai.api_key = os.environ.get("OPENAI_API_KEY")

    omi_group = app_commands.Group(name="omi", description="關鍵字檢測指令群組")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        content = re.sub(r"https?://\S{2,}\b", "", message.content)

        if is_exception_content(message):
            return

        disabled = self.check_in_omi(message)

        if disabled:
            return

        # 中獎判斷
        if random.randint(1, 10_000) == 1:
            await message.channel.send("10000分之1的機率,被雷劈", reference=message)

        if random.randint(1, 22_000_000) == 1:
            await message.channel.send("2200萬分之一的機率,威力彩頭獎", reference=message)

        # 關鍵字判斷
        if any(word in content for word in ("笑", "草", "ww")):
            word = random.choice(("笑死", "草", ""))
            await message.channel.send(word + "w" * random.randint(2, 5))

        if "好" in content:
            await message.channel.send("好耶")

        if re.search(r"[確雀][實石食]", content):
            word = random.choice(("確實", "雀石", "雀食"))
            await message.channel.send(word)

    def check_in_omi(self, message: discord.Message) -> bool:
        """檢查是否在忽略名單

        Args:
            message (discord.Message): message

        Returns:
            bool: 是否在忽略名單
        """
        try:
            cursor = self.conn_dom.cursor()
            disabled = any(
                (
                    cursor.execute(
                        "SELECT 1 FROM guilds WHERE id = ?",
                        (message.guild.id,),
                    ).fetchone(),
                    cursor.execute(
                        "SELECT 1 FROM channels WHERE id = ?",
                        (message.channel.id,),
                    ).fetchone(),
                    cursor.execute(
                        "SELECT 1 FROM users WHERE id = ?",
                        (message.author.id,),
                    ).fetchone(),
                )
            )
        finally:
            cursor.close()

        return disabled

    def omi_insert(self, table: str, id: int, name: str):
        """新增忽略

        Args:
            table (str): 資料表
            id (int): id
            name (str): 名稱
        """
        try:
            cursor = self.conn_dom.cursor()
            cursor.execute(
                f"INSERT OR IGNORE INTO {table} VALUES (?, ?)",
                (id, name),
            )
            self.conn_dom.commit()
        finally:
            cursor.close()

    def omi_delete(self, table: str, id: int):
        """刪除忽略

        Args:
            table (str): 資料表
            id (int): id
        """
        try:
            cursor = self.conn_dom.cursor()
            cursor.execute(
                f"DELETE FROM {table} WHERE id = ?",
                (id,),
            )
            self.conn_dom.commit()
        finally:
            cursor.close()

    @omi_group.command()
    @app_commands.checks.has_permissions(manage_guild=True)
    async def guild(self, interaction: discord.Interaction, status: bool):
        """忽略伺服器的關鍵字檢測

        Args:
            interaction (discord.Interaction): interaction
            status (bool): 開關
        """
        if status:
            self.omi_insert(
                "guilds",
                interaction.guild_id,
                interaction.guild.name,
            )
        else:
            self.omi_delete("guilds", interaction.guild_id)

        await interaction.response.send_message(
            f"已**{'忽略' if status else '啟用'}**此伺服器的關鍵字檢測", ephemeral=True
        )

    @omi_group.command()
    @app_commands.checks.has_permissions(manage_channels=True)
    async def channel(self, interaction: discord.Interaction, status: bool):
        """忽略頻道的關鍵字檢測

        Args:
            interaction (discord.Interaction): interaction
            status (bool): 開關
        """
        if status:
            self.omi_insert(
                "channels",
                interaction.channel_id,
                interaction.channel.name,
            )
        else:
            self.omi_delete("channels", interaction.channel_id)

        await interaction.response.send_message(
            f"已**{'忽略' if status else '啟用'}**此頻道的關鍵字檢測", ephemeral=True
        )

    @guild.error
    @channel.error
    async def guild_and_channel_error(self, interaction: discord.Interaction, error):
        if isinstance(error, MissingPermissions):
            await interaction.response.send_message("你沒有權限這麼做", ephemeral=True)

    @omi_group.command()
    async def me(self, interaction: discord.Interaction, status: bool):
        """忽略你的關鍵字檢測

        Args:
            interaction (discord.Interaction): interaction
            status (bool): 開關
        """
        if status:
            self.omi_insert(
                "users",
                interaction.user.id,
                interaction.user.name,
            )
        else:
            self.omi_delete("users", interaction.user.id)

        await interaction.response.send_message(
            f"已**{'忽略' if status else '啟用'}**你的關鍵字檢測", ephemeral=True
        )

    @omi_group.command()
    async def status(self, interaction: discord.Interaction):
        """查看忽略狀態

        Args:
            interaction (discord.Interaction): interaction
        """
        try:
            cursor = self.conn_dom.cursor()
            guild_status = not not cursor.execute(
                "SELECT 1 FROM guilds WHERE id = ?", (interaction.guild_id,)
            ).fetchone()
            channel_status = not not cursor.execute(
                "SELECT 1 FROM channels WHERE id = ?", (interaction.channel_id,)
            ).fetchone()
            user_status = not not cursor.execute(
                "SELECT 1 FROM users WHERE id = ?", (interaction.user.id,)
            ).fetchone()
        finally:
            cursor.close()

        def format_status(status: bool, name: str) -> str:
            return f"{'+' if status else '-'} {name}: {'忽略' if status else '偵測'}"

        await interaction.response.send_message(
            f"**忽略狀態:**\n```diff\n{format_status(guild_status, '伺服器')}\n{format_status(channel_status, '頻道')}\n{format_status(user_status, '你')}```",
            ephemeral=True,
        )


async def setup(bot: commands.Bot):
    print("已讀取Event")
    await bot.add_cog(Event(bot))


async def teardown(bot: commands.Bot):
    print("已移除Event")
    await bot.remove_cog("Event")
