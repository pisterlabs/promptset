from time import sleep
import discord
import logging

from logic.openai_server import OpenAIServer

class DiscordBot:
    def __init__(self, token: str, userID: int, openAIToken: str):
        self.userID = userID
        self.token = token
        self.client = discord.Client(intents=discord.Intents.all())
        self.isReady = False
        self.openAI = OpenAIServer(token=openAIToken)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.level = logging.INFO

    def run(self):
        try:
            @self.client.event
            async def on_ready():
                if self.isReady:
                    return
                logging.info("Discord bot is Ready")
                if self.openAI.server.isRunning():
                    self.openAI.server.stop()
                self.openAI.server.start()
                self.isReady = True
                try:
                    user = await self.client.fetch_user(self.userID)
                    await user.send("Botが起動しました")
                except:
                    self.logger.error("Failed to send message to user")

            @self.client.event
            async def on_message(message: discord.Message):
                try:
                    replyMessage = None
                    if message.author.bot or message.guild != None:
                        return
                    if message.author.id != self.userID:
                        return
                    while self.openAI.runningLock:
                        sleep(5)
                    embed = discord.Embed()
                    attachmentPath = None

                    if len(message.attachments) > 0:
                        embed.title = "ファイルをダウンロード中"
                        embed.color = discord.Color.blue()
                        embed.add_field(name="完了", value=str(0))
                        embed.add_field(name="ファイル数", value=str(len(message.attachments)))

                        replyMessage = await message.reply(embed=embed)
                        attachmentPath = f"/tmp/{message.id}"
                        self.openAI.server.runCommand(f"mkdir -p {attachmentPath}")
                        for i in range(len(message.attachments)):
                            attachment = message.attachments[i]
                            rawData = await attachment.read()
                            self.openAI.server.writeRawFile(f"{attachmentPath}/{attachment.filename}", rawData)
                            embed.set_field_at(0, name="完了", value=str(i+1))
                            await replyMessage.edit(embed=embed)

                    embed.clear_fields()
                    embed.title = "実行中"
                    embed.color = discord.Color.blue()
                    if replyMessage is None:
                        replyMessage = await message.reply(embed=embed)
                    else:
                        await replyMessage.edit(embed=embed)

                    await self.openAI.run(message.content, attachmentPath=attachmentPath)

                    embed.title = "実行完了"
                    embed.color = discord.Color.green()
                    embed.add_field(name="開放中のポート", value=(", ".join([str(port) for port in self.openAI.server.ports]) if len(self.openAI.server.ports) > 0 else "なし"))
                    if len(self.openAI.reports) > 0:
                        embed.add_field(name="レポート", value=self.openAI.reports[-1])
                    await replyMessage.edit(embed=embed)
                    
                except Exception as e:
                    self.logger.error(e)
                    embed.title = "エラー"
                    embed.description = "```\n"+str(e)+"\n```"
                    embed.color = discord.Color.red()
                    await replyMessage.edit(embed=embed)
            
            self.client.run(token=self.token)
        finally:
            self.openAI.server.stop()
