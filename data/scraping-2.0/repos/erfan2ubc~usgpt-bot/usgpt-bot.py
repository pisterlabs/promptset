from twitchio.ext import commands
import os
import openai

bot = commands.Bot(
    token='oauth:3m24yqpg3ttzmasj956w2gpsm1fvdy',
    client_id='gavmseq7e7dp958747d506kw223g6o',
    nick='usgpt_bot',
    prefix='!',
    initial_channels=['A_US2']
)


@bot.event
async def event_message(ctx):
    print(ctx.content)
    print(ctx.author.name)
    await bot.handle_commands(ctx)

@bot.command(name='test')
async def test(ctx):
    await ctx.send('test passed')

if __name__ == "__main__":
    bot.run()
