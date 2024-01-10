import discord
from discord.ext import commands
import openai
from utils.constants import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


class StoryCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.prompt = """
Subjects:
fortune, happiness, old man
Story:
An old man lived in the village. He was one of the most unfortunate people in the world. The whole village was tired of him; he was always gloomy, he constantly complained and was always in a bad mood.
The longer he lived, the more bile he was becoming and the more poisonous were his words. People avoided him, because his misfortune became contagious. It was even unnatural and insulting to be happy next to him.
He created the feeling of unhappiness in others.
But one day, when he turned eighty years old, an incredible thing happened. Instantly everyone started hearing the rumour:
“An Old Man is happy today, he doesn’t complain about anything, smiles, and even his face is freshened up.”
The whole village gathered together. The old man was asked:
Villager: What happened to you?
“Nothing special. Eighty years I’ve been chasing happiness, and it was useless. And then I decided to live without happiness and just enjoy life. That’s why I’m happy now.” – An Old Man
Moral of the story:
Don’t chase happiness. Enjoy your life.
####
Subjects:
man, wisdom, problems
Story:
People have been coming to the wise man, complaining about the same problems every time. One day he told them a joke and everyone roared in laughter.
After a couple of minutes, he told them the same joke and only a few of them smiled.
When he told the same joke for the third time no one laughed anymore.
The wise man smiled and said:
“You can’t laugh at the same joke over and over. So why are you always crying about the same problem?”
Moral of the story:
Worrying won’t solve your problems, it’ll just waste your time and energy.
####
Subjects:
luck, foolish, donker
Story:
A salt seller used to carry the salt bag on his donkey to the market every day.
On the way they had to cross a stream. One day the donkey suddenly tumbled down the stream and the salt bag also fell into the water. The salt dissolved in the water and hence the bag became very light to carry. The donkey was happy.
Then the donkey started to play the same trick every day.
The salt seller came to understand the trick and decided to teach a lesson to it. The next day he loaded a cotton bag on the donkey.
Again it played the same trick hoping that the cotton bag would be still become lighter.
But the dampened cotton became very heavy to carry and the donkey suffered. It learnt a lesson. It didn’t play the trick anymore after that day, and the seller was happy.
Moral of the story:
Luck won’t favor always.
####
Subjects:
"""

    @commands.command(name="ss")
    @commands.cooldown(1, 300, commands.BucketType.user)
    async def shortstory(self, ctx, first: str, second: str, third: str):
        if not ctx.channel.nsfw:
            await ctx.send("This command can only be used in NSFW channels!")
            return

        prompt = self.prompt + f"{first}, {second}, {third}"

        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0.8,
            max_tokens=350,
            top_p=1.0,
            frequency_penalty=0.1,
            presence_penalty=0,
            stop=["####"]
        )

        output = response['choices'][0]['text']

        await ctx.send(output)


def setup(bot):
    bot.add_cog(StoryCog(bot))
