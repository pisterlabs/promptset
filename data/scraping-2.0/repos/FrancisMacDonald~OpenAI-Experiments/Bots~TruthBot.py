import openai


class TruthBot:
    _initial_prompt = ""

    def query_openai_truth(self, text):
        openai.api_key = self._openai_api_key

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f'Is "{text}" a true fact? Answer with "true", "sure whatever", "maybe true", "maybe false", "false" only',
            temperature=0.0,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
        )

        if response is None:
            return None

        if response['choices'] is None:
            return None

        if len(response['choices']) == 0:
            return None

        if not response['choices'][0]['text']:
            return None

        return response['choices'][0]['text']

    async def bot_war(self, ctx):
        self.talk_length = 0

    async def get_truth(self, ctx):
        if ctx.message.author == self.bot.user:
            return
        last_message = await self.get_last_message(ctx)
        answer = self.query_openai_truth(last_message)

        if answer is not None:
            await ctx.send(str(answer).lower().replace('\n', '').replace('.', '').strip())

