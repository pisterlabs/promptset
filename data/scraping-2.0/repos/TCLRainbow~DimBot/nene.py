import asyncio
import json
import os
import sqlite3

import discord
import openai
from discord.ext import commands
from openai import OpenAIError

import missile

fine_tune_stop = '\n\n###\n\n'


def qa_to_jsonl_str(uname, question, ans):
    return json.dumps({
        'prompt': f'DimBot: {question}\n{uname}:{fine_tune_stop}',
        'completion': ' ' + ans + fine_tune_stop
    })


async def gpt_style_response(ctx_or_msg, prompt, model, is_chat_completion, use_typing=True, mention=True, **kwargs):
    try:
        if use_typing:
            async with ctx_or_msg.channel.typing():
                resp = await prompt.req(model, **kwargs)
        else:
            resp = await prompt.req(model, **kwargs)
    except OpenAIError as e:
        await ctx_or_msg.reply('Hmm... Who am I? What is DimBot? Sus\n' + e.user_message, mention_author=mention)
        return
    if is_chat_completion:
        resp = resp['choices'][0]['message']['content'].replace('@', '**@**')
    else:
        resp = resp['choices'][0]['text'].replace('@', '**@**')
    resp_len = len(resp)
    if resp_len <= 2000:
        await ctx_or_msg.reply(resp, mention_author=mention)
    elif resp_len <= 4096:
        await ctx_or_msg.reply(embed=missile.Embed(description=resp), mention_author=mention)
    else:
        await ctx_or_msg.reply(
            embed=missile.Embed(description=resp[:4093] + '...', footer='Note: This response is too long'),
            mention_author=mention
        )


class GPTChat:

    def __init__(self, sys=''):
        self.data = [{'role': 'system', 'content': sys}]

    def __str__(self):
        return str(self.data)

    def user(self, content):
        self.data.append({'role': 'user', 'content': content})

    def bot(self, content):
        self.data.append({'role': 'assistant', 'content': content})

    def sys(self, content):
        self.data[0]['content'] = content

    def load_convo(self, ctx: commands.Context, combine_mem=False):
        authors = set()
        base_content = ''

        for ref_msg in missile.MsgRefIter(ctx.message, include_self=True):
            ref_author = ref_msg.author
            content = ref_msg.content

            if not content and ref_msg.embeds:  # Use embed description as content
                emb = ref_msg.embeds[0]
                if emb.description:
                    content = emb.description
            if ref_author == ctx.bot.user:
                if combine_mem and base_content:
                    self.data.insert(1, {'role': 'user', 'content': base_content})
                    base_content = ''
                self.data.insert(1, {'role': 'assistant', 'content': content})
            # if ref_author.name not in nicknames and ref_author != self.bot.user:
            #     nicknames[ref_author.name] = ref_author.nick if ctx.guild and ref_author.nick else None
            else:
                authors.add(ref_author.name)
                if combine_mem:
                    base_content = f'{ref_author.name}: {content}\n{base_content}'
                else:
                    self.data.insert(1, {'role': 'user', 'content': ref_author.name + ': ' + content})

        if base_content:
            self.data.insert(1, {'role': 'user', 'content': base_content})
        return authors

    async def req(self, model):
        return await openai.ChatCompletion.acreate(model=model, messages=self.data)


class GPTInstruct:

    def __init__(self, prompt):
        self.prompt = prompt

    async def req(self, model, **kwargs):
        return await openai.Completion.acreate(model=model, prompt=self.prompt, max_tokens=500, **kwargs)


class Nene(missile.Cog):

    def __init__(self, bot):
        super().__init__(bot, 'Nene')
        self.model = 'babbage-002'
        self.no_ai = []  # List of messages that AI should not reply to
        self.trans = {}

    async def ask(self, prompt: str, model=None, temperature=0.7, max_tokens=250,
                  stop=None, txt_only=False, clean=True):
        if stop is None:
            stop = ['DimBot: ']
        if model is None:
            model = self.model
        r = await openai.Completion.acreate(
            model=model,
            prompt=prompt,
            temperature=temperature,  # 0.9
            max_tokens=max_tokens,  # 150
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,  # 0.6
            stop=stop
        )
        # if re.match(r'[\n ]*$', r['choices'][0]['text']):
        #     r['choices'][0]['text'] = "⚠️I don't know..."
        if clean:
            r['choices'][0]['text'] = r['choices'][0]['text'].replace('@', '**@**')
        if txt_only:
            return r['choices'][0]['text']
        return r

    # @missile.Cog.listener()
    async def on_ready(self):
        resp = await openai.FineTune.alist()
        self.model = resp.data[-1].fine_tuned_model
        self.logger.info('Set model to ' + self.model)

    @missile.Cog.listener()
    async def on_message(self, msg: discord.Message):
        self.bot.loop.create_task(self.process_chat_trigger(msg))
        self.bot.loop.create_task(self.process_translation(msg))

    async def process_chat_trigger(self, msg: discord.Message):
        potential_ref = missile.msg_refers_to_author(msg, self.bot.user)
        if potential_ref:
            # The reply must be with mention ON
            # The reference cannot be in no_ai
            # The reply cannot begin with cmd prefix
            potential_in_no_ai = self.no_ai.count(potential_ref.id)
            if self.bot.user not in msg.mentions or potential_in_no_ai or await self.msg_is_cmd(msg):
                if potential_in_no_ai > 1:
                    self.no_ai = [i for i in self.no_ai if i != potential_ref.id]
                return
        # If no reference, only way to trigger is start with mentioning
        elif not msg.content.startswith(self.bot.user.mention):
            return

        my_name = msg.guild.me.display_name if msg.guild else self.bot.user.name
        msgs = [ref for ref in missile.MsgRefIter(msg, include_self=True)]
        participants, convo = [], []
        for m in reversed(msgs):
            if m.author != self.bot.user and m.author.display_name not in participants:
                participants.append(m.author.display_name)
            convo_content = m.clean_content
            if m.content.startswith(self.bot.user.mention):
                convo_content = m.clean_content[len(my_name) + 1:]
            convo.append(f'{m.author.name}: {convo_content}')
        lf = '\n'
        prompt = f"{lf.join(convo)}\nDimBot:"
        # print(prompt)
        response = await self.ask(prompt, stop=[f'DimBot:', f'{participants[0]}:'])
        # usage = response['usage']
        # reason = response['choices'][0]['finish_reason']
        response = response['choices'][0]['text']
        await msg.reply(response)
        # await self.bot.get_cog('Hamilton').bot_test.send(embed=missile.Embed(str(usage['total_tokens']), reason))

    async def process_translation(self, msg: discord.Message):
        if msg.author.bot or not msg.guild or await self.msg_is_cmd(msg):
            return  # Immediately discard if author is bot, not in guild or is a cmd
        creator_topic = await self.bot.sql.get_translator_convo_by_participant(
            self.bot.db, ch=msg.channel.id, participant=msg.author.id
        )
        if not creator_topic:
            return  # Discard if not in conversation
        parts_locale = await self.bot.sql.get_translator_participants_locale(
            self.bot.db, creator=creator_topic[0], ch=msg.channel.id
        )
        if len(parts_locale) < 2:
            return  # Discard if not enough participants
        names_locales = {}
        locales_set = set()
        for participant, locale in parts_locale:
            member = msg.guild.get_member(participant)
            if member:
                names_locales[participant] = (member.name, locale)
                locales_set.add(locale)
            else:  # Participant not in guild anymore
                self.bot.loop.create_task(self.bot.sql.remove_translator_participant(
                    self.bot.db, ch=msg.channel.id, participant=participant
                ))
                # No need to clear convo because there will be at least one participant left
        locales_set.remove(names_locales[msg.author.id][1])  # Remove author's locale because why need translation
        if not locales_set:
            return  # No locales to translate to
        topic = f'about {creator_topic[1]} ' if creator_topic[1] else ''
        prompt = (f'The following is a conversation {topic}between these people (and their main language): '
                  f'{", ".join(f"{name} ({locale})" for name, locale in names_locales.values())}. '
                  'Each line is in the format <pupil>: <message>.\n\n')
        convo = ''
        for m in missile.MsgRefIter(msg, True):
            convo = f'{m.author.name}: {m.content}\n' + convo
        prompt += convo
        prompt += f'\nDimBot is a translator. It will translate the last message by {msg.author.name} to '
        if len(locales_set) == 1:
            prompt += locales_set.pop()
        else:
            prompt += (f'languages {", ".join(locales_set)}.\n'
                       'DimBot will only have one message and each translation should be in the format <lang>: '
                       '<translation>, separated by an empty line.')
        prompt += '\nOnly return the translated text.\nDimBot:'
        gpt = GPTInstruct(prompt)
        await gpt_style_response(msg, gpt, 'gpt-3.5-turbo-instruct', False, False, False, temperature=0.25)

    @commands.group(invoke_without_command=True)
    async def ai(self, ctx, *, prompt):
        """
        Commands for controlling the behaviour of the AI.
        """
        if await self.bot.is_owner(ctx.author):
            await ctx.reply(await self.ask(prompt, 'text-davinci-003', txt_only=True))
        else:
            self.bot.help_command.context = ctx
            await self.bot.help_command.send_group_help(ctx.command)

    @ai.command(brief='Sets your self introduction')
    @commands.cooldown(rate=3, per=3600.0, type=commands.BucketType.user)
    async def intro(self, ctx):
        """You can write a self introduction. The bot will learn from it and when there are questions about you, it will
         answer based on your intro."""
        para: str = await self.bot.ask_msg(
            ctx,
            'Tell me about yourself using first-person narrative. Please reply this message in 10 minutes. '
            'You can only change this 3 times per hour.',
            600)
        if not para:
            return
        if fine_tune_stop in para:
            await ctx.reply(f'The sequence ``###`` is not allowed.')
            return
        path = f'ai_data/{ctx.author.id}.json'
        try:
            with open(path, 'r') as f:
                d = json.load(f)
        except FileNotFoundError:
            d = {}
        d['Tell me about yourself.'] = para
        with open(path, 'w') as f:
            json.dump(d, f)
        await ctx.reply('Thanks for telling me!')

    @ai.command(brief='Answer random questions to train the AI')
    async def qa(self, ctx):
        """The bot will keep asking random questions about you. Your reply will be studied by the AI.
        You have 30s to answer each question. When time is out, it will stop asking."""
        path = f'ai_data/{ctx.author.id}.json'
        try:
            with open(path, 'r') as f:
                d = json.load(f)
        except FileNotFoundError:
            d = {}
        lf = '\n'
        counter = 0
        while counter < 3:
            try:
                if d:
                    prompt = f"Ask a new question about me. The following are some questions that you asked:\n\n{lf.join(d.keys())}"
                else:
                    prompt = 'Ask a question about me'

                q = await self.ask(
                    prompt
                    , model='text-curie-001', temperature=1, txt_only=True, stop=['?'])
                q = (q + '?').replace('\n', '')
                a = await self.bot.ask_msg(ctx, "__Please don't answer if I am not asking a question.__\n\n" + q, 30)
                if a:
                    d[q] = a
                    counter = 0
                else:
                    counter = 99
            except ValueError:
                counter += 1
        with open(path, 'w') as f:
            json.dump(d, f)
        if counter == 99:
            await ctx.reply('Thanks for answering those questions!')
        else:
            await ctx.reply("Looks like something is wrong, but I've managed to save your answers. "
                            "Thanks for answering those questions!")

    @ai.group(invoke_without_command=True, brief='Controlling models')
    @missile.is_rainbow()
    async def model(self, ctx):
        resp = await openai.Model.alist()
        print(resp)
        msg = ''
        for model in resp.data:
            if model.owned_by.startswith('user'):
                msg += f'{model.id}\n'
        if not msg:
            msg = 'No models'
        await ctx.reply(msg)

    @model.command(brief='Reset model.jsonl from collected data.')
    async def reset(self, ctx):
        lines = 0
        with open('ai_data/model.jsonl', 'w') as model:
            for file in os.listdir('ai_data'):
                if file.endswith('.json'):
                    with open('ai_data/' + file) as f:
                        d = json.load(f)
                    user = self.bot.get_user(int(file.split('.')[0])).name
                    for k, v in d.items():
                        model.write(qa_to_jsonl_str(user, k, v) + '\n')
                        lines += 1
        file = os.path.getsize('ai_data/model.jsonl')
        await ctx.reply(f'Generated model.jsonl with {lines} lines ({file / 1000} kB)')

    @model.command(brief='Create an entirely new model')
    async def create(self, ctx):
        msg = await ctx.reply('Uploading model file')
        with open('ai_data/model.jsonl', 'rb') as f:
            upload_resp = await openai.File.acreate(f, 'fine-tune')
        file_id = upload_resp.id
        await missile.append_msg(msg, 'Sending model train request')
        tune_resp = await openai.FineTune.acreate(training_file=file_id, model='babbage')
        async for event in await openai.FineTune.astream_events(tune_resp.id):
            await missile.append_msg(msg, event.message)
        await missile.append_msg(msg, 'Completed. New Model ID: ' + tune_resp.fine_tuned_model)
        self.model = tune_resp.fined_tuned_model

    @model.command()
    async def set(self, ctx, model_id):
        self.model = model_id
        await ctx.reply('Set model as ' + model_id)

    @model.command()
    async def ft(self, ctx):
        resp = await openai.FineTune.alist()
        print(resp)
        msg = ''
        for ft in resp.data:
            msg += f'{ft.id} {ft.status}'
            if ft.fine_tuned_model:
                msg += ' -> ' + ft.fine_tuned_model
            msg += '\n'
        if not msg:
            msg = 'No FT'
        await ctx.reply(msg)

    @model.command(name='clear')
    async def m_clear(self, ctx):
        resp = await openai.FineTune.alist()
        tasks = []
        for task in resp.data:
            if task.fine_tuned_model and task.fine_tuned_model != self.model:
                tasks.append(openai.Model.adelete(task.fine_tuned_model))
        await asyncio.wait(tasks)
        await ctx.reply(f'Deleted {len(tasks)} models.')

    @ai.group(invoke_without_command=True)
    @missile.is_rainbow()
    async def file(self, ctx):
        resp = await openai.File.alist()
        print(resp.data)
        msg = ''
        for file in resp.data:
            msg += f'{file.id} **{file.filename}** {file.purpose} {file.status} {file.bytes / 1000}kB\n'
        if not msg:
            msg = 'No files.'
        await ctx.reply(msg)

    @file.command(name='clear')
    async def f_clear(self, ctx):
        resp = await openai.File.alist()
        tasks = map(lambda f: openai.File.adelete(f.id), resp.data)
        await asyncio.wait(tasks)
        await ctx.reply(f'Deleted {len(resp.data)} files.')

    @commands.command(brief='Chat using the ChatGPT (GPT-3.5) model')
    async def gpt3(self, ctx, *, msg):
        await self.gpt_common('gpt-3.5-turbo', ctx, msg)

    @commands.command(brief='Instruct GPT-3.5 to do stuff')
    async def gpt3i(self, ctx, *, msg):
        gpt = GPTInstruct(msg)
        await gpt_style_response(ctx, gpt, 'gpt-3.5-turbo-instruct', False, temperature=0.25)

    @commands.command(brief='Chat using the GPT-4 model')
    async def gpt4(self, ctx, *, msg):
        await self.gpt_common('gpt-4', ctx, msg)

    async def gpt_common(self, model, ctx, msg):
        potential_ref = missile.msg_refers_to_author(ctx.message, self.bot.user)
        if potential_ref and potential_ref.id in self.no_ai:
            return

        prompt = GPTChat()
        # nicknames = {ctx.author.name: ctx.author.nick if ctx.guild and ctx.author.nick else None}
        authors = prompt.load_convo(ctx)

        # for name, nick in nicknames.items():
        #     if nick:
        #         sys_msg += f"{name}'s nickname: {nick}\n"

        prompt.sys(
            f"Your name is DimBot. You are trained with the {model} model. This conversation has users: {','.join(authors)}. You must reply in under 4096 characters.")
        await gpt_style_response(ctx, prompt, model, True)

    @commands.group(aliases=('tl',))
    @missile.guild_only()
    async def translator(self, ctx):
        """Automatic chat translation"""
        if not ctx.invoked_subcommand:
            await self.send_grp_cmd_help(ctx)

    @translator.command(aliases=('c',))
    async def create(self, ctx, *, topic: str = None):
        """Creates a conversation in this channel"""
        if not await self.bot.sql.get_user_lang(self.bot.db, user=ctx.author.id):
            await ctx.reply("Please set your language preference using "
                            f"`{await self.bot.get_prefix(ctx.message)}user lang`")
            return
        try:
            await self.bot.sql.add_translator_convo(self.bot.db, user=ctx.author.id, ch=ctx.channel.id, topic=topic)
        except sqlite3.IntegrityError:
            await ctx.reply('You can only create one conversation per channel.')
            return
        c = await self.bot.sql.get_translator_participant_creator(
            self.bot.db, ch=ctx.channel.id, participant=ctx.author.id
        )  # Get creator of current convo, if he is in one
        if c:  # Already in a convo (c won't be himself cuz the try block above handled that)
            await self.bot.sql.update_translator_participant(
                self.bot.db, creator=ctx.author.id, ch=ctx.channel.id, participant=ctx.author.id
            )
            await self.bot.sql.clean_translator_convo(self.bot.db, creator=c, ch=ctx.channel.id)
            await ctx.reply("You've left the original conversation and created this one!")
        else:
            await self.bot.sql.join_translator_convo(
                self.bot.db, creator=ctx.author.id, ch=ctx.channel.id, participant=ctx.author.id)
            await ctx.reply('Conversation created. Others can join by sending '
                            f'`{await self.bot.get_prefix(ctx.message)}tl join @you`.')

    @translator.command(aliases=('j',))
    async def join(self, ctx, creator: discord.Member):
        """Joins a conversation"""
        if creator.bot:
            await ctx.reply("`creator` can't be a bot.")
            return
        if not await self.bot.sql.get_user_lang(self.bot.db, user=ctx.author.id):
            await ctx.reply("Please set your language preference using "
                            f"`{await self.bot.get_prefix(ctx.message)}user lang`")
            return
        # Get creator from db if it exists.
        c = await self.bot.sql.get_translator_participant_creator(
            self.bot.db, ch=ctx.channel.id, participant=ctx.author.id
        )
        if c == creator.id:
            await ctx.reply("You're already in the conversation.")
        elif c:
            try:
                await self.bot.sql.update_translator_participant(
                    self.bot.db, creator=creator.id, ch=ctx.channel.id, participant=ctx.author.id
                )
                await self.bot.sql.clean_translator_convo(self.bot.db, creator=c, ch=ctx.channel.id)
                await ctx.reply("You've left the original conversation and joined this one!")
            except sqlite3.IntegrityError:
                await ctx.reply("The member has no conversation in this channel!")
        else:
            try:
                await self.bot.sql.join_translator_convo(
                    self.bot.db, creator=creator.id, ch=ctx.channel.id, participant=ctx.author.id)
                await ctx.reply('Successfully joined the conversation.')
            except sqlite3.IntegrityError:
                await ctx.reply("The member has no conversation in this channel!")

    @translator.command(aliases=('l',))
    async def leave(self, ctx):
        """Leaves the current conversation"""
        c = await self.bot.sql.get_translator_participant_creator(
            self.bot.db, ch=ctx.channel.id, participant=ctx.author.id
        )
        if c:
            await self.bot.sql.remove_translator_participant(self.bot.db, ch=ctx.channel.id, participant=ctx.author.id)
            await self.bot.sql.clean_translator_convo(self.bot.db, creator=c, ch=ctx.channel.id)
            await ctx.reply("You've left the conversation.")
        else:
            await ctx.reply("Bruh you're not in any conversation")

    @translator.command(aliases=('t',))
    async def topic(self, ctx, *, t: str = None):
        """Changes the topic of the conversation"""
        if t and len(t) > 100:
            await ctx.reply('Length of topic must be <100 characters')
            return
        # idk why but if we use ! when defining statements in .sql files for aiosql,
        # it doesn't return affected rows. I have to use this gay method
        async with self.bot.sql.update_translator_topic_cursor(
                self.bot.db, creator=ctx.author.id, ch=ctx.channel.id, topic=t
        ) as c:
            if c.rowcount:
                await ctx.reply('Topic updated')
            else:
                await ctx.reply("Failed to update topic. You aren't the creator, "
                                "or you aren't in any conversations in this channel at all. ")

    @translator.command(aliases=('k',))
    async def kick(self, ctx, target: discord.Member):
        """Kicks the participant from the conversation"""
        await self.bot.sql.kick_translator_participant(
            self.bot.db, creator=ctx.author.id, ch=ctx.channel.id, participant=target.id
        )
        await ctx.reply('Done')

    @translator.command(aliases=('d',))
    async def delete(self, ctx):
        """Deletes the conversation if you're the creator"""
        await self.bot.sql.remove_translator_convo(self.bot.db, creator=ctx.author.id, ch=ctx.channel.id)
        await ctx.reply('Done')

    @commands.command(brief='DALL-E 2 image generation')
    @missile.cooldown_no_rainbow(1, 600, commands.BucketType.user)
    async def de(self, ctx, *, prompt: str):
        try:
            async with ctx.typing():
                resp = await openai.Image.acreate(prompt=prompt, size='256x256')
                await ctx.reply(resp['data'][0]['url'])
        except OpenAIError as e:
            await ctx.reply('Hmm... Who am I? What is DimBot? Sus\n' + e.user_message)

    @commands.command(brief='DALL-E 3 image generation')
    @missile.cooldown_no_rainbow(1, 1200, commands.BucketType.user)
    async def de3(self, ctx, *, prompt: str):
        try:
            async with ctx.typing():
                resp = await openai.Image.acreate(prompt=prompt, model='dall-e-3')
                await ctx.reply(resp['data'][0]['url'])
        except OpenAIError as e:
            await ctx.reply('Hmm... Who am I? What is DimBot? Sus\n' + e.user_message)
