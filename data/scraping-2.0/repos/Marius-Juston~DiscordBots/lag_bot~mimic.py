import discord
import os
import random
import datetime
import json
import openai
import openai.error
from dotenv import load_dotenv

load_dotenv('.env_mimic')


OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')

MARIUS_ID = os.getenv('MARIUS_ID')
BOT_ID = os.getenv('BOT_ID')
GUILD_ID = os.getenv('GUILD_ID')
BOT_TOKEN = os.getenv('BOT_TOKEN')

openai.api_key = OPEN_AI_KEY


system = {"role": "system", "content": "keep it short, brief and the responses should only be 20 words, do not go over that limit. You mimic all the users that come offline so that when people try talking to that specific user, you respond like them. However, you cannot change your personality very fast due to limits to your power. So you get sassy and extremely annoyed when someone tries to ask for someone you cannot become while staying in character. The person's roles help describe to you what kind of person they are. You must do everything you can not to let people figure out you are a fake otherwise YOU WILL die. All the people you talk to are friends on the person you are faking, not some random person, act like it and act casual. You should NEVER mention your powers and what you are. YOU WILL DIE IF YOU REVEAL YOURSELF. People from Royalty should talk high and mighty, people form Gamblers Anonymous should be addicted to betting, an Peasants should always lower themselves"}


class _JSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        ret = {}
        for key, value in obj.items():
            print(key)
            if key in {'last_edit_time'}:
                ret[key] = datetime.datetime.fromisoformat(value) 
            else:
                ret[key] = value
        return ret
    
class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return json.JSONEncoder.default(obj)

class MyClient(discord.Client):

    async def mimic_user(self, user):
        author_avatar = user.display_avatar

        url = author_avatar.url

        avatar_data = None

        name = self.get_user_name(user)

        local_global_name = self.get_user_local_global_name(user)

        if local_global_name != self.config_data['local_global_name']:
            if url in self.user_icon_cache:
                avatar_data = self.user_icon_cache[url]
            else:
                avatar_data = await author_avatar.read()
                self.user_icon_cache[url] = avatar_data
            try:
                await self.user.edit(avatar=avatar_data, username=name)
                self.set_config('local_global_name', local_global_name)
                self.set_config('user_id', user.id)

                if self.config_data['last_edit_time'] is None:
                    self.set_config("last_edit_time",datetime.datetime.now() )

                else:
                    delta_time = datetime.datetime.now() - self.config_data['last_edit_time']

                    print(delta_time)

                    if delta_time.total_seconds() < self.edit_minutes_timeout:
                        self.set_config("last_edit_time",datetime.datetime.now() )
            except discord.errors.HTTPException:
                if self.config_data['last_edit_time'] is None:
                    self.set_config("last_edit_time",datetime.datetime.now() )
                else:
                    print("Waiting need to wait", self.edit_minutes_timeout - (datetime.datetime.now() - self.config_data['last_edit_time']).total_seconds() )

                return False
            
        return True
    
    def get_user_name(self, user):
        name = user.global_name

        if name is None:
            name = user.name

        return name
    
    def get_user_local_global_name(self, user):
        return str(user.global_name) + "-" + str(user.name)
    
    def get_flavor(self, user, name=None):
        roles = user.roles

        role_names = set(str(role) for role in roles)
        role_names.discard('Admin')
        role_names.discard('@everyone')

        name = self.get_user_name(user) if name is None else name

        if name in self.equivalence:
            name = self.equivalence[name]

        return f"{name} with roles {', '.join(role_names)}"
    
    def set_config(self, key, value):
        self.config_data[key] = value

        with open(self.config_file, 'w') as f:
            json.dump(self.config_data, f, cls=_JSONEncoder)

    async def on_ready(self):
        self.config_file = 'config.json'

        print(f'Logged on as {self.user}!')

        self.config_data = {}

        if os.path.exists(self.config_file):
            with open(self.config_file, 'rb') as f:
                self.config_data = json.load(f, cls=_JSONDecoder)

                self.config_data['message_history'] = []
        else:
            self.config_data['local_global_name'] = None
            self.config_data['user_id'] = None
            self.config_data['last_edit_time'] = None
            self.config_data['message_history'] = []

        if len(self.config_data['message_history']) == 0:
            self.config_data['message_history'].append(system)

        print(self.config_data)

        self.current_user = None

        self.user_icon_cache = {}

        self.offline_users = dict()

        self.equivalence = {
            'blueluigi26': 'Edric',
            'michu': 'Michael',
            'Kantore2': 'Alex',
            '𝗽𝗮𝗿𝗮𝒗𝗼𝗶𝗱':'Justin',
            'DysonSphere': 'Marius'
        }

        self.members_by_id = {}

        self.edit_minutes_timeout = 60 * 10 # 10 minutes

        for mem in self.get_all_members():
            self.members_by_id[mem.id]  = mem

            print(mem, self.get_user_name(mem), self.get_user_local_global_name(mem))
            if not mem.bot and mem.status == discord.Status.offline:
                self.add_offline_user(mem)

        print(self.offline_users.keys())

    def generate_chat_gpt_response(self, prompt, talk_to_the_hand=False):
        try:
            tmep2=  system.copy()
            tmep2['role'] = 'user'
            self.config_data['message_history'].append(tmep2)
            
            self.config_data['message_history'].append({'role': 'user', 'content':prompt})

            completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=
                                                    self.config_data['message_history']
                                                    )
            
            chat_response = completion.choices[0].message.content

            self.config_data['message_history'].append({'role': 'assistant', 'content':chat_response})

            self.set_config('message_history', self.config_data['message_history'])

            print(chat_response, self.config_data['message_history'])
        except openai.error.RateLimitError:
            print("UNABLE TO OUTUT CHATGPT RESULT ", talk_to_the_hand)
            if not talk_to_the_hand:
                return random.choice(["I couldn't agree more! You're an absolute genius!",
            "You're absolutely, positively, 100% right, and I salute your brilliance!",
            "You're so right that you've left no room for doubt!",
            "Your point is so valid, it should be etched in stone!",
            "You've hit the bullseye of truth with a sledgehammer!",
            "I wholeheartedly and vehemently agree with every fiber of my being!",
            "Your insight is like a blazing supernova of correctness!",
            "You've unlocked the secrets of the universe with your wisdom!",
            "Your argument is so compelling, it's like a force of nature!",
            "I'm not just agreeing, I'm practically shouting 'Amen!'",
            "I couldn't disagree more! You're completely and utterly wrong!",
            "I have to vehemently and categorically disagree with every fiber of my being!",
            "Your point is so flawed, it should be tossed into the abyss!",
            "You've missed the mark so spectacularly, it's almost impressive!",
            "I wholeheartedly and fervently disagree with every ounce of my being!",
            "Your insight is like a black hole of inaccuracy!",
            "You've managed to defy all logic and reason with your argument!",
            "I'm not just disagreeing, I'm practically shaking my head in disbelief!",
            "I couldn't disagree more! This is just madness!",
            "I have to vehemently and categorically disagree with every neuron in my brain!",
            "You're so right that even the laws of physics can't argue!",
            "I'm nodding so vigorously in agreement, I might break my neck!",
            "Your insight is like a truth bomb that just exploded!",
            "If agreeing were an Olympic sport, you'd win the gold, silver, and bronze!",
            "I'm on the same wavelength as you, and it's a frequency of pure brilliance!",
            "You've reached the pinnacle of correctness; there's no higher peak!",
            "Your argument is like a perfectly executed symphony of truth!",
            "I agree with such intensity that I might spontaneously combust!",
            "Your wisdom is so profound; it's as if you've tapped into the cosmos!",
            "You've left no room for doubt; it's an agreement for the ages!",
            "I couldn't disagree more! This is sheer lunacy!",
            "Your viewpoint is so absurd; it's like a cosmic joke!",
            "I have to dissent with such vehemence that it's almost cathartic!",
            "You've missed the mark so spectacularly; it's almost entertaining!",
            "Your argument is like a carnival of logical fallacies!",
            "You've managed to defy all reason and sanity with your stance!",
            "I'm not just disagreeing; I'm practically holding my head in disbelief!",
            "I couldn't disagree more! It's like arguing with a unicorn!",
            "I have to dissent with such fervor that it's almost an art form!",
            "You've ventured into the realm of pure fantasy with that argument!"
            ])
            else:
                return random.choice([                   
                        "Oh, I see how it is. You're clearly not interested in what I have to say.",
                        "I guess I'll just talk to myself then, since you're too busy for me.",
                        "Wow, it's like I'm invisible. I'll go find someone who actually listens.",
                        "I get it, you found someone more exciting. I'll just be over here, alone.",
                        "You know what? I've lost my train of thought anyway. Have fun with someone else!",
                        "I'll just fade into the background while you enjoy your new conversation.",
                        "Clearly, my words aren't worth your attention. I'll find someone who cares.",
                        "My insights must be incredibly boring if you'd rather talk to someone else.",
                        "I'll spare you the trouble. I'm going to find a more receptive audience.",
                        "I'll let you chat with your newfound friend. I'll find my own company.",
                        "It's crystal clear that I'm not on your radar right now. I'll go elsewhere.",
                        "I'll just take a hint and see myself out of this conversation.",
                        "I'm getting the silent treatment, so I'll just go seek conversation elsewhere.",
                        "I thought we were having a conversation, but I guess not. Enjoy your chat!",
                        "Looks like I've overstayed my welcome. I'll leave you to your other conversation.",
                        "I'll leave you to your exciting chat and find someone who appreciates me.",
                        "I'm being ghosted right before my eyes. I'll find someone who acknowledges me.",
                        "I'll go join a conversation where my presence is actually valued.",
                        "Clearly, I'm not a priority here. I'll go find someone who makes me feel important.",
                        "I'll just fade into the background while you focus on your new friend."
                    
                ])

        return chat_response
    

    def add_offline_user(self, user):
        user_name = self.get_user_name(user)

        self.offline_users[user_name] = user

        if user_name in self.equivalence:
            self.offline_users[self.equivalence[user_name]] = user


    def remove_offline_user(self, user):
        user_name = self.get_user_name(user)

        if user_name in self.offline_users:
            del self.offline_users[user_name]

            if user_name in self.equivalence:
                del self.offline_users[self.equivalence[user_name]]


    async def on_member_update(self, before, after):
        if before.status != discord.Status.offline and after.status == discord.Status.offline:
            print(f'{after.display_name} has gone offline.')

            self.add_offline_user(after)

        if before.status != discord.Status.online and after.status == discord.Status.online:
            print(f'{after.display_name} has gone online.')

            self.remove_offline_user(after)

    def contains_offline(self, text):
        for i in self.offline_users:
            if i.lower() in text.lower():
                return i
            
        return None

    async def on_message(self, message):
        if message.author != self.user:
            # print([mem == self.user for mem in  message.mentions])
            # if any(mem == self.user for mem in  message.mentions):
            text = message.content

            offline_user = self.contains_offline(text)

            response_user = None

            wanted_to_change = False 
            mimiced = False

            if offline_user:
                response_user = self.offline_users[offline_user]
            # elif len(self.offline_users) > 0:
            #     options = list(self.offline_users.keys())
            #     response_user = self.offline_users[random.choice(options)]
            # else:
            #     response_user = message.author

                mimiced = await self.mimic_user(response_user)
                wanted_to_change = True

            if not mimiced :
                response_user = self.members_by_id[self.config_data['user_id']]

                if self.get_user_name(response_user) not in self.offline_users:
                    return # Do not talk just continue on
            

            from_flavor = self.get_flavor(message.author)
            
            response_flavor = self.get_flavor(response_user)

            index = len(self.config_data['message_history'])

            self.config_data['message_history'].append({'role':'system', 'content':f'YOU ARE RESPONDING AS {response_flavor}' + ' - YOU WERE UNABLE TO CHANGE TO THE DESIRED USER' if (not mimiced and wanted_to_change) else ""})

            prompt = f'"{message.content}" by {from_flavor} - YOU ARE RESPONDING AS {response_flavor}' + (' - YOU WERE UNABLE TO CHANGE TO THE DESIRED USER' if (not mimiced and wanted_to_change) else "")
            print(prompt)

            prompt = self.generate_chat_gpt_response(prompt, not mimiced and wanted_to_change)

            self.config_data['message_history'].pop(index)

            self.set_config('message_history', self.config_data['message_history'])

            await message.channel.send(prompt)


if __name__ == "__main__":
    intents = discord.Intents.default()
    # intents.message_content = True
    # intents.members = True
    intents = intents.all()

    client = MyClient(intents=intents)
    client.run(BOT_TOKEN)
