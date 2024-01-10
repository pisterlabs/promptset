import json
import os
import random
import time
from pprint import pprint
from typing import List

import discord
import openai
from discord import Message, Member, Reaction, TextChannel, File
from dotenv import load_dotenv
import re

regex = r"^((?:R|SS|SH|SM|H|M|B))(?::(\d+(?:\.\d+)?))?(?:=|:)(?:(\d+)(?:,(A|N|F|L))?|(.+)):\s*(.+)"
load_dotenv('.env_emoji')

OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
BOT_TOKEN = os.getenv('BOT_TOKEN')

openai.api_key = OPEN_AI_KEY

LOAD_FILE = 'character.json'
FORMATTED_LOAD_FILE = 'formatted_character.json'

system = {
    "role": "system",
    "content": "You are a very good game master, creating a game where Emojis battle each other.\n\nYou are tasked to create the character sheet for each Emoji that the user inputs. As such, you must respond with the following rules.\n\nYou must set the base HP of the Emoji as \"HP={number here}\"\n\nYou must respond with a single value or a combination of the following. All the numbers must be positive.\nFor ranged attacks, \"R={input number here},{method}\".  The number means how much the range of attack will be. The method must be one of the following: \"A,\" which means all the people will get that damage; \"F,\" which means only the first person will be damaged. \"L\" means the last person will be damaged.\nFor melee attacks, \"M={input number here}\". It is the amount the person will attack the person ahead of them.\nFor healing, \"H={heal amount}{method}\". The number is the amount that will be added back to the health. The method is the following: \"F\" means the first person will heal that amount, and \"N\" means the people next to this will heal that amount.\nFor blocking, \"B={max blocking amount}\". The number will be the max blocking amount. When played, a random value will be chosen within \"[0, max blocking amount]\" inclusively.\n\nSome emojis will have special abilities:\n\"SS={emoji}\" will summon this emoji once it dies,\n\"SH={heal amount}\" will heal + increase max heal of the emoji if it kills another emoji\n\"SM={heal amount}\" will increase all damage health it kills another emoji\n\nThe emoji can be multiple of each attack mode; however, a probability must be associated with each of them. Each attack mode must be followed by a one-sentence explanation of how the emoji would perform that action. This must reflect the specific emoji.\n\nExample:\n\"User: 🤮\"\n\"\nHP=3\nR:0.5=1,A: Pukes on everyone!\nM:0.5=2: Stumbles and hits them\n\"\nor \n\"User: 🤓\"\n\"\nHP=5\nR:0.7=1,A: Everyone gets bored!\nR:0.3=5,F: This nerd goes to the gym!\n\"\nor\n\"User: 💪🏻\"\n\"\nHP=2\nM=10: Folds you in two.\nSM=3: My man goes to the gym!\n\"\nor\n\"User: 🤰🏻\"\n\"\nHP=10\nB=5: Motherly instinct to protect baby\nSS=👶🏻\n\"\nor\n\"User: 👶🏻\"\n\"\nHP=1\nM=4: Crying attack!\nSH=3: My boy is growing up!\n\"\n\nIn your response, do not include the quotation marks. The range for each of the numbers should be between 1-10."
}

USE_HISTORY = False

MAX_EMOJIS = 4
MAX_CHARACTER_LENGTH = 1500

MAX_RUNS = 200

GAME_RECORDS = "games.json"

BOT_ID = 1160319126930210990

# define a retry decorator
def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def get_emoji_data(emoji: str, history: list):
    try:
        user_prompt = {
            'role': 'user',
            'content': emoji
        }

        history.append(user_prompt)

        gpt_input = history

        if not USE_HISTORY:
            gpt_input = [
                system, user_prompt
            ]

        response = completions_with_backoff(
            model="gpt-4",
            messages=gpt_input,
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        response = response.choices[0].message.content
        history.append({
            'role': 'assistant',
            'content': response
        })

        return response
    except openai.error.OpenAIError as e:
        print(e)
        print(e.http_status)
        print(e.error)


def generate_emojis(extra_emojis):
    start = 0x1f600
    end = 0x1f644

    n = end - start

    print('Using ', n, ' emojis')

    emojis = []

    for i in range(n):
        print(hex(start), start, chr(start))
        emojis.append(chr(start))

        start += 1

    emojis.extend(extra_emojis)

    emojis = set(emojis)

    print(emojis)

    character_data = {'characters': {}, 'history': []}

    chatpgpt_history = [system]

    if os.path.exists(LOAD_FILE):
        with open(LOAD_FILE, 'r') as f:
            character_data = json.load(f)

        for c in character_data['characters']:
            if c in emojis:
                emojis.remove(c)

        chatpgpt_history = character_data['history']

    if os.path.exists(FORMATTED_LOAD_FILE):
        with open(FORMATTED_LOAD_FILE, 'r') as f:
            character_data_formatted = json.load(f)

        for c, v in character_data_formatted.items():
            print(c)
            for a in v['specials']:
                if a is None:
                    continue

                q = a['type']

                print(q, )
                if q == 'SS':
                    e = a['summon']

                    print("Summon", e)
                    if e not in character_data_formatted:
                        emojis.add(e)

    print(emojis)

    emojis = list(emojis)

    random.shuffle(emojis)

    new = False

    for i, emoji in enumerate(emojis):
        emoji_data = get_emoji_data(emoji, chatpgpt_history)

        if emoji_data is not None:

            character_data['characters'][emoji] = emoji_data
            character_data['history'] = chatpgpt_history

            print(emoji, (i + 1) / len(emojis) * 100)
            print(emoji_data)
            new = True

            with open(LOAD_FILE, 'w') as f:
                json.dump(character_data, f)

            # time.sleep(10)
        else:
            print("Failure with emoji", emoji)

    return new


def extract_attack(attack):
    matches = re.finditer(regex, attack)

    for matchNum, match in enumerate(matches, start=1):

        # print("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum, start=match.start(),
        #                                                                     end=match.end(), match=match.group()))

        data = {}

        groups = match.groups()

        for i, group in enumerate(groups):
            if i == 0:
                data['type'] = group
            if i == 1:
                if group is None:
                    data['p'] = 1.0
                else:
                    data['p'] = float(group)
            if i == 2:
                if group is not None:
                    data['amount'] = int(group)
            if i == 3:
                if group is not None:
                    data['mode'] = group
            if i == 4:
                if group is not None:
                    data['summon'] = group
            if i == 5:
                if group is not None:
                    data['text'] = group.strip()

        return data


def reformat_emojis():
    with open(LOAD_FILE, 'r') as f:
        data = json.load(f)
        characters_raw = data['characters']

    formatted_characters = dict()

    for c, val in characters_raw.items():

        attacks = val.strip().split('\n')

        moves = []

        formatted_characters[c] = dict()

        formatted_characters[c]['HP'] = int(attacks[0].split('=')[-1])

        special_moves = []

        for attack in attacks[1:]:
            output = extract_attack(attack)
            print(output, attack)
            if output['type'].startswith("S"):
                special_moves.append(output)
            else:
                moves.append(output)

        formatted_characters[c]['attacks'] = moves
        formatted_characters[c]['specials'] = special_moves

    pprint(formatted_characters)

    with open(FORMATTED_LOAD_FILE, 'w') as f:
        json.dump(formatted_characters, f)


class User:
    def __init__(self, game, discord_user: Member, hp=10):
        self.game = game
        self.discord_user = discord_user
        self.base_deck = []
        self.rep = []
        self.hp = hp

        self.deck = []
        self.hps = []
        self.initial_ranges = []
        self.blocks = []

    def add(self, emoji, index=None):
        # print(emoji, index)
        if index is None:
            self.base_deck.append(emoji)
        elif len(self.base_deck) < MAX_EMOJIS:
            self.base_deck.insert(index, emoji)
        elif index < len(self.base_deck):
            self.replace(emoji, index)
        else:
            self.base_deck.append(emoji)

    def swap(self, index1, index2):
        self.base_deck[index1], self.base_deck[index2] = self.base_deck[index2], self.base_deck[index1]

    def replace(self, emoji, index):
        self.base_deck[index] = emoji

    def reset(self):
        self.deck = self.base_deck.copy()
        self.hps = [[i, Game.EMOJIS[e]['HP'], 0] for i, e in enumerate(
            self.deck)]  # TODO NEED TO MAKE THIS A COPY OF EMOJI REPRESENTATION ENHANCE DICTIONARY WITH EMOJI + EXTRA DAMMAGE QUANTIFIER
        self.initial_ranges = self.ranges_moves()
        self.reset_blocks()

    def reset_blocks(self):
        self.blocks = [0 for _ in range(len(self.deck))]

    def step(self):
        if len(self.initial_ranges) > 0:
            return self.initial_ranges.pop(0), True
        else:
            first = max(self.hps, key=lambda x: x[0])[0]

            emoji = self.deck[first]

            emoji_ = Game.EMOJIS[emoji]

            melee = []
            prob = []

            ranges = []
            ranges_prob = []

            for a in emoji_['attacks']:
                if a['type'] == 'R':
                    ranges.append(a)
                    ranges_prob.append(a['p'])
                else:
                    melee.append(a)
                    prob.append(a['p'])

            if len(melee) > 0:
                move = random.choices(melee, prob)
            else:
                move = random.choices(ranges, ranges_prob)

            move = move[0]

            damage = True

            if move['type'] == 'B':
                self.blocks[first] += move['amount']
                damage = False

            elif move['type'] == 'H':
                amount = move['amount']

                if move['mode'] == 'F':
                    self.hps[-1][1] += amount

                    self.hps[-1][1] = min(Game.EMOJIS[self.deck[self.hps[-1][0]]]['HP'] * 2, self.hps[-1][1])
                elif move['mode'] == 'N':
                    for i in range(len(self.hps)):
                        if self.hps[i][0] == i:
                            prev = i - 1
                            ne = i + 1

                            if prev > 0:
                                self.hps[prev][1] += amount

                                self.hps[prev][1] = min(Game.EMOJIS[self.deck[self.hps[prev][0]]]['HP'] * 2,
                                                        self.hps[prev][1])
                            if ne < len(self.hps):
                                self.hps[ne][1] += amount

                                self.hps[ne][1] = min(Game.EMOJIS[self.deck[self.hps[ne][0]]]['HP'] * 2,
                                                      self.hps[ne][1])
                damage = False

            elif move['type'] == 'M':
                move = move.copy()
                move['amount'] += self.hps[first][2]

            return (first, self.deck[first], move), damage

    def ranges_moves(self):
        moves = []

        range_moves = 0

        for i, _, _ in self.hps:
            e = self.deck[i]
            emoji = Game.EMOJIS[e]

            probabilities = [m['p'] for m in emoji['attacks']]
            indexes = list(range(len(probabilities)))

            choice = random.choices(indexes, weights=probabilities)[0]
            move = emoji['attacks'][choice]

            if move['type'] == "R":
                moves.insert(range_moves, [i, e, move])
                range_moves += 1

        return moves

    def manage_hps(self):
        deaths = []
        summons = []

        i = len(self.hps) - 1

        while i >= 0:
            if self.hps[i][1] <= 0:
                hp = self.hps[i]

                deaths.append([hp[0], self.deck[hp[0]]])

                if len(self.initial_ranges) > 0:
                    for at in range(len(self.initial_ranges)):
                        if self.initial_ranges[at][0] == i:
                            self.initial_ranges.pop(at)
                            break

                specials = Game.EMOJIS[self.deck[hp[0]]]['specials']

                self.deck.pop(i)
                self.hps.pop(i)

                summoned = -1

                shift = 0

                for s in specials:
                    if s['type'] == 'SS':
                        a = s['summon']

                        self.deck.insert(i + shift, a)
                        self.hps.insert(i + shift, [i + shift, Game.EMOJIS[a]['HP'], 0])

                        summons.append(a)

                        summoned += 1
                        shift += 1

                for index in range(len(self.hps)):
                    if self.hps[index][0] > i + shift:
                        self.hps[index][0] += summoned

            i -= 1
        return deaths, summons

    def hit(self, move):
        _, emoji, move = move

        amount = move['amount']

        damage = 0

        if move['type'] == 'M':
            damage = min(self.blocks[-1] - amount, 0)

            self.game.text_data.append(f"{self.deck[-1]} was dealt melee damage of {damage}")

            self.hps[-1][1] += damage

        elif move['type'] == 'R':
            mode = move['mode']

            if mode == 'L':
                damage = min(self.blocks[0] - amount, 0)

                self.game.text_data.append(f"{self.deck[0]} was dealt range damage of {damage}")

                self.hps[0][1] += damage
            elif mode == 'F':
                damage = min(self.blocks[-1] - amount, 0)

                self.game.text_data.append(f"{self.deck[-1]} was dealt range damage of {damage}")

                self.hps[-1][1] += damage
            elif mode == 'A':
                for i in range(len(self.hps)):
                    damage = min(self.blocks[i] - amount, 0)

                    self.game.text_data.append(f"{self.deck[i]} was dealt range damage of {damage}")

                    self.hps[i][1] += damage

        deaths, summons = self.manage_hps()

        return deaths, summons

    def special(self, move):
        index, _, _ = move

        # if index >= len(self.deck):
        #     print("ERROR Index out of Bound", self.deck, index)
        #     return

        emoji = self.deck[index]

        # print("Checking specials for", emoji, index)

        for s in Game.EMOJIS[emoji]['specials']:
            if s['type'] == 'SH':
                self.game.text_data.append(f"{emoji} {s['text']} by {s['amount']}")
                # print("Performing Max Healing")
                for i in range(len(self.hps)):
                    if self.hps[i][0] == index:
                        self.hps[i][1] += s['amount']
            elif s['type'] == 'SM':
                self.game.text_data.append(f"{emoji} {s['text']} by {s['amount']}")
                # print("Performing Max Attack")
                for i in range(len(self.hps)):
                    if self.hps[i][0] == index:
                        self.hps[i][2] += s['amount']


class Game:
    EMOJIS = dict()

    def __init__(self, users: List[Member]):
        self.user1 = User(self, users[0])
        self.user2 = User(self, users[1])

        # self.user1.add('🤠')
        # self.user1.add('🌞')
        # self.user1.add('🐆')
        # self.user1.add('🐯')
        #
        # self.user2.add('☃️')
        # self.user2.add('🥶')
        # self.user2.add('🐻')
        # self.user2.add('☠️')

        self.count = 0

        self.users = [self.user1, self.user2]

        self.member = dict(zip(users, self.users))

        self.user_flip = random.randint(0, len(self.users) - 1)

        self.money_per_user = dict()
        self.done_shop_per_user = dict()

        self.user_shop = {}
        self.reset_shop()
        self.reset_done_buying(True)

        self.text_data = []

        self.ai_expected = []

    def reset_shop(self):
        self.reset_money()
        self.reset_done_buying()

        for user in self.users:
            self.shop(user)

    def finished_shopping(self, user):
        if user in self.done_shop_per_user:
            self.done_shop_per_user[user] = True

    def shop_is_done(self):
        return all(self.done_shop_per_user.values())

    def get_user(self, discord_member):
        return self.member[discord_member]

    def reset_user_moves(self):
        for u in self.users:
            u.reset()

    def prepare_game(self):
        self.reset_user_moves()

        self.user_flip = random.randint(0, len(self.users) - 1)

        self.text_data = []

        self.count = 0

    def reset_blocks(self, active_user):
        for u in self.users:
            if u == active_user:
                u.reset_blocks()

    def step(self):
        # print("USER's", self.user_flip, "TURN")

        current = self.users[self.user_flip]
        target = self.users[1 - self.user_flip]

        self.text_data.append(f"{current.discord_user.display_name}'s turn ({self.count})")

        self.reset_blocks(current)

        output = current.step()
        move, damage = output

        self.text_data.append(
            f"{move[1]} {move[2]['text']} for {move[2]['type']}={move[2]['amount']}" + (
                f",{move[2]['mode']}" if 'mode' in move[2] else ''))

        if damage:
            death, summons = target.hit(move)

            if len(death) > 0:
                self.text_data.extend([f'{e[1]} was killed' for e in death])
                # print("DEATHS", death)
                current.special(move)
            if len(summons) > 0:
                self.text_data.append(f"{target.discord_user.display_name} summons:")
                self.text_data.extend([f'{e} was summoned' for e in summons])
                # print("SUMMONS", summons)

        self.text_data.append("End of round healths: ")

        text = ''
        for e, hp in zip(current.deck, current.hps):
            text += f'{e}={hp[1]} '

        text += '\n'
        for e, hp in zip(target.deck, target.hps):
            text += f'{e}={hp[1]} '

        self.text_data.append(text)
        self.text_data.append('')

        self.user_flip = 1 - self.user_flip
        self.count += 1

        # print(target.blocks)

        return move

    def round_winner(self):
        if all([len(u.hps) == 0 for u in self.users]):
            return -1
        if len(self.user1.hps) == 0:
            return 1
        elif len(self.user2.hps) == 0:
            return 0
        else:
            return -1

    def shop(self, user):
        emojis_to_buy = random.choices(list(self.EMOJIS.keys()), k=4)

        self.user_shop[user] = emojis_to_buy

        return emojis_to_buy

    def purchase(self, user, amount=3):
        if self.money_per_user[user] >= amount:
            self.money_per_user[user] -= amount

            return True
        return False

    def reroll(self, user):
        if self.money_per_user[user] >= 3:
            self.purchase(user)
            return self.shop(user)

        return None

    def buy_emoji(self, user: User, emoji: str, place_index: int):
        index = self.user_shop[user].index(emoji)

        if index != -1 and self.purchase(user):
            emoji = self.user_shop[user].pop(index)

            user.add(emoji, place_index)

            return True
        return False

    def finished(self):
        return self.round_winner() != -1 or (self.count // 2) > MAX_RUNS

    def reset_money(self):
        self.money_per_user = dict((u, 9) for u in self.users)

    def reset_done_buying(self, val=False):
        self.done_shop_per_user = dict((u, val) for u in self.users)

    def calculate_expected(self, e, use_range=True):
        e = Game.EMOJIS[e]

        ranges_p = []
        ranges = []

        melee_p = []
        melee = []

        total_m = 0

        total = 0

        for i in e['attacks']:
            if i['type'] == "R":
                ranges.append(i['amount'] * (4 if i['mode'] == "A" else 1))
                ranges_p.append(i['p'])
            elif i['type'] == "M":
                melee.append(i['amount'])

                total_m += i['p']
                melee_p.append(i['p'])
            total += i['p']

        if use_range:
            r_s = total
        else:
            r_s = sum(ranges_p)
        ranges_p = sum([p / r_s * a for p, a in zip(ranges_p, ranges)])
        m_s = total_m
        melee_p = sum([p / m_s * a for p, a in zip(melee_p, melee)])

        if len(melee) > 0 and not use_range:
            ranges_p = 0

        total = ranges_p + melee_p + e['HP']

        for i in e['specials']:
            if i['type'] == "SS":
                total += self.calculate_expected(i['summon'], False)

        return total

    async def ai_purchase(self, channel, other_user):
        exp = other_user.base_deck

        expected = [(i, self.calculate_expected(e)) for i, e in enumerate(exp)]

        shop = self.user_shop[other_user]

        expected_shop = [(i, self.calculate_expected(e)) for i, e in enumerate(shop)]
        expected_shop = sorted(expected_shop, key=lambda x: x[1], reverse=True)

        if len(exp) > 0:
            min_expected = min(expected, key=lambda x: x[1])
        else:
            min_expected = []

        if len(expected) == 0:
            index = random.randint(0, MAX_EMOJIS)
            emoji = shop[expected_shop[0][0]]

            continue_buying = self.buy_emoji(other_user, emoji, index)

            if continue_buying:
                await channel.send(
                    f"{other_user.discord_user.mention} Bought {emoji} putting it in index {index}")
        elif len(shop) == 0 or (len(exp) == MAX_EMOJIS and min_expected[1] > expected_shop[0][1]):
            continue_buying = self.reroll(other_user) is not None

            if continue_buying:
                await channel.send(
                    f"{other_user.discord_user.mention} Re-rolled new shop {' '.join(self.user_shop[other_user])}")
        else:
            index = min_expected[0]
            emoji = shop[expected_shop[0][0]]
            continue_buying = self.buy_emoji(other_user, shop[expected_shop[0][0]], min_expected[0])

            if continue_buying:
                await channel.send(
                    f"{other_user.discord_user.mention} Bought {emoji} putting it in index {index}")

        if continue_buying:
            await self.ai_purchase(channel, other_user)


class EmojiGame(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

        self.running_games = dict()
        self.game_messages = dict()

        if os.path.exists(GAME_RECORDS):
            with open(GAME_RECORDS, 'r') as f:
                self.game_records = json.load(f)
        else:
            self.game_records = dict()

    async def on_reaction_add(self, reaction: Reaction, user):
        if user == self.user:
            return

        message = reaction.message

        if message not in self.game_messages:
            return

        game_user = self.game_messages[message]

        if not game_user.discord_user == user:
            return

        game: Game = game_user.game

        if reaction.emoji == '🐛':
            for e in game.user_shop[game_user]:
                # print(e)
                await message.add_reaction(e)
            return

        if reaction.emoji == '📃':
            await self.show_emoji_info(message.channel, game_user)

        if reaction.emoji == '🔁':
            rerolled = game.reroll(game_user)

            if rerolled:
                await self.display_shop(message.channel, game_user.game, game_user)
            else:
                await reaction.message.channel.send(
                    f"{game_user.discord_user.mention} Unable to reroll, not enough money!")

            return

        if reaction.emoji == "✅":
            game.finished_shopping(game_user)

            other_user: User = game.users[1 - game.users.index(game_user)]

            await message.channel.send(f"{game_user.discord_user.mention} Finished shopping!")

            del self.game_messages[message]

            if other_user.discord_user == self.user:
                await game.ai_purchase(message.channel, other_user)

                game.finished_shopping(other_user)
                await message.channel.send(f"{other_user.discord_user.mention} Finished shopping!")

        if game.shop_is_done():
            await self.play_game(message.channel, game)

            return

        counter = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]

        emoji = None

        index = None

        avalaible_shop_emojis = game_user.game.user_shop[game_user]

        for reaction in message.reactions:
            if reaction.count > 1:
                if reaction.emoji in avalaible_shop_emojis:
                    emoji = reaction.emoji
                elif reaction.emoji in counter:
                    index = counter.index(reaction.emoji)

        if not emoji is None and not index is None:
            # print(emoji, index)
            bought = game.buy_emoji(game_user, emoji, index)

            if bought:
                await reaction.message.channel.send(
                    f"{game_user.discord_user.mention} Bought {emoji} putting it in index {index}")
                await self.display_shop(message.channel, game_user.game, game_user)
            else:
                await reaction.message.channel.send(f"{game_user.discord_user.mention} Unable to buy {emoji}")

    def record_game(self, game, end):
        for i, u in enumerate(game.users):
            u1 = str(u.discord_user.id)

            if u1 not in self.game_records:
                self.game_records[u1] = {
                    'wins': 0,
                    'total': 0,
                    'draws': 0,
                    'losses': 0,
                    'games': {}
                }

            if end == -1:
                self.game_records[u1]['draws'] += 1
            elif end == i:
                self.game_records[u1]['wins'] += 1
            else:
                self.game_records[u1]['losses'] += 1

            other = str(game.users[1 - i].discord_user.id)

            if other not in self.game_records[u1]['games']:
                self.game_records[u1]['games'][other] = {
                    'wins': 0,
                    'losses': 0,
                    'draws': 0,
                    'total': 0
                }

            battle_data = self.game_records[u1]['games'][other]

            if end == -1:
                battle_data['draws'] += 1
            elif end == i:
                battle_data['wins'] += 1
            else:
                battle_data['losses'] += 1

            self.game_records[u1]['total'] += 1
            battle_data['total'] += 1

        with open(GAME_RECORDS, 'w') as f:
            json.dump(self.game_records, f)

    async def display_shop(self, channel: discord.channel.TextChannel, game, user: User):
        shop_items = game.user_shop[user]

        embed = discord.Embed(title="Shop",
                              description=f"This is user {user.discord_user.mention}'s shop",
                              color=discord.Color.yellow())

        embed.add_field(name="Money", value=str(game.money_per_user[user]))
        embed.add_field(name="Current Deck", value=" ".join(user.base_deck))
        embed.add_field(name="Emojis", value=" ".join(shop_items), inline=False)

        message = await channel.send(embed=embed)

        for e in shop_items:
            await message.add_reaction(e)

        await message.add_reaction("1️⃣")
        await message.add_reaction("2️⃣")
        await message.add_reaction("3️⃣")
        await message.add_reaction("4️⃣")
        await message.add_reaction("🔁")
        # await message.add_reaction('🐛')
        await message.add_reaction('📃')

        await message.add_reaction("✅")

        self.game_messages[message] = user

    async def show_stat(self, channel, user_id):
        if isinstance(user_id, int):
            user_id = str(user_id)

        if user_id not in self.game_records:
            await channel.send(
                "Sorry you have never played. Clearly you think you are better than the rest of us...")
        else:
            user = await self.fetch_user(int(user_id))
            embed = discord.Embed(title="Player Stats",
                                  description=f"Info for {user.mention}",
                                  color=discord.Color.blue())

            user_data = self.game_records[user_id]

            embed.add_field(name="Wins", value=user_data["wins"], inline=True)
            embed.add_field(name="Draws", value=user_data["draws"], inline=True)
            embed.add_field(name="Losses", value=user_data["losses"], inline=True)
            embed.add_field(name="Total", value=user_data["total"], inline=True)

            data = []
            for u, v in user_data['games'].items():
                user = await self.fetch_user(int(u))

                data.append(
                    f'{user.display_name} T:{v["total"]} W:{v["wins"]} D:{v["draws"]} L:{v["losses"]}')

            embed.add_field(name="Battles", value='\n'.join(data), inline=False)

            await channel.send(embed=embed)

    async def on_message(self, message: Message):
        if message.author.bot:
            return

        playing_users = {message.author}

        if len(message.mentions) > 0:
            if not message.mentions[0] == self.user:
                return

            if not len(message.mentions) == 1 and not len(message.mentions) == 2:
                await message.channel.send("Please either mention 1 or 2 users to play with.")
                return

            clean_content = re.sub("<[@#!&](.*?)>", "", message.content).strip()

            if len(message.mentions) == 1:
                playing_users.add(self.user)

                if clean_content in Game.EMOJIS:
                    embed = discord.Embed(title="Emoji Info",
                                          description=f"Info for {clean_content}",
                                          color=discord.Color.light_grey())

                    self.generate_character_sheet(embed, clean_content)

                    await message.channel.send(embed=embed)

                    return
                elif clean_content.lower() == "stat":
                    bot_mention = f"<@{BOT_ID}>"

                    if message.content[len(bot_mention):].find(bot_mention) >= 0:
                        user_id = str(BOT_ID)
                    else:
                        user_id = str(message.author.id)

                    await self.show_stat(message.channel, user_id)

                    return
                elif clean_content.lower() == "leaderboard":
                    embed = discord.Embed(title="Leaderboard",
                                          description=f"Current users with most wins",
                                          color=discord.Color.blue())

                    a = []

                    for k, v in self.game_records.items():
                        user = await self.fetch_user(int(k))
                        a.append((user.display_name, v['wins']))

                    a = sorted(a, key=lambda x: x[1], reverse=True)

                    if len(a) >= 1:
                        embed.add_field(name="👑", value=f"{a[0][0]} {str(a[0][1])}", inline=False)

                        if len(a) >= 2:

                            embed.add_field(name="🥈", value=f"{a[1][0]} {str(a[1][1])}", inline=False)

                            if len(a) >= 3:
                                embed.add_field(name="🥉", value=f"{a[2][0]} {str(a[2][1])}", inline=False)

                                if len(a) >= 4:
                                    embed.add_field(name="💩", value=f"{a[-1][0]} {str(a[-1][1])}", inline=False)

                    await message.channel.send(embed=embed)

                    return
                elif len(clean_content) > 0:
                    return

            else:
                next_player = message.mentions[1]

                playing_users.add(next_player)

                if clean_content == 'stat':
                    await self.show_stat(message.channel, next_player.id)
                    return
        else:
            return

        playing_users = frozenset(playing_users)

        if playing_users not in self.running_games:
            await message.channel.send("Thank you for playing 'Emoji game'™©®")

            game = Game(list(playing_users))
            self.running_games[playing_users] = game
        else:
            game = self.running_games[playing_users]

        await self.reset_shop(message.channel, game)

    async def reset_shop(self, channel, game):
        if game.shop_is_done():
            game.reset_shop()

        for user in game.users:
            await self.display_shop(channel, game, user)

    async def show_users(self, channel, game):
        embed = discord.Embed(title="Users",
                              color=discord.Color.red())

        for u in game.users:
            embed.add_field(name=f'__**{u.discord_user.display_name}**__', value='', inline=False)
            embed.add_field(name="HP", value=u.hp)
            embed.add_field(name="Deck", value=" ".join(u.base_deck))

        await channel.send(embed=embed)

    async def play_game(self, channel: TextChannel, game: Game):
        user_mentioned = " ".join(u.discord_user.mention for u in game.users)

        await channel.send(f'{user_mentioned} Running game!')
        game.prepare_game()

        await self.show_users(channel, game)

        while not game.finished():
            # print(game.user1.hps, game.user2.hps)
            # print(game.user1.deck, game.user2.deck)
            index, _, move = game.step()
            # print(index, move)
            # print()
            # print()

        # print(game.user1.hps, game.user2.hps)

        lengths = [len(t) for t in game.text_data[::-1]]

        text_data = game.text_data

        if sum(lengths) > MAX_CHARACTER_LENGTH:
            file_name = f"{game.users[0].discord_user.display_name}-{game.users[1].discord_user.display_name}.txt"

            with open(file_name, 'w', encoding='utf-8') as f:
                f.write("\n".join(text_data))

            await channel.send(file=File(file_name, "run.txt"))

            os.remove(file_name)

            # text_data = []
            #
            # total_count = 0
            #
            # data = game.text_data[::-1]
            #
            # i = 0
            # while i < len(data):
            #     d = data[i]
            #     l = len(d)
            #
            #     if total_count + l < MAX_CHARACTER_LENGTH:
            #         text_data.append(d)
            #         total_count += l
            #     else:
            #         break
            #
            #     i += 1
            #
            # text_data = text_data[::-1]
        else:
            if len(text_data) > 0:
                info = "```" + "\n".join(text_data) + "```"

                await channel.send(info)

        if game.round_winner() == -1:
            for u in game.users:
                u.hp -= 1

            await channel.send(f'The round ended in a draw.')
        else:
            game.users[1 - game.round_winner()].hp -= 3

            await channel.send(f"This round's winner is {game.users[game.round_winner()].discord_user.mention}.")

        await self.show_users(channel, game)

        hps = [u.hp <= 0 for u in game.users]

        if any(hps):
            result = -1

            if all(hps):
                await channel.send(f"{user_mentioned} THE GAME HAS ENDED IN A DRAW THERE IS NO WINNER")
            else:
                index = hps.index(False)
                m = game.users[index].discord_user.mention

                result = index

                await channel.send(f"{user_mentioned} THE GAME HAS ENDED THE WINNER IS {m}")

            self.record_game(game, result)
            data = frozenset(set(u.discord_user for u in game.users))
            del self.running_games[data]

            print("RUNNING GAMES", self.running_games)
        else:
            await self.reset_shop(channel, game)

    def generate_character_sheet(self, embed, e):
        info = Game.EMOJIS[e]

        normal = info['attacks']

        embed.add_field(name=f"{e}, HP={info['HP']}", value='', inline=False)

        string = []
        for a in normal:
            string.append(f'{a["type"]}:{a["p"]}={a["amount"]}' + (f",{a['mode']}" if "mode" in a else ""))

        embed.add_field(name="Normal", value="\n".join(string), inline=True)

        special = info['specials']
        string = []
        for a in special:
            if a['type'] == "SS":
                string.append(f'{a["type"]}={a["summon"]}')
            else:
                string.append(f'{a["type"]}={a["amount"]}')

        embed.add_field(name="Specials", value="\n".join(string), inline=True)

    async def show_emoji_info(self, channel, user):
        embed = discord.Embed(title="Emoji Info",
                              description=f"Description of the Emojis for {user.discord_user.mention}'s deck and shop.",
                              color=discord.Color.light_gray())

        embed.add_field(name="__**Base Deck**__", value="", inline=False)

        for e in user.base_deck:
            e: str
            self.generate_character_sheet(embed, e)

        embed.add_field(name="__**Shop**__", value="", inline=False)

        for e in user.game.user_shop[user]:
            self.generate_character_sheet(embed, e)

        await channel.send(embed=embed)


if __name__ == '__main__':
    # random.seed(42)
    generate = False

    if generate:
        extra_emojis = ['🥶', '💀', '🤑', '🤐', '🥵']
        battle_emojis = [
            "😀", "😁", "😂", "🤣", "😃", "😄", "😅", "😆", "😉", "😊",
            "😋", "😎", "😍", "😘", "😗", "😙", "😚", "☺️", "🙂", "🤗",
            "🤩", "🤔", "🤨", "😐", "😑", "😶", "🙄", "😏", "😣", "😥",
            "😮", "🤐", "😯", "😪", "😫", "😴", "😌", "😛", "😜", "😝",
            "🤤", "😒", "😓", "😔", "😕", "🙃", "🤑", "😲", "🙁", "😖",
            "😞", "😟", "😤", "😢", "😭", "😦", "😧", "😨", "😩", "🤯",
            "😬", "😰", "😱", "😳", "🤪", "😵", "😡", "😠", "🤬", "😷",
            "🤒", "🤕", "🤢", "🤮", "🤧", "😇", "🤠", "🤡", "🤥", "🤫",
            "🤭", "🧐", "🤓", "😈", "👿", "👹", "👺", "💀", "☠️", "👻",
            "👽", "👾", "🤖", "💩", "🙊", "🙉", "🙈", "🐵", "🐶", "🐺",
            "🦊", "🦝", "🐱", "🐈", "🦁", "🐯", "🐅", "🐆", "🐴", "🐎",
            "🦄", "🐮", "🐂", "🐃", "🐄", "🐷", "🐖", "🐗", "🐽", "🐏",
            "🐑", "🐐", "🐪", "🐫", "🦙", "🦒", "🐘", "🦏", "🦛", "🐭",
            "🐁", "🐀", "🐹", "🐰", "🐇", "🐿️", "🦔", "🦇", "🐻", "🐨", '🖕', '🌞',
            '🧀', '🛸', '🥕', '🔥', '🌈', '🍯'
        ]

        extra_emojis.extend(battle_emojis)

        new = True

        while new:
            new = generate_emojis(extra_emojis)

            reformat_emojis()

            if new:
                print("NEW EMOJI GOING AGAIN")

    with open(FORMATTED_LOAD_FILE, 'r') as f:
        Game.EMOJIS = json.load(f)

    print("STARTING DISCORD")

    # class Temp:
    #     def __init__(self, name):
    #         self.display_name = name

    # random.seed(42)
    # game = Game([Temp("A"), Temp("B")])
    #
    # game.prepare_game()
    #
    # while not game.finished():
    #     game.step()
    #
    # print(game.text_data)

    intents = discord.Intents.default()
    intents.message_content = True

    client = EmojiGame(intents=intents)

    client.run(BOT_TOKEN)

# SAVE SCOREBOARD OF USERS WINNING
# ADD ITEMS TO PERMANTLY BUFF
