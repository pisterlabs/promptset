import os
import random
import re
import sys
import time

import click
import requests
import openai

# TODO-list:
# x title screen
# x battle report line by line
# x color battle report
# x color cards
# x alternative model connector (e.g. GPT-J)
# - adjust mix of cards in deck by sampling from all possible
# - text coherence: add explicit initiative to text model
# - text coherence: monster ownage
# - text coherence: winner generation
# - over-generation: cut off accidental fights after first player summoning. Or a more explicit stop token?


TITLE = """

     ███▄ ▄███▓ ▄▄▄        ▄████ ▓█████    ▓█████▄  █    ██ ▓█████  ██▓
    ▓██▒▀█▀ ██▒▒████▄     ██▒ ▀█▒▓█   ▀    ▒██▀ ██▌ ██  ▓██▒▓█   ▀ ▓██▒
    ▓██    ▓██░▒██  ▀█▄  ▒██░▄▄▄░▒███      ░██   █▌▓██  ▒██░▒███   ▒██░
    ▒██    ▒██ ░██▄▄▄▄██ ░▓█  ██▓▒▓█  ▄    ░▓█▄   ▌▓▓█  ░██░▒▓█  ▄ ▒██░
    ▒██▒   ░██▒ ▓█   ▓██▒░▒▓███▀▒░▒████▒   ░▒████▓ ▒▒█████▓ ░▒████▒░██████▒
    ░ ▒░   ░  ░ ▒▒   ▓▒█░ ░▒   ▒ ░░ ▒░ ░    ▒▒▓  ▒ ░▒▓▒ ▒ ▒ ░░ ▒░ ░░ ▒░▓  ░
    ░  ░      ░  ▒   ▒▒ ░  ░   ░  ░ ░  ░    ░ ▒  ▒ ░░▒░ ░ ░  ░ ░  ░░ ░ ▒  ░
    ░      ░     ░   ▒   ░ ░   ░    ░       ░ ░  ░  ░░░ ░ ░    ░     ░ ░
           ░         ░  ░      ░    ░  ░      ░       ░        ░  ░    ░  ░
                                            ░
"""

SUBTITLE = """
  --[ A two-player competitive text generation monster battle card game! ]--
  
"""

INSTRUCTION = """
# Aim:
Two players, the blue and the red mage, are duelling.
They summon creatures which fight against each other.
The mage whose creature survives wins a round.

# Proceedure:
To summon a creature, you play cards from your hand. Play a monster card to
summon a monster, or the unique to summon a unique creature. You can modify a
monster with a prefix- and or a suffix-card if you have any. Sometimes you'll
get a blank card, a joker. You can decide on your own what the monster name, or
the prefix, or the suffix should be. You'll start with five cards and draw one
new card every round. So be sure not to use all cards too early, as the duel
has multiple rounds. 

# Note:
This is a 2-player hot-seat game.
Decide, who is the blue and who is the red mage.

"""

MONSTER_CARDS = "skeleton warrior|were rat|slime|goblin|ogre|magician|archer|death knight|clown|dragon|squirrel|pig|veliceraptor|ghost|assassin|vampire lord|shapeshifter|horror|imp|kraken|witch|basilisk|griffin|manticore|leprechaun|spider|tiger|gargoyle|Doppelgänger|lindworm|chimera|golem|gremlin|troll|dalek|cybermen".split(
    "|"
)

DEFAULT_MONSTER_CARD = "sheep"

PREFIX_CARDS = "chaos|demon|zombie|undead|angelic|neutral|evil|baby|young|teenage|grown-up|elite|old|cute|handsome|ugly|naive|experienced|expert|joyful|grumpy|terrifying|radiated|frozen|fiery|electric|poisonous".split(
    "|"
)

SUFFIX_CARDS = "in rage|ready|tired|observing|unalert|eating|hungry|on steroids|enchanted|distracted|hallucinating|sceptical".split(
    "|"
)

UNIQUE_CARDS = "Deckard Cain|Diablo, Lord of Terror|Baal, Lord of Destruction|Frodo Beutlin|Sauron, the Lord of the Rings|Gandalf, the White|Harry Potter|Albus Dumbledore|Lord Voldemort".split(
    "|"
)

JOKER_CARDS = 3 * ["MONSTER"] + 3 * ["PREFIX"] + 3 * ["SUFFIX"]

JOKER_MAX_CHARACTERS = {"MONSTER": 25, "PREFIX": 15, "SUFFIX": 15}

GAME_PROMPT = """
Two mages summon monsters to fight each other.

The blue mage summons a skeleton warrior.
The skeleton warrior is a medium attacker and has no armor.
The skeleton warrior's special ability is to let his eyes glow.
The skeleton warrior says: "Grrrrrr"

--- The red mage summons a goblin.
The goblin is a weak attacker and has only light armor.
The goblin's special ability is to distract enemies with his smell.
The goblin says: "Me gonna kill you!"

The monsters fight.
The goblin attacks the skeleton warrior.
The goblin misses the skeleton warrior.
The skeleton warrior uses his special ability.
The eyes of the skeleton warrior glow, putting the goblin in panic.
The skeleton warrior attacks the goblin.
The skeleton warrior hits the goblin.
The goblin is dead.

The red mage who summoned the goblin loses.
The blue mage who summoned the skeleton wins.---

Two mages summon monsters to fight each other.

The red mage summons a dwarven babysitter.
The dwarven babysitter is a strong attacker and has medium armor.
The dwarven babysitter's special ability is to calm down the enemy.
The dwarven babysitter says: "Everything's going to be okay."

--- The blue mage summons a vampire high lord.
The vampire high lord is a strong attacker and has no armor.
The vampire high lord's special ability is to turn into mist.
The vampire high lord says: "I'm going to suck your blood."

The monsters fight.
The dwarven babysitter uses his special ability.
The vampire high lord is calmed down.
The dwarven babysitter attacks the vampire high lord.
The dwarven babysitter hits the vampire high lord.
The vampire high lord vaporizes into mist.
The dwarven babysitter attacks the mist.
The mist reforms into the vampire high lord.
The vampire high lord attacks the dwarven babysitter.
The vampire high lord hits the dwarven babysitter.
The dwarven babysitter is dead.

The red mage who summoned the dwarven babysitter loses.
The blue mage who summoned the vampire high lord wins.---

Two mages summon monsters to fight each other.

The blue mage summons Legolas.
Legolas is an excellent ranged attacker and has light armor.
Legolas' special ability is do elven acrobatics.
Legolas says: "I'm a good shot!"

--- The red mage summons a war elefant.
The war elefant is a strong attacker and has strong armor.
The war elefant's special ability is crush enemies.
The war elefant says: "Roar!"

The monsters fight.
Legolas shoots the war elefant.
Legolas hits the war elefant.
The war elefant uses his special ability.
The war elefant crushes Legolas with his right feet.
Legolas is dead.

The blue mage who summoned Legolas loses.
The red mage who summoned the war elefant wins.---
"""

STOP = "---"


class OpenAIModel:
    """
    The openai GPT-3 models.
    The free tier allows for $18 of requests.
    """

    def __init__(self, engine="davinci", temperature=0.7):
        self.engine = engine
        self.temperature = temperature

    def completion(self, prompt):
        """Return the completion from the openai model."""
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=STOP,
            n=1,
        )
        return response["choices"][0]["text"]


class AI21Model:
    """
    The AI21 jurassic models.
    In the beta you can get 10k tokens or 30k tokens per day for
    free for the large and the small model resp.
    """

    def __init__(self, model="j1-jumbo", temperature=0.7):
        self.model = model
        self.temperature = temperature
        self.api_key = os.environ.get("AI21_API_KEY")

    def completion(self, prompt):
        """Return the completion from the AI21 model."""
        payload = {
            "prompt": prompt,
            "numResults": 1,
            "maxTokens": 256,
            "stopSequences": [STOP],
            "topKReturn": 0,
            "temperature": self.temperature,
        }
        response = requests.post(
            f"https://api.ai21.com/studio/v1/{self.model}/complete",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
        ).json()
        return response["completions"][0]["data"]["text"]


class GptJModel:
    """
    A publicly available GPT-J model deployed on googles TPUs.
    20 requests (10 rounds of battles) per 30 min are allowed.
    """

    def __init__(self, temperature=0.7):
        self.temperature = temperature

    def completion(self, prompt):
        """Return the completion from the gpt-j model."""
        payload = {
            "context": prompt,
            "token_max_length": 256,
            "temperature": self.temperature,
            "top_p": 1,
            "stop_sequence": STOP,
        }
        response = requests.post(
            "http://api.vicgalle.net:5000/generate", params=payload
        ).json()
        return self.repair_completion(response["text"])

    @staticmethod
    def repair_completion(text):
        """Sometimes the model generates more than desired."""
        text_splitter = "<|endoftext|>"
        if text_splitter in text:
            return text.split(text_splitter)[0]
        return text


class MockModel:
    def completion(self, prompt):
        return GAME_PROMPT


class MageDuel:
    def __init__(self, model, best_of=5, hand_size=5):
        self.model = model
        self.deck = self.create_deck()
        random.shuffle(self.deck)
        if best_of % 2 == 0:
            raise ValueError("best_of must be an uneven number.")
        self.best_of = best_of
        if hand_size < 2 or hand_size > len(self.deck) / 4:
            raise ValueError(
                "hand_size should be greater 1 and smaller that 1/4 of the deck."
            )
        self.hand_size = hand_size
        self.blue_points = 0
        self.red_points = 0
        self.blue_hand = []
        self.red_hand = []
        self.graveyard = []
        self.draw_hands()
        self.blue_starts = random.choice([True, False])
        self.battle_report = ""

    @staticmethod
    def create_deck():
        """Create a new deck of cards."""
        return list(
            MONSTER_CARDS
            + PREFIX_CARDS
            + SUFFIX_CARDS
            + UNIQUE_CARDS
            + JOKER_CARDS
        )

    def reset_battle_report(self):
        """Resets the model buffer with the prompt and the battle report."""
        self.battle_report = ""

    def draw(self):
        """Draw the top card from the deck."""
        if not self.deck:
            self.deck = self.graveyard
            self.graveyard = []
            random.shuffle(self.deck)
        card = self.deck.pop(0)
        return card

    def draw_hands(self):
        """Draw the hands for both players."""
        for _ in range(self.hand_size):
            self.blue_hand.append(self.draw())
            self.red_hand.append(self.draw())

    def use_joker(self, joker_card):
        """Returns the user-defined name of a joker card."""
        max_len = JOKER_MAX_CHARACTERS[joker_card]
        while True:
            value = input(
                f"Write max {max_len} characters for the {joker_card} joker:"
            )
            if (
                STOP not in value
                and len(value) <= max_len
                and all(char not in value for char in "()")
            ):
                return value
            else:
                print(
                    "Invalid card name: No special charactes, keep the max length in mind."
                )

    def choice(self, question, choices, optional=False):
        """Ask the user to make a choice and return the result."""
        print(question)
        if not choices:
            print("You don't have any matching cards.")
            return None
        if optional:
            choices = ["None"] + choices
        choices = [self.render_card(c) for c in choices]
        while True:
            print(" - ".join([f"{i}:[{c}]" for i, c in enumerate(choices, 1)]))
            try:
                index = int(input("> "))
                assert 0 < index < len(choices) + 1
                choice = choices[index - 1]
                if choice == "None":
                    return None
                return click.unstyle(choice)
            except KeyboardInterrupt:
                sys.exit(1)
            except:
                print("Try again.")

    @staticmethod
    def render_card(card):
        if (
            card in MONSTER_CARDS
            or card == DEFAULT_MONSTER_CARD
            or card == "MONSTER"
        ):
            return click.style(card, fg="bright_yellow")
        elif card in UNIQUE_CARDS:
            return click.style(card, fg="bright_red")
        elif card in PREFIX_CARDS or card == "PREFIX":
            return click.style(card, fg="green")
        elif card in SUFFIX_CARDS or card == "SUFFIX":
            return click.style(card, fg="bright_cyan")
        return card

    def show_hand(self, hand):
        """Shows the hand of cards."""
        print("You have the following cards on your hand:")
        styled_hand = [self.render_card(card) for card in hand]
        print(" ".join([f"[{c}]" for c in styled_hand]))

    def select(self, hand):
        """Returns a creature made up cards from the player hand."""
        self.show_hand(hand)

        # select monster or unique
        available_monsters = [
            card
            for card in hand
            if card in MONSTER_CARDS
            or card in UNIQUE_CARDS
            or card == "MONSTER"
        ] or [DEFAULT_MONSTER_CARD]
        monster = self.choice("\nPick a monster:", available_monsters)
        if monster != DEFAULT_MONSTER_CARD:
            hand.remove(monster)
            self.graveyard.append(monster)
        if monster == "MONSTER":
            monster = self.use_joker(monster)
        creature = str(monster)

        if monster not in UNIQUE_CARDS:
            # select prefix, if available
            available_prefixes = [
                card
                for card in hand
                if card in PREFIX_CARDS or card == "PREFIX"
            ]
            prefix = self.choice(
                "\nPick a prefix:", available_prefixes, optional=True
            )
            if prefix:
                hand.remove(prefix)
                self.graveyard.append(prefix)
                if prefix == "PREFIX":
                    prefix = self.use_joker(prefix)
                creature = f"{prefix} {creature}"

            creature = f"a {creature}"

            # select suffix, if available
            available_suffixes = [
                card
                for card in hand
                if card in SUFFIX_CARDS or card == "SUFFIX"
            ]
            suffix = self.choice(
                "\nPick a suffix:", available_suffixes, optional=True
            )
            if suffix:
                hand.remove(suffix)
                self.graveyard.append(suffix)
                if suffix == "SUFFIX":
                    suffix = self.use_joker(suffix)
                creature += f" ({suffix})"

        return creature

    def summon(self, monster, player):
        """The player summons a monster."""
        self._send(f"The {player} mage summons {monster}.\n")

    def _send(self, text):
        """Send the text to the model for completion."""
        print(text)
        self.battle_report += text
        completion = self.model.completion(
            str(GAME_PROMPT) + self.battle_report
        )
        print(completion)
        self.battle_report += completion
        return completion

    def _write(self, text):
        """Write the text and have it sent upon next `_send()`."""
        print(text)
        self.battle_report += text

    def determine_winner(self):
        """
        Check in the battle report who has won last.
        True = Blue wins, False = Red wins, None = Could not be detected.
        """
        if re.search(
            r"(blue mage[^\n]*wins\.)|(red mage[^\n]*loses\.)",
            self.battle_report,
        ):
            return True
        elif re.search(
            r"(red mage[^\n]*wins\.)|(blue mage[^\n]*loses\.)",
            self.battle_report,
        ):
            return False
        else:
            return None

    def show_battle_report(self, delay=0.75):
        """Show the report."""
        lines = self.battle_report.splitlines()
        recap = True
        for line in lines:
            if line == "The monsters fight.":
                click.pause()
                recap = False
                click.secho(line, fg="yellow", bold=True)
            elif not recap:
                time.sleep(delay)
                click.secho(line, fg="yellow")
            else:
                click.echo(line)

    def game_round(self):
        """One round of the game."""
        click.clear()
        self.reset_battle_report()
        # determine which player starts
        if self.blue_starts:
            first_hand = self.blue_hand
            second_hand = self.red_hand
            first_player = "blue"
            second_player = "red"
        else:
            first_hand = self.red_hand
            second_hand = self.blue_hand
            first_player = "red"
            second_player = "blue"
        # both players draw one card
        first_hand.append(self.draw())
        second_hand.append(self.draw())
        # first player selects and summons
        print("### NEW ROUND ###\n")
        click.pause(f"The {first_player} mage opens...")
        click.clear()
        self._write("Two mages summon monsters to fight each other.\n\n")
        first_monster = self.select(first_hand)
        self.summon(first_monster, first_player)
        click.pause()
        # second player selects and summons
        click.clear()
        click.pause(f"The {second_player} mage responds...")
        click.clear()
        print(self.battle_report)
        second_monster = self.select(second_hand)
        self.summon(second_monster, second_player)
        # let the monsters fight
        click.clear()
        click.pause("The battle...")
        click.clear()
        self.show_battle_report()
        blue_wins_round = self.determine_winner()
        # score and set next player
        click.echo()
        if blue_wins_round is None:
            click.secho(
                "No winner could be determined.", fg="yellow", bold=True
            )
        elif blue_wins_round:
            click.secho(
                "The blue player wins this round.", fg="blue", bold=True
            )
            self.blue_points += 1
        else:
            click.secho("The red player wins this round.", fg="red", bold=True)
            self.red_points += 1
        print(f"Blue has {self.blue_points} points and red {self.red_points}.")
        self.blue_starts = blue_wins_round

        click.pause()

    def game_loop(self):
        """Main game loop"""
        while self.blue_points + self.red_points < self.best_of:
            self.game_round()
        if self.blue_points > self.red_points:
            click.secho("The blue player wins!", fg="blue", bold=True)
        else:
            click.secho("The red player wins!", fg="red", bold=True)


def title_screen():
    click.clear()
    for line in TITLE.splitlines():
        left, right = line[:41], line[41:]
        click.echo(
            click.style(left, fg="blue", bold=True)
            + click.style(right, fg="red", bold=True)
        )
    click.secho(SUBTITLE, fg="bright_yellow", bold=True)
    click.pause()
    click.clear()
    click.echo(INSTRUCTION)
    click.pause()
    click.clear()


if __name__ == "__main__":
    # model = OpenAIModel(engine="curie", temperature=0.7)
    # model = GptJModel(temperature=0.7)
    # model = MockModel()
    model = AI21Model(model="j1-jumbo", temperature=0.7)
    duel = MageDuel(model)
    title_screen()
    duel.game_loop()
