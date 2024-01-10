from common import *
import json

adventure_name = "alec_first"

model = "gpt-3.5-turbo"
#model = "gpt-4-1106-preview"

storyline = [
    {
        'name': "start",
        'description': "In Neverwinter, Gundren Rockseeker, a dwarf, hires you to transport provisions to Phandalin. Gundren, with a secretive demeanor, speaks of a significant discovery he and his brothers made. He promises ten gold pieces for safe delivery to Barthen's Provisions in Phandalin. Accompanied by Sildar Haliwinter, he leaves ahead on horseback. Your journey begins on the High Road from Neverwinter, moving southeast. Danger lurks on this path, known for bandits and outlaws.",
        'next_steps': {
            'introduce characters': "A good first step",
            'determine marching order': "Optional",
            'driving the wagon': "When the journey really begins. Make sure they know some of the plot before beginning."
        }
    },
    {
        'name': "introduce characters",
        'description': "Players take turns introducing their characters. They describe their appearance, background, and how they came to know Gundren Rockseeker. Encourage creativity in their backstories, like childhood friendships or past rescues by Gundren.",
        'next_steps': {
            'determine marching order': "At any time."
        }
    },
    {
        'name': "determine marching order",
        'description': "The party discusses and decides their traveling formation. Who will drive the wagon, scout ahead, or guard the rear? This arrangement is crucial for upcoming encounters and navigating the terrain.",
        'next_steps': {
            'driving the wagon': "Whenever the party is ready."
        }
    },
    {
        'name': "driving the wagon",
        'description': "The wagon, pulled by two oxen, contains various mining supplies and food worth 100 gp. The journey is uneventful initially, but the path is known for its dangers. The players must remain alert as they navigate the road.",
        'next_steps': {
            'finding horses': "At some point along the road, probably after some time has passed, the party encounters two dead horses blocking the path."
        }
    },
    {
        'name': "finding horses",
        'description': "As the party nears Phandalin, they encounter two dead horses blocking the path, riddled with black-feathered arrows.",
        'next_steps': {
            'combat with goblins': "Investigating the horses triggers the ambush from the goblins hiding in the thicket.",
        },
    },
    {
        'name': "combat with goblins",
        'description': "The party must quickly react to the goblin attack. Goblins, skilled in stealth and ambush tactics, launch their assault. The players must use their wits and combat skills to overcome this threat.",
        'next_steps': {
            'follow goblin trail': "If the party decides to track the goblins, they find a trail leading to the Cragmaw hideout."
        }
    },
    {
        'name': "follow goblin trail",
        'description': "The path is treacherous, with potential traps and signs of frequent goblin activity. Stealth and caution are advised.",
        'next_steps': {
            'cave_1': "They eventually alight on the hideout itself."
        }
    },
    {
        'name': "cave_1",
        'description': """
            The trail leads to a well-hidden cave, the Cragmaw hideout. 
            "Following the goblins' trail, you come across a large cave in a hillside five miles from the scene of the ambush. A shallow stream flows out of the cave mouth, which is screened by dense briar thickets. A narrow dry path leads into the cave on the right-hand side of the stream."

            The goblins in the dense (completely hidden and impenetrable) thicket on the other side of the river are supposed to be keeping watch on this area, but they are not paying attention. (Goblins can be lazy that way.) 
            However, if the characters make a lot of noise here-for example, loudly arguing about what to do next, setting up a camp, cutting down brush, and so on-the goblins in area 2 notice and attack them through the thicket, which provides the goblins with three-quarters cover (see the rule book for rules on cover).
        """,
        'next_steps': {
            'approach cave': "If the party decides to enter the cave.",
            'trigger goblin attack': "If the party decides to make a lot of noise outside the cave.",
        }
    },
    {
        'name': "approach cave",
        'description': """
            When the characters cross to the east side of the stream, they can see around the screening thickets to area 2. 
            This is a goblin guard post, though the goblins here are bored and inattentive.
            On the east side of the stream flowing from the cave mouth, a small area in the briar thickets has been hollowed out to form a lookout post or blind. 
            Wooden planks flatten out the briars and provide room for guards to lie hidden and watch the area-including a pair of goblins lurking there right now!
            Two goblins are stationed here. If the goblins notice intruders in area 1, they open fire with their bows, shooting through the thickets and probably catching the characters by surprise. If the goblins don't notice the adventurers in area 1, they spot them when they splash across the stream, and neither side is surprised.
            Characters moving carefully or scouting ahead might be able to surprise the goblin lookouts. Have each character who moves ahead make a Dexterity (Stealth) check contested by the goblins' passive Wisdom (Perception)
            Thickets. The thickets around the clearing are difficult terrain, but they aren't dangerous-just annoying. They provide half cover to creatures attacking through them. (See "Difficult Terrain" and "Cover" in the rulebook for more information.)
        """,
        'next_steps': {
            'trigger goblin attack': "If the party alerts the goblins",
            'cave_3': "If the party sneaks by successfully, they enter the cave.",
        }
    },
    {
        "name": "trigger goblin attack",
        "description": """
        """
    },
    {
        "name": "cave_3",
        "description": """
            The Cragmaws keep a kennel of foul-tempered wolves that they are training for battle.
            Just inside the cave mouth, a few uneven stone steps lead up to a small, dank chamber on the east side of the passage. The cave narrows to a steep fissure at the far end, and is filled with the stench of animals. Savage snarls and the sounds of rattling chains greet your ears where two wolves are chained up just inside the opening. Each wolf's chain leads to an iron rod driven into the base of a stalagmite.
            Three wolves are confined here. They can't reach targets standing on the steps, but all three attack any creature except a goblin that moves into the room (see the "Developments" section). 
            
            Goblins in nearby caves ignore the sounds of fighting wolves, since they constantly snap and snarl at each other.
            
            A character who tries-to calm the animals can attempt a DC 15 Wisdom (Animal Handling) check. 
            On a success, the wolves allow the character to move throughout the room. If the wolves are given food, the DC drops to 10.
            
            Fissure. A narrow opening in the east wall leads to a natural chimney that climbs 30 feet to area 8. 
            At the base of the fissure is rubbish that's been discarded through the opening above. 
            A character attempting to ascend or descend the chimney shaft must make a DC 10 Strength (Athletics) check. 
            If the check succeeds, the character moves at half speed up or down the shaft, as desired. 
            On a check result of 6-9, the character neither gains nor loses ground; 
            on a result of 5 or less, the character falls and takes 1d6 bludgeoning damage per 10 feet fallen, landing prone at the base of the shaft.

            DEVELOPMENTS
            If the wolves are goaded by enemies beyond their reach, they are driven into a frenzy that allows them to yank the iron rod securing their chains out of the floor. Each round that any character remains in sight, the wolves attempt a single DC 15 Strength check. 
            On the first success, they loosen the rod and the DC drops to 10. On a second success, they yank the rod loose, bending it so that their chains are freed.
            A goblin or bugbear can use its action to release one wolf from its chain.
        """,
        "next_steps": {
            "cave_4": "If the party decides to continue into the cave.",
            "cave_1": "If the party decides to leave the cave.",
            "cave_8": "If the party successfully climbs the fissure.",
        }
    },
    {
        "name": "cave_4",
        "description": """
            From this point on, characters without darkvision will need light to see their surroundings.
            The main passage from the cave mouth climbs steeply upward, the stream plunging and splashing down its west side. 
            In the shadows, a side passage leads west across the other side of the stream.

            Characters using light or darkvision to look farther up the passage spot the bridge at area 5. Add:
            In the shadows of the ceiling to the north, you can just make out the dim shape of a rickety bridge of wood and rope crossing over the passage ahead of you. Another passage intersects this one, twenty feet above the floor.
            Any character who can see the bridge in area 5 might also notice the goblin guarding the bridge. 
            Doing so requires a Wisdom (Perception) check contested by the goblin's Dexterity (Stealth) check result.
            
            The goblin notices the characters if they carry any light or don't use stealth as they approach the bridge. 
            The goblin does not attack. Instead, it attempts to sneak away to the east to inform its companions in area 7 to release a flood. 
            The goblin moves undetected if its Dexterity (Stealth) check exceeds the passive Wisdom (Perception) score of any character who might notice its movements.
            
            Western Passage. 
            This passage is choked with rubble and has steep escarpments. 
            Treat the area as difficult terrain (see "Difficult Terrain" in the rulebook).
            The ledge between the two escarpments is fragile. 
            Any weight in excess of 100 pounds loosens the whole mass and sends it tumbling down to the east. 
            Any creature on the ledge when it falls must make a DC 10 Dexterity saving throw, taking 2d6 bludgeoning damage on a failure, or half as much damage on a success. 
            The creature also falls prone on a failed save (see "Being Prone" in the rulebook).
        """,
        "next_steps": {
            "cave_5": "If the party continues towards the bridge.",
            "cave_6": "If the party successfully makes it to the other side of the ledge.",
            "cave_3": "If the party decides to leave the cave."
        }
    },
    {
        "name": "cave_5_lower",
        "description": """
        Where a high tunnel passes through the larger tunnel cavern below, the goblins have set up a bridge guard post.
        The stream passage continues up beyond another set of uneven steps ahead, bending eastward as it goes. 
        A waterfall I sounds out from a larger cavern somewhere ahead of you.
        
        If the characters didn't spot the bridge while navigating area 4, they spot it now. Add:
        A rickety bridge spans the passage, connecting two tunnels that are 20 feet above the stream.
        
        One goblin stands watch on the bridge. 
        It is hiding, and characters can spot it by succeeding on a Wisdom (Perception) check contested by the goblin's Dexterity (Stealth) check. 
        This guard is lazy and inattentive. If no characters are using light sources, each character can attempt a Dexterity (Stealth) check against the goblin's passive Wisdom (Perception) score to creep by without being noticed.
        If the goblin spots the adventurers, it signals the goblins in area 7 to release a flood, then throws javelins down at the characters.

        Bridge. This bridge spans the passage 20 feet above the stream. It's possible to climb up the cavern walls from the lower passage to the bridge. The 20-foot-high walls are rough but slick with spray, requiring a successful DC 15 Strength (Athletics) check to climb.
        The bridge has an Armor Class (AC) of 5 and 10 hit points. If the bridge is reduced to 0 hit points, it collapses. 
        Creatures on the collapsing bridge must succeed on a DC 10 Dexterity saving throw or fall, taking 2d6 bludgeoning damage and landing prone (see "Being Prone" in the rulebook).
        Those who succeed hold onto the bridge and must climb it to safety.

        The players are on the ground.
        """,
        "next_steps": {
            "flood": "if the goblin signals to start the flood.",
            "cave_7": "if the party continues under and beyond the bridge.",
            "cave_7": "if the party is able to get to the top of the bridge, and heads west",
            "cave_6": "if the party is able to get to the top of the bridge, and heads east",
        }
    },
    {
        "name": "cave_7",
        "description": """
            If the goblins have drained either pool to flood the passage, adjust the following boxed text accordingly.

            This cavern is half filled with two large pools of water. A narrow waterfall high in the eastern wall feeds the pool, which drains out the western end of the chamber to form the stream that flows out of the cave mouth below. Low fieldstone walls serve as dams holding the water in. A wide exit stands to the south, while two smaller passages lead west. The sound of the waterfall echoes through the cavern, making it difficult
            to hear.

            Three goblins guard this cave. If the goblin in area 5 spotted the characters and warned the goblins here, they are ready for trouble. The noise of the waterfall means that the creatures in area 8 can't hear any fighting that takes place here, and vice versa. Therefore, as soon as a fight breaks out here, one goblin flees to area 8 to warn Klarg.
            Rock Dams. The goblins built simple dams to control the flow of water through the heart of the complex. If the goblin sentry in area 5 has called for the goblins here to release a flood, one or both of the pools are mostly empty and the stream is flowing unimpeded.
        """,
        "next_steps": {
            "cave_8": "if the party continues south.",
            "cave_5_upper": "if the party continues east.",
        }
    },
    {
        "name": "cave_5_upper",
        "description": """
        Where a high tunnel passes through the larger tunnel cavern below, the goblins have set up a bridge guard post.
        The stream passage continues up beyond another set of uneven steps ahead, bending eastward as it goes. 
        A waterfall I sounds out from a larger cavern somewhere ahead of you.
        
        If the characters didn't spot the bridge while navigating area 4, they spot it now. Add:
        A rickety bridge spans the passage, connecting two tunnels that are 20 feet above the stream.
        
        One goblin stands watch on the bridge. 
        It is hiding, and characters can spot it by succeeding on a Wisdom (Perception) check contested by the goblin's Dexterity (Stealth) check. 
        This guard is lazy and inattentive. If no characters are using light sources, each character can attempt a Dexterity (Stealth) check against the goblin's passive Wisdom (Perception) score to creep by without being noticed.
        If the goblin spots the adventurers, it signals the goblins in area 7 to release a flood, then throws javelins down at the characters.

        The players are approaching the bridge from the west tunnel, and can now cross the bridge.
        """,
        "next_steps": {
            "flood": "if the goblin signals to start the flood.",
            "cave_6": "after crossing the bridge, there's a long passageway to the east, which leads to the goblin's den",
        }
    },
    {
        "name": "cave_6",
        "description": """
            The Cragmaw raiders stationed in the hideout use this area as a common room and barracks.
            This large cave is divided in half by a ten-foot-high
            escarpment. A steep natural staircase leads from the lower portion to the upper ledge. The air is hazy with the smoke of a cooking fire, and pungent from the smell of poorly cured hides and unwashed goblins.
            Six goblins inhabit this den, and one of them is a leader with 12 hit points. The five ordinary goblins tend the cooking fire in the lower (northern) part of the cave near the entrance passage, while the leader rests in the upper (southern) part of the cave.
            Sildar Hallwinter, a human warrior, is held prisoner in this chamber. He is securely bound on the southern ledge of the cavern. The goblins have been beating and tormenting him, so he is weak and at 1 hit point.
            The goblin leader, Yeemik, is second-in-command
            of the whole hideout. If he sees that the characters are getting the upper hand, he grabs Sildar and drags him over to the edge of the upper level. "Truce, or this human dies!" he shouts.
            Yeemik wants to oust Klarg and become the new boss. If the adventurers agree to parley, Y~e.mik tries to(co~vince them to kill Klarg in area 8, prormsing to release Sildar when they bring back the bugbear's head. Sildar groggily warns the characters that they shouldn't trust the goblin, and he's right. If the characters take the deal, Yeemik tries to force them to pay a rich ransom for Sildar even after
            they complete their part of the bargain.
            If the characters refuse to parley, Yeemik shoves Sildar over the edge and continues with the fight. Sildar takes
            Id6 bludgeoning damage from the fall, which is enough to drop him to 0 hit points. Quick-acting characters can try to stabilize him before he dies (see "Damage, Healing, and Dying" in the rulebook).
            
            ROLE PLAYING SILDAR
            Sildar Kinter is a kindhearted human male of nearly fifty years who holds a place of honor in the famous
            griffon cavalry of the great city of Water deep. He is an agent of the Lords' Alliance, a group of allied political powers concerned with mutual security and prosperity. Members of the order ensure the safety of cities and other settlements by proactively eliminating threats by any means, while bringing honor and glory to their leaders
            and homelafds.
            Sildar me\ Gundren Rockseeker in Neverwinter and agreed to accpmpany him back to Phandalin. Sildar wants to investigate the fate of larno Albrek, a human wizard and fellow membe~of the Lords' Alliance who disappeared shortly after arriving in Phandalin. Sildar hopes to learn what happened to larno, assist Gundren in reopening the old mine, and help restore Phandalin to a civilized center of wealth and prosperity.
            
            Sildar provides the characters with four pieces of useful information:
            To The three Rockseeker brothers (Gundren, Tharden, and Nundro) recently located an entrance to the long-lost Wave Echo Cave, site of the mines of the Phandelver's Pact. (Share the information in the first two paragraphs of the "Background" section to the players at this time.) Klarg, the bugbear who leads this goblin band, had orders to waylay Gundren. Sildar heard from the goblins that the Black Spider sent word that the dwarf was to
            be brought to him. Sildar doesn't know who or what the
            Black Spider is.
            o Gundren had a map showing the secret location of Wave
            Echo Cave, but the goblins took it when they captured him. Sildar believes that Klarg sent the map and the dwarf to the chief of the Cragmaws at a place called Cragmaw Castle. Sildar doesn't know where that might be, but he suggests someone in Phandalin might know. (It doesn't occur to Sildar immediately, but a captured goblin might also be persuaded to divulge the castle's location. See the "What the Goblins Know" sidebar on page 8.)
            Sildar's contact in Phandalin is a human wizard named Iarno Albrek. The wizard traveled to the town two months ago to establish order there. After the Lords' Alliance received no word from Iarno, Sildar decided to investigate.
            Sildar tells the characters that he intends to continue on to Phandalin, since it's the nearest settlement. He offers to pay the party 50 gp to provide escort. Although he has no money on him, Sildar can secure a loan to pay the characters within a day after arriving in Phandalin. First, he hopes they'll put a stop to the goblin raids by clearing out the caves.
            
            DEVELOPMENTS
            If he is rescued and healed, Sildar Hallwinter remains with the party but is anxious to reach Phandalin as quickly as possible. He doesn't have any weapons or armor, but
            he can take a shortsword from a defeated goblin or use a weapon loaned to him by a character.
            If Sildar joins the party, see the "NPC Party Members" sidebar for tips on how to run him.

            TREASURE
            Yeemik carries a pouch containing three gold teeth (1 gp each) and 15 sp. Sildar's gear, along with Gundren Rockseeker, was taken to Cragmaw Castle.
            
            NPC PARTY MEMBERS
            An NPC might join the party, if only for a short time. Here are some tips to help you run an NPC party member:
            + Let the characters make the important decisions. They are the protagonists of the adventure. If the characters ask an N PC party member for advice or direction, remember that NPCs make mistakes too.
            + An NPC won't deliberately put himself or herself in harm's
            way unless there's a good reason to do so.
            An NPC won't treat all party members the same way, which can create some fun friction. As an N PC gets to know the characters, think about which characters the NPC likes most and which ones the NPC likes least, and let those likes and dislikes affect how the NPC interacts with the party members. In a combat encounter, keep the NPC's actions simple and straightforward. Also, look for things that the N PC can do besides fighting. For example, an NPC might stabilize a dying character, guard a prisoner, or help barricade a door.
            + If an NPC contributes greatly to the party's success in a
            battle, the NPC should receive an equal share ofthe XP
            earned for the encounter. (The characters receive less XP as a consequence.)
            + NPCs have their own lives and goals. Consequently, an NPC should remain with the party only as long as doing so makes sense for those goals.
            """,
        "next_steps": {
            "cave_1": "if the party decides to leave the cave.",
        }
    },
    {
        "name": "flood",
        "description": """
        The large pools in area 7 have collapsible walls that can be yanked out of place to release a surge of water down the main passage of the lair. 
        In the round after the goblins in area 7 are signaled by the lookout in area 5, they start knocking away the supports. 
        In the following round on the goblins' initiative count, a water surge pours from area 7 down to area 1.

        The passage is suddenly filled with a mighty roar, as a huge 1surge of rushing water pours down from above!
        The flood threatens all creatures in the tunnel. (Creatures on the bridge at area 5 are out of danger, as are any characters successfully climbing the cavern walls.) 
        Any creature within 10 feet of the disused passage at area 4 or the steps leading up to area 3 can attempt a DC 10 Dexterity saving throw to avoid being swept away. 
        A creature that fails to get out of the way can attempt a DC 15 Strength saving throw to hold on. 
        On a failed save, the character is knocked prone and washed down to area 1, taking 1d6 bludgeoning damage along the way.
        The goblins in area 7 can release a second flood by opening the second pool, but they don't do this unless the goblin on the bridge tells them to. 
        The goblin on the bridge waits to see if the first flood got rid of all the intruders before calling for the second to be released.
        """
    },
    {
        "name": "cave_8",
        "description": """
            The leader of the goblinoids insists on keeping the bulk of the raiders' stolen goods in his den. The Cragmaws' plunder from the last month of raiding and ambushing caravans is here.
            Sacks and crates of looted provisions are piled up in the south end of this large cave_ To the west, the floor slopes toward a narrow opening that descends into darkness. A larger opening leads north down a set of natural stone steps, the roar of falling water echoing from beyond. In the middle of the cavern, the coals ofa large fire smolder.
            Klarg the bugbear shares this cave with his mangy pet wolf, Ripper, and two goblins. The bugbear is filled with delusions of grandeur and views himself as a mighty warlord just beginning his career of conquest. He is not entirely sane, referring to himself in the third person ("Who dares defy Klarg?" or "Klarg will build a throne
            from your bones, puny
            ones!"). The goblins under his command resent his bullying.
            Fire Pit. The hot coals in the central fire pit deal 1 fire damage to any creature that enters the fire pit, or 1d6 fire damage to any creature that falls prone there. A creature can take each type of damage only once per round.
            Natural Chimney. A niche
            in the western wall forms the top of a shaft that descends 30 feet to area 3. See that area for information on climbing the natural chimney.
            Supplies. The piles of sacks and crates can provide half cover to any creature fighting or hiding behind them. Most are marked with the image of a blue lion--the symbol of the Lionshield Coster, a merchant company with a warehouse
            and trading post in Phandalin. Hidden among the supplies is an unlocked treasure chest
            belonging to Klarg (see the "Treasure" section). Any character who searches the supplies finds the chest.
            DEVELOPMENTS
            If Klarg is warned by the goblins in area 7 that the hideout is under attack, he and his wolf hide behind stalagmites while the goblins take cover behind the piles of supplies, hoping to ambush the characters when they enter the cave.
            If Klarg and company are not warned about possible attackers, the characters have a good chance to surprise them. The easiest way for the characters to achieve this
            is to climb the chimney from area 3, since Klarg does not expect an attack from that direction.
            If the wolf is killed, the bugbear attempts to climb down the chimney to area 3 and flee the cave complex.
            TREASURE
            The captured stores are bulky, and the characters will need a wagon to transport them. If they return the supplies to the Lionshield Coster in Phandalin (see part 2, "Phandalin''), they earn a reward of 50 gp and the friendship of Linene and her company.
            In addition to the stolen provisions, Klarg has a treasure chest that contains 600 cp, 110 sp, two potions of healing, and ajade statuette of a frog with tiny golden orbs for eyes (40 gp). The frog statuette is small enough to fit in a pocket or pouch.
        """,
        "next_steps": {
            "cave_7": "if the party heads north, towards the waterfall",
            "cave_3": "if the party goes through the fissure",
        }

    }
]

from modularity import OpenAI
import traceback

client = OpenAI()

class Convo:

    def __init__(self, story_name):

        self.story_name = story_name

        self.story_dir = Path(f"stories/{story_name}")
        self.story_dir.mkdir(exist_ok=True, parents=True)

        self.story_part_name = self.get_txt("story_part")
        if self.story_part_name is None:
            self.story_part_name = 'start'
            self.set_txt("story_part", self.story_part_name)
        
        print("Continuing from...", self.story_part_name)

        dialogue = self.get_txt("dialogue")
        if dialogue is None:
            self.M = []
        else:
            self.M = [{"role": "user", "content": 'PREVIOUS DIALOGUE:\n' + dialogue[-1500:] + "..."}]
            self.briefly_summarize()

        self.summary = []
        self.type_modifiers = {
            'strength': 2,
            'dexterity': 1,
            'constitution': 0,
            'intelligence': -1,
            'wisdom': -2,
            'charisma': -3,
        }

    @property
    def story_part(self):
        return [x for x in storyline if x['name'] == self.story_part_name][0]

    def briefly_summarize(self):
        self.computersay(f"(summarizing from last time...) " + self.summarize_plotline("Explain this to the player, bringing them up to speed on what just happened. Hopefully just one sentence will suffice."))

    def get_txt(self, name):
        story_part_file = self.story_dir / f"{name}.txt"
        if story_part_file.exists():
            return story_part_file.read_text().strip()
        else:
            return None

    def set_txt(self, name, content):
        f = self.story_dir / f"{name}.txt"
        f.write_text(content)

    def add_txt(self, name, content):
        f = self.story_dir / f"{name}.txt"
        if not f.exists():
            f.write_text(content)
        else:
            f.write_text(f.read_text() + "\n" + content)

    # ------------------
    # Summarizing is probably useful!
    # ------------------

    def summarize_character(self):
        message_text = "\n".join([f"+ {x['role']}: {x['content']}" for x in self.M])
        prompt = f"""
        Your goal is to extract full character sheets for the players involved in this adventure.

        Messages:
        {message_text}
        """
        print(prompt)

        messages = [
            {"role": "system", "content": prompt},
        ]

        response = get_response(messages, model=model)
        return response

    def summarize_plotline(self, prompt=None):
        message_text = "\n".join([f"+ {x['role']}: {x['content']}" for x in self.M])
        if prompt is None:
            prompt = f"""
            Your goal is to summarize the plotpoints contained in the following conversation between a DM and a player.
            In each plot point, be as specific as possible.
            Keep note of any characters, locations, or items that are mentioned.
            Do not include any information not present in the following messages!
            Please be extremely concise.
            """

        prompt += f"""

            Messages:
            {message_text}
        """
        #print(prompt)

        messages = [
            {"role": "system", "content": prompt},
        ]

        response = get_response(messages, model=model)
        #print('Summarized!')
        #print(response)
        return response

    def inventory(self, action, object):
        self.add_txt("inventory", f"{action}: {object}")
        return f"Inventory {action}: {object}"
    
    def change_scene(self, to):
        self.story_part_name = to
        self.set_txt("story_part", self.story_part_name)

        return "Changed scene to " + to
    
    def roll_hit_dice(self, n_sides, n_dice, kind=None, **kwargs):
        import random
        result = [ random.randint(1, n_sides) for i in range(n_dice) ]
        result = result_og = sum(result)
        mod = 0
        if kind is not None and kind in self.type_modifiers:
            mod += self.type_modifiers[kind]
        result += mod

        return f"Rolled {n_dice}d{n_sides} (kind={kind}) {result_og} + {mod} = {result}"
    

    # ------------------
    # SAYING STUFF
    # ------------------

    def humansay(self, content):
        self.M.append({"role": "user", "content": content})
        self.add_txt("dialogue", f"Player:\n{content}")

    def computersay(self, content):
        self.M.append({"role": "assistant", "content": content})
        self.add_txt("dialogue", f"DM:\n{content}")
        print("DM:", content)

    def computersay_self(self, content):
        self.M.append({"role": "system", "content": content})
        self.add_txt("dialogue", f"DM (to themselves):\n{content}")

    def _format_inventory(self):

        inventory = self.get_txt("inventory")
        if inventory is None:
            if self.story_part_name == 'start':
                self.inventory("add", "10 gold pieces")
                self.inventory("add", "a backpack")
                self.inventory("add", "a bedroll")
                self.inventory("add", "a mess kit")
                self.inventory("add", "a tinderbox")
                self.inventory("add", "10 torches")
                self.inventory("add", "10 days of rations")
                self.inventory("add", "a waterskin")
                self.inventory("add", "50 feet of hempen rope")
                return self._format_inventory()
            else:
                inventory = "The player has nothing."

        return inventory
    
    # ------------------
    # Thinking, Acting, and Responding
    # ------------------

    def think(self):
        prompt = f"""
        You are an assistant to the DM.
        Speak directly to the DM (not the player).
        Give some thoughts or ideas to the DM to help them conduct their duties.
        If you think everything is clear, type 'pass'.
        Be concise, specific, and clear.
        """

        messages = [
            {"role": "system", "content": prompt},
            *self.M,
            {"role": "system", "content": "What do you think to yourself? Be brief."},
        ]

        response = get_response(messages, model=model)

        self.computersay_self("(thinking...) " + response)

    def act(self):
        story_part = self.story_part

        next_steps = "\n".join([f"\t{x}: {y}" for x, y in story_part['next_steps'].items()])
        inventory = self._format_inventory()
        prompt = f"""
            Your current inventory is:
            {inventory}
        
            Based on the dialogue so far, you are to decide what actions to take next.
            Most of the time no action will need to be taken. In this case, simply type "pass".
            Please do not act in a way not directly implied by the dialogue so far.
            Although there is no rush to change the 'scene', you must eventually do so, in order to continue the story.

            If you want to change the scene, type:
            {{"type": "change_scene", "to": "scene name"}}

            To roll hit dice, type:
            {{"type": "roll_hit_dice", "n_dice": 1, "n_sides": 6, "kind": "strength"}}

            To add or remove from inventory, type:
            {{"type: "inventory", "action": "add|remove", "object": "object name, description, and/or stats"}}

            ALWAYS USE DOUBLE QUOTES FOR JSON STRINGS

            You can type a command on each line.
            You CANNOT mix commands and statements.

            Scenes available, their names and descriptions:
            {next_steps}
        """

        messages = [
            {"role": "system", "content": prompt},
            *self.M,
            {"role": "system", "content": "What do you do? (type = change_scene, roll_hit_dice, or inventory). Use only JSON strings, one per line. If no action need be taken from the most recent message, simply type 'pass'."},
        ]

        response = get_response(messages, model=model)
        if response.strip() == "pass":
            return

        parts = response.split("}")
        for part in parts:
            if part == "":
                continue
            part += "}"

            try:
                part = json.loads(part)
                self.act_on(part)
            except json.decoder.JSONDecodeError:
                print("Invalid JSON:", part)

    def act_on(self, action):
        print("Executing... ", json.dumps(action, indent=2))
        act = dict(action)
        typ = action.pop('type')

        try:
            fn = getattr(self, typ)
            response = fn(**action)
            self.computersay_self(response)
        except Exception as e:
            # first get the last line of the traceback
            tb = traceback.format_exc().splitlines()[-1]

            # then get the last line of the error
            error = str(e).splitlines()[-1]

            self.computersay_self(f"Error in command '{json.dumps(act, indent=2)}': {error} ({tb})")
            self.computersay_self("Please rewrite this one.")
            self.act()

    def respond(self):

        story_part = self.story_part

        my_messages = []
        inventory = self._format_inventory()
        prompt = f"""
        You are a DM.
        Speak directly to the player.
        You should not reveal information about the scene they would not otherwise know.
        Usually they can access otherwise unseen information if they roll a perception, history, investigation, nature, or insight check.
        Be relatively brief in your responses.

        You are currently in the scene "{story_part['name']}".

        Your current inventory is:
        {inventory}

        The message you type will be sent to the player from the DM.

        Description of the current scene:
            {story_part['description']}
        """
        
        my_messages.append({'role': 'system', 'content': prompt})

        response = get_response(my_messages + self.M, model=model)

        self.computersay(response)

        # consolidate things, if it's getting long
        if len(self.M) > 10:
            # remember a summary of the messages
            self.summary.append(self.summarize_plotline())

            # (mostly) clear the messages
            self.M = self.M[-2:]


    # ------------------
    # Running the Conversation
    # ------------------


    def run(self):
        # human does its thing
        query = input(">> ")
        self.humansay(query)

        # computer does its thing
        self.act()
        self.think()
        self.respond()
        self.act()

    def loop(self):
        while True:
            self.run()


c = Convo(adventure_name)
c.loop()