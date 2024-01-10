import openai
import urllib.request
from datetime import datetime
from openai_apikey import OPENAI_API_KEY

def dalle2(prompt):
    openai.api_key = OPENAI_API_KEY
    response = openai.Image.create(

        prompt=prompt,
        #"black & white heroic style image of :barbarian glancing at the skull of fallen foe, with a litte smug on his victoriues face",
        #"black & white heroic style image of :white siamese cat holding a sword that is actually a possessed by demon and wants to devour some souls, now it looks for some snack",
        n=1,
        size="1024x1024"

    )
    # print(image_url=response['data'][0]['url'])
    image_url=response['data'][0]['url']
    print(image_url)
    #urllib.request.urlretrieve(image_url, "/home/luksi02/DALL_E/2022_12_13/3.png")
    now = datetime.now()
    print(now)
    date_string = now.strftime("%Y_%m_%d_%H_%M_%S")

    print(date_string)
    dalle_output_dir = "/home/luksi02/DALL_E/spirit"+date_string+".png"
    print(dalle_output_dir)
    urllib.request.urlretrieve(image_url, dalle_output_dir)
    return print("alles_gut!")

def dalle2_in_loop(n):
    for i in range(1, n+1):
        print(i)
        #prompt = "fantasy heroic comic-style image of :barbarian glancing at the skull of fallen foe, with a litte smug on his victoriues face"
        dalle2(prompt)
        print(i, "complete")

barbarian_prompt = "fantasy heroic comic-style image of :barbarian glancing at the skull of fallen foe at his hands, with a litte smug on his victorious face"

AGONy_monster_goblin = "fantasy heroic comic-style image of Goblin :Little, nasty green creature, filled with hate and hunger - looks like it wants to be its next meal!"
AGONy_monster_orc = """horror-fantasy heroic comic-style image of Orc :Big, angry green creature that finds you very attractive... as a food, 
                               and you guessed it - it means you should be afraid!"""

AGONy_monster_bandit="""horror-fantasy heroic comic-style image of Bandit :ever heard saying: dont talk to strangers? Well, one od them just approached you, and seems like he 
                               wants to befriend you - why elese would he shout "Your money or your life!"?"""


AGONy_monster_wolf="""horror-fantasy heroic comic-style image of Wolf :Ever heard tales and stories why you 
                                should not walk into the woods? You guessed it - here comes 
                                the wolf and it counts you'll be a fancy snack!"""

AGONy_monster_spider="""horror-fantasy heroic comic-style image of Spider: Itsy bitsy giant venomous spider - Oh, I'm so cute:
                                I have eight long furry legs, eight terryfing eyes set on you, and you 
                                guessed it! I want some cuddels and cover you in webs and then eat! Come to papa!"""

AGONy_monster_angry_bird_bear = """horror-fantasy heroic comic-style image of Angry Bird-Bear: Have you ever heard of angry bird? Probably. 
                               Heard of angry bear? Probably. Heard of Bird-Bear? Never? Well, some kind of 
                               psycho-druid created this abonomination, and now it's up to you to face IT. 
                               And get rid of IT. For everyone!"""

AGONy_monster_dragon = """horror-fantasy heroic comic-style image of Angry Dragon: Mystic and poud creature, but (there's always a but!) has a 
                               nasty habit - it hoards anything gold-like and shiny! (it wants a new addition 
                               to it's collection, and you guessed it - it wants you and your shines!"""

AGONy_monster_zombie = """horror-fantasy heroic comic-style image of Angry Zombie"""

"""Clumsy, stinking, brainless... 
                                those damn zombies! Ugh, one 
                               just dropped its liver. DISGUSTANG! Well, brainless for now - it wants your 
                               brainzzz! Now protect it or 
                               become another brainless, wiggly, rotting walking zombie!"""


AGONy_monster_skeleton = """horror-fantasy heroic comic-style image of Angry Skeleton: There's something there! Something white and full of calcium. 
                               Hey, why those bones hover in air? 
                               Hey, why those skull turned into my direction? Oh hell no, why it moves 
                               towards me? Shouldn't it behave nicely and just stay dead?"""

AGONy_monster_spirit = """horror-fantasy heroic comic-style image of Angry-Vegenful Spirit: some spirits stay on earth even after death - mostly 
                               because their life was ended tragically. 
                               Now you encountered one. Not a pleasant spirit this one is, oh no."""

#by murder or other foul action.

AGONy_monster_ = """horror-fantasy heroic comic-style image of Angry """

prompt = AGONy_monster_spirit


#dalle2(prompt)
dalle2_in_loop(10)


"""

watch your step! While watching beautiful bird you fell into a cave, a dark, dark cave. 
        Youre lucky you didnt break your legs. Anyway, escaping cave took a lot od time.
[' One day, a lot of time', ' It was a lot of time', '\n 
             Finally, you found a way out! You found a beautiful bird on the way and it was singing',
              ' You followed the bird, but sadly, it was just an ordinary bird', '\n    
        The End!\n\nMy dearest diary! It is 12th day of my quest to earn fame and glory! New day comes', ' New challenges', " Hope I, the Percy McPerson, am ready for what comes next and I'm ready for these adventures! Maybe one day I will be remembered as a legend? Let's find out!\n              Meanwhile, today's adventure: watch your step! While watching beautiful bird you fell into a cave, a dark, dark cave", " You're lucky you"]

While  wandering through plains you felt watched - 
        but it's too late to do anything else but fight! Draw your weapon!
['    \n\nMy dearest diary! It is 13th day of my quest to earn fame and glory! New day comes'
, ' New challenges', " Hope I, the Percy McPerson, am ready for what comes next and I'm ready for 
these adventures! Maybe one day I will be remembered as a legend? Let's find out!\n Meanwhile, 
today's adventure:\n     While  wandering through plains you felt watched \n                                                                                                            "]

Ooh, shiney! You found something! Seems like after all it was worth to walk 
                            and put yourself in all this danger. Now let's see what you found!
["\n\nToday's adventure:\n\nOoh, shiney! You found something! Seems like after all it was worth to
 walk so much and put yourself in all this danger", " Now let's see what you found!\n\nIt's a \n\n
 Piece of a cooking pan!\n\nGreat! Now I can make more rice!\n\nHuh? You want to know what I'm 
 talking about? I'm talking about rice!\n\nYes, rice", ' A popular food', " \n\nIt's a\n\nPiece of a 
 cooking pan!\n\nGreat! Now I can make more rice!\n\nHuh? You want to know what I'm talking about? I'm 
 talking about rice!\n\nYes, rice", ' A popular food', "\n\nI don't know what this means", '']

[
wandering through Forest you notice more and more dead trees. Then, you 
        notice why - you stumble upon an old and grim crypt - do you dare to enter IT?
['\n\nMy dear diary-\n\nToday was a good day', ' I was out on my quest and I managed to 
kill two monsters', ' One tried to kill me with a spear, so I responded in kind by stabbing
it on the head', ' The other one tried to attack me with a club, but I cut it in half with 
my sword', " It was great! But then I stumbled upon a crypt, and I'm not sure what to do", ' 
I was about to enter, but I decided to stop', " I'm not sure if I want to enter", " 
What if the monsters inside are stronger than the ones I killed today? 
What if the monsters are hiding a treasure? I'm not sure", ' The only thing I know is 
that this crypt is an obstacle on my way to fame', '\n\nI have to keep going', 
'\n\nPercy McPerson\n\nI found a dead tree!']

"""