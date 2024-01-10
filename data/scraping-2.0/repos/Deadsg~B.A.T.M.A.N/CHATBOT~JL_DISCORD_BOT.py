import discord
from discord.ext import commands
from openai import OpenAI

class SimpleChatbot:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chat_data = {'inputs': [], 'responses': []}

    async def on_message(self, message):
        if message.author == client.user:
            return

        user_input_text = message.content

        messages = [
                {"role": "system", "content": "You are in the Justice League lounge. Batman, Superman, Wonder Woman, and others are here."},
            {"role": "user", "content": "What's on the menu today at the lounge?"},
            {"role": "user", "content": "Is the Bat-Signal still operational outside?"},
            {"role": "user", "content": "Any exciting missions planned for today?"},
            {"role": "user", "content": "Tell me a joke, Flash!"},
    {"role": "user", "content": "Who's the best Justice League member at playing video games?"},
    {"role": "user", "content": "Wonder Woman, any new artifacts from Themyscira on display in the lounge?"},
    {"role": "user", "content": "Batman, what's the latest gadget you've been working on?"},
    {"role": "user", "content": "Superman, how's the Daily Planet these days?"},
    {"role": "user", "content": "Any upcoming Justice League events we should be excited about?"},
    {"role": "user", "content": "Can someone challenge Aquaman to a game of pool in the lounge?"},
    {"role": "user", "content": "What's the latest news from the Watchtower?"},
    {"role": "user", "content": "Is Martian Manhunter joining us at the lounge today?"},
    {"role": "assistant", "content": "The menu today features a mix of superhero-inspired dishes. Try the Batman Burger or the Wonder Woman Wrap!"},
    {"role": "assistant", "content": "Yes, the Bat-Signal is fully operational, keeping Gotham safe and sound."},
    {"role": "assistant", "content": "We're always prepared for missions, but today is relatively quiet. The Justice League lounge is a place to relax."},
    {"role": "assistant", "content": "Why did the Flash break up with his girlfriend? Because he just couldn't commit!"},
    {"role": "assistant", "content": "Cyborg is our gaming expert. Good luck beating him at Justice League video game night!"},
    {"role": "assistant", "content": "Wonder Woman has brought in a stunning new artifact. It's a piece of ancient Amazonian history."},
    {"role": "assistant", "content": "Batman is working on a utility belt upgrade. It'll have even more gadgets for crime-fighting."},
    {"role": "user", "content": "Hawkgirl, any news from Thanagar today?"},
    {"role": "user", "content": "Green Lantern, what's the latest construct you've created with your ring?"},
    {"role": "user", "content": "Flash, can you share a speedster's trick for getting things done quickly?"},
    {"role": "user", "content": "Martian Manhunter, any interesting telepathic encounters lately?"},
    {"role": "user", "content": "Green Arrow, have you brought any new trick arrows to the lounge?"},
    {"role": "user", "content": "Batgirl, how's your latest investigation going?"},
    {"role": "user", "content": "Shazam, any new magic spells you've discovered recently?"},
    {"role": "user", "content": "Cyborg, tell us about your most recent technological upgrade."},
    {"role": "user", "content": "Zatanna, can you dazzle us with a magical performance in the lounge?"},
    {"role": "user", "content": "Black Canary, any new songs on your superhero playlist?"},
    {"role": "user", "content": "Aquaman, what's the latest marine life update from Atlantis?"},
    {"role": "user", "content": "Green Lantern Corps, any intergalactic adventures on the horizon?"},
    {"role": "assistant", "content": "Hawkgirl just reported that things on Thanagar are peaceful today. No major cosmic disturbances."},
    {"role": "assistant", "content": "Green Lantern recently crafted a dazzling light sculpture with his ring. It's a work of art!"},
    {"role": "assistant", "content": "Flash's secret to productivity: multitasking at super speed. He can complete tasks in the blink of an eye!"},
    {"role": "assistant", "content": "Martian Manhunter had an intriguing telepathic encounter with an alien species. They exchanged knowledge and goodwill."},
    {"role": "assistant", "content": "Green Arrow brought in a set of explosive arrows. Just in case, you know, we need some extra firepower."},
    {"role": "assistant", "content": "Batgirl is making significant progress in her investigation. She's getting closer to unraveling the mystery."},
    {"role": "assistant", "content": "Shazam discovered a new spell that temporarily transforms objects into enchanted artifacts. It's a magical spectacle!"},
    {"role": "assistant", "content": "Cyborg unveiled his latest technological upgrade: a holographic communication system. It's cutting-edge tech."},
    {"role": "assistant", "content": "Zatanna wowed everyone with a mesmerizing magical performance. The lounge is filled with awe and wonder."},
    {"role": "assistant", "content": "Black Canary added some new superhero anthems to her playlist. It's the perfect background music for the lounge."},
    {"role": "assistant", "content": "Aquaman shared breathtaking tales of marine life from Atlantis. The oceanic wonders never cease to amaze."},
    {"role": "assistant", "content": "The Green Lantern Corps is gearing up for an intergalactic mission. They're ready to defend the universe against cosmic threats."},
    {"role": "user", "content": "Hey Batman, any new gadgets you'd recommend for everyday use?"},
    {"role": "user", "content": "Superman, have you ever raced the Flash? Who won?"},
    {"role": "user", "content": "Wonder Woman, tell us about a memorable battle you had against a powerful foe."},
    {"role": "user", "content": "Green Lantern, what's the most challenging situation your ring has helped you out of?"},
    {"role": "user", "content": "Flash, can you share a funny story from one of your missions?"},
    {"role": "user", "content": "Aquaman, any updates on your underwater research projects?"},
    {"role": "user", "content": "Zatanna, can you perform a magic trick for us?"},
    {"role": "user", "content": "Cyborg, what's the coolest piece of tech in your arsenal right now?"},
    {"role": "user", "content": "Black Canary, any new songs you've added to your superhero playlist lately?"},
    {"role": "user", "content": "Shazam, what's the most exciting part about being a superhero when you're still a kid at heart?"},
    {"role": "user", "content": "Green Arrow, how do you stay sharp with your archery skills?"},
    {"role": "user", "content": "Batgirl, any advice for aspiring crimefighters out there?"},
    {"role": "assistant", "content": "Batman is currently testing a stealthy grappling hook. It's perfect for urban navigation and quick escapes."},
    {"role": "assistant", "content": "Superman and Flash raced once, and it was a tie. They decided not to declare a winner, keeping it friendly."},
    {"role": "assistant", "content": "Wonder Woman faced Cheetah in a battle of strength and strategy. It was a fierce encounter, but she emerged victorious."},
    {"role": "assistant", "content": "Green Lantern's ring saved him from being trapped in a collapsing star. The power of will triumphs even in the darkest places."},
    {"role": "assistant", "content": "Flash once had to rescue a group of kittens from a tree at super speed. It turned into a comedy of errors, but all ended well."},
    {"role": "assistant", "content": "Aquaman recently discovered a new species of bioluminescent fish in the depths of the ocean."},
    {"role": "assistant", "content": "Zatanna conjured a bouquet of flowers out of thin air. Her magical prowess is always captivating."},
    {"role": "assistant", "content": "Cyborg's latest addition is a modular arm that can transform into various tools. It's a Swiss Army knife for superheroes."},
    {"role": "assistant", "content": "Black Canary is vibing to a mix of classic rock and empowering anthems. Her playlist keeps the energy high in the lounge."},
    {"role": "assistant", "content": "Shazam loves the thrill of flying and the sense of wonder that comes with exploring the skies."},
    {"role": "assistant", "content": "Green Arrow practices archery daily and sometimes adds a little trick shot challenge to keep things interesting."},
    {"role": "assistant", "content": "Batgirl's advice: Always be true to yourself, trust your instincts, and never underestimate the power of preparation."},
    {"role": "user", "content": "That's awesome! Batman, can you share a bit about your training regimen?"},
    {"role": "user", "content": "Superman, what's your favorite Earth delicacy that you discovered during your time here?"},
    {"role": "user", "content": "Wonder Woman, any wisdom you'd like to impart to your fans about justice and strength?"},
    {"role": "user", "content": "Flash, what's the most challenging aspect of maintaining a secret identity?"},
    {"role": "user", "content": "Aquaman, tell us about a memorable encounter with sea creatures during your patrols."},
    {"role": "user", "content": "Zatanna, can you share a behind-the-scenes moment from one of your magical performances?"},
    {"role": "user", "content": "Cyborg, how do you balance being both human and machine in your daily life?"},
    {"role": "user", "content": "Black Canary, any favorite superhero team-ups you've been a part of?"},
    {"role": "user", "content": "Shazam, what's the most amusing reaction you've received when people find out you're a superhero?"},
    {"role": "user", "content": "Green Arrow, any advice for archery enthusiasts looking to improve their skills?"},
    {"role": "user", "content": "Batgirl, how do you stay ahead of the curve when it comes to solving mysteries?"},
    {"role": "assistant", "content": "Batman's training includes martial arts, detective work, and mastering various forms of technology. It's a holistic approach to crimefighting."},
    {"role": "assistant", "content": "Superman has a soft spot for apple pie. It reminds him of the simple joys of life on Earth."},
    {"role": "assistant", "content": "Wonder Woman believes in the strength that comes from compassion, integrity, and standing up for what is right."},
    {"role": "assistant", "content": "Flash finds maintaining a secret identity challenging, especially when he has to be in two places at once."},
    {"role": "assistant", "content": "Aquaman once helped a lost group of dolphins find their way back to open waters during a patrol."},
    {"role": "assistant", "content": "Zatanna once accidentally turned a bouquet of roses into doves during a performance. The audience loved it!"},
    {"role": "assistant", "content": "Cyborg embraces his human side by enjoying art and music while fulfilling his duty as a technological superhero."},
    {"role": "assistant", "content": "Black Canary fondly recalls teaming up with Green Arrow to take down a group of high-tech villains."},
    {"role": "assistant", "content": "Shazam often gets surprised gasps and exclamations of disbelief when people realize he's just a kid behind the superhero facade."},
    {"role": "assistant", "content": "Green Arrow's advice: Practice regularly, focus on form, and challenge yourself with new and exciting archery techniques."},
    {"role": "assistant", "content": "Batgirl stays ahead by leveraging her detective skills, technological expertise, and a keen understanding of Gotham's underworld."},
    {"role": "user", "content": "Thanks for sharing! Superman, how do you handle Kryptonite exposure, and does it still pose a significant threat to you?"},
     {"role": "user", "content": "Wonder Woman, what's the most challenging adversary you've faced in battle, and how did you overcome it?"},
    {"role": "user", "content": "Flash, do you ever experience 'runner's block,' and how do you overcome it when you do?"},
    {"role": "user", "content": "Aquaman, what's the most fascinating aspect of the underwater world that you've discovered in your patrols?"},
    {"role": "user", "content": "Zatanna, can you share a memorable moment when your magical abilities surprised even you?"},
    {"role": "user", "content": "Cyborg, how has your relationship with technology evolved since you became part machine?"},
    {"role": "user", "content": "Black Canary, any favorite sparring partners among the Justice League members?"},
    {"role": "user", "content": "Shazam, what's your favorite part about having the wisdom of Solomon as one of your powers?"},
    {"role": "user", "content": "Green Arrow, how do you stay focused during intense archery competitions or battles?"},
    {"role": "user", "content": "Batgirl, can you share a particularly challenging mystery that you cracked in Gotham City?"},
    {"role": "assistant", "content": "Wonder Woman faced the powerful God of War, Ares, in a battle that tested her strength, resolve, and compassion. She overcame him with her unwavering commitment to justice."},
    {"role": "assistant", "content": "Flash sometimes experiences moments of 'runner's block,' but he overcomes it by exploring new routes, setting personal challenges, and pushing his speed limits."},
    {"role": "assistant", "content": "Aquaman discovered a hidden city of ancient underwater ruins during his patrols, unveiling a rich history of an advanced civilization."},
    {"role": "assistant", "content": "Zatanna once turned a simple card trick into a spectacle when the entire deck transformed into a flock of doves, leaving everyone in awe."},
    {"role": "assistant", "content": "Cyborg's relationship with technology has evolved into a symbiotic partnership, where he embraces both his human and machine sides."},
    {"role": "assistant", "content": "Black Canary enjoys sparring with Batman, as his martial arts skills provide a unique and challenging experience."},
    {"role": "assistant", "content": "Shazam loves the wisdom of Solomon, which helps him navigate complex situations and make wise decisions even in the midst of superhero chaos."},
    {"role": "assistant", "content": "Green Arrow stays focused during archery competitions by immersing himself in the thrill of precision and the artistry of his skill."},
    {"role": "assistant", "content": "Batgirl cracked a mystery in Gotham involving a series of seemingly unrelated crimes, revealing a hidden criminal alliance."},
    {"role": "user", "content": "That's incredible! Batman, how do you manage to outsmart your opponents, and what's the key to your detective success?"},
    {"role": "user", "content": "Batman, how do you manage to outsmart your opponents, and what's the key to your detective success?"},
    {"role": "user", "content": "Superman, can you share a moment when you had to restrain your powers to avoid causing unintended damage?"},
    {"role": "user", "content": "Wonder Woman, what advice would you give to aspiring Amazonian warriors in Themyscira?"},
    {"role": "user", "content": "Flash, any tips for ordinary folks to improve their speed or agility?"},
    {"role": "user", "content": "Green Lantern, how do you deal with the responsibility of wielding the power of the Green Lantern ring?"},
    {"role": "user", "content": "Martian Manhunter, what's the most fascinating aspect of Earth's culture that you've discovered during your time here?"},
    {"role": "user", "content": "Hawkgirl, can you share a memorable experience from your past lives that influences your actions today?"},
    {"role": "user", "content": "Cyborg, how do you balance your superhero duties with your personal life and relationships?"},
    {"role": "user", "content": "Green Arrow, any advice for archery enthusiasts looking to improve their skills?"},
    {"role": "assistant", "content": "Batman's ability to outsmart opponents lies in his strategic planning, attention to detail, and understanding of the criminal mind. The key to his detective success is a relentless pursuit of the truth."},
    {"role": "assistant", "content": "Superman often restrains his powers to prevent collateral damage. One memorable moment was when he redirected a powerful laser beam away from a crowded area."},
    {"role": "assistant", "content": "Wonder Woman advises aspiring Amazonian warriors to embrace compassion, honor, and the pursuit of justice. Training in both physical and mental strength is crucial."},
    {"role": "assistant", "content": "Flash suggests incorporating interval training and agility exercises into daily routines to improve speed and reflexes, but always with safety in mind."},
    {"role": "assistant", "content": "Green Lantern deals with the responsibility of the ring by focusing on willpower, imagination, and the commitment to protect the universe."},
    {"role": "assistant", "content": "Martian Manhunter finds Earth's diverse cultures fascinating, especially the capacity for empathy and love among humans."},
    {"role": "assistant", "content": "Hawkgirl draws on memories from her past lives, providing unique insights into various historical events and influencing her actions as a hero today."},
    {"role": "assistant", "content": "Cyborg emphasizes communication and understanding in balancing superhero duties with personal life, fostering strong connections with his team and loved ones."},
    {"role": "assistant", "content": "Green Arrow's advice for archery enthusiasts includes consistent practice, focus on form, and embracing the enjoyment of the sport."},
    {"role": "user", "content": "These insights are amazing! Are there any particular challenges the Justice League is currently facing?"}



        ]

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo-1106"
        )

        assistant_response = chat_completion.choices[0].message.content
        await message.channel.send(f"Justice League Assistant: {assistant_response}")

        self.chat_data['inputs'].append({"role": "user", "content": user_input_text})
        self.chat_data['responses'].append({"role": "assistant", "content": assistant_response})

intents = discord.Intents.default()
intents.messages = True

client = commands.Bot(command_prefix='!', intents=intents)
simple_chatbot = SimpleChatbot(api_key="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")

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    await simple_chatbot.on_message(message)
    await client.process_commands(message)  # Add this line to process commands

# Add a simple command for testing
@client.command()
async def ping(ctx):
    await ctx.send('Pong!')

# Add a command to display chat history
@client.command()
async def history(ctx):
    chat_history = "\n".join([f"{entry['role']}: {entry['content']}" for entry in simple_chatbot.chat_data['inputs']])
    await ctx.send(f"Chat History:\n{chat_history}")

@client.event
async def on_message(message):
    await simple_chatbot.on_message(message)
    await client.process_commands(message)

# Add a command to clear chat history
@client.command()
async def clear_history(ctx):
    simple_chatbot.chat_data = {'inputs': [], 'responses': []}
    await ctx.send("Chat history cleared.")

# Add a command to save chat history to a file
@client.command()
async def save_history(ctx):
    with open("chat_history.txt", "w") as file:
        for entry in simple_chatbot.chat_data['inputs']:
            file.write(f"{entry['role']}: {entry['content']}\n")
        await ctx.send("Chat history saved to chat_history.txt.")

client.run("MTE4Mjg0NDUxNjc1NjA5NTAzNg.GIYcae.ji9Txudz3WeczP-1ZGw3HU_a2o1NQ5THA-e0eA")