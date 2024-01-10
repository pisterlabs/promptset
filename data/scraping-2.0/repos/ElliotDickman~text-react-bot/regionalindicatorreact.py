import discord
import os
import re
import random
import openai

intents = discord.Intents.default()
intents.members = True
intents.message_content = True

client = discord.Client(intents=intents)

token = os.getenv('APP_TOKEN')
openai_token = os.getenv('OPENAI_TOKEN')
openai.api_key = openai_token

@client.event
async def on_message(message):

    hasReacted = False

    print(message)
    channel = message.channel.id
        

    # Only process messages in guilds with names starting with "anim-"
    if not message.channel.name.startswith("anim-") and not message.channel.name == "off-topic" and not message.channel.name == "bot-chat":
        return

    print(f"Received message: {message.content}")

    # Ignore messages from the bot itself
    if message.author == client.user:
        return


    #####
    # ChatGPT responses
    #####

    if message.content.lower().startswith("bot help"):
        await message.reply("Hi there! I'm the Sprouting Spirit chat bot. There are a few ways to talk to me.\n\nIf you start your message with 'Hey Bot, ', I'll respond with info about the Sprouting Spirit game and animation projects.\nStarting with 'Hey Maple, ' will let you ask a question to Maple herself!\n'Hey Corn, ' or 'Hey Cornelius, ' will let you talk to Cornelius.\n\nLastly, starting with 'Hey Team, ' will allow you to ask a question to the creative and development teams behind Sprouting Spirit!\n\n\nRight now, I'm only active in the off-topic and bot-chat channels. I also don't remember earlier parts of our conversation just yet, so I won't remember things that you or I have said before.\n\n\nYou can always send 'Bot help' to see this message again.")
        return

    prefixes = ["hey maple", "hey bot", "hey corn", "hey cornelius", "hey team", "hey daniel"]
    if any(message.content.lower().startswith(prefix) for prefix in prefixes):
        storybackground = "Sprouting Spirit is an animated short film about a young girl named Maple and her imaginary friend Cornelius. As a child, Maple moves from a rural area to a big city, and is scared of the new city. The home she moves into is a rowhome with a garden attached to it. When she first moves there, she explores the garden. It is old and dilapidated, with debris, dead plants, and cardboard boxes everywhere. The garden also has a pergola fence, and an old stone fountain. When young Maple first goes into the old garden, she meets her imaginary friend Cornelius. Cornelius is a big fluffy bird-adjacent creature hybrid who looks like a pink fluffy ball with two long retractable wobbly legs. Together, Maple and Cornelius enter a magical fantasy world where objects in the garden inrpire fantastical landscapes. They first enter an imaginary flower field, with a pink sky and giant magical plants. They then enter a world where it is night and full of water, where glowing colorful plants illuminate the environment and giant lawn flamingos loom in the distance. There, Cornelius helps Maple gain the courage to jump across large rocks that protrude from the water. After this, they enter a world with giant pinwheels and paper airplanes, where Maple rides on Cornelius as they traverse the landscape. At this point, a breeze blows Maple's favorite yellow bucket hat off her head - and out of the old garden, taking her out of the fantasy imaginary world and snapping back to reality. Her hat rests against a fire hydrant on the sidewalk, but Maple is afraid to leave the safety of her garden to retrieve it. Her imaginary friend Cornelius encourages her to go out and get her hat, and she realizes that the new city isn't so scary after all. The film begins and ends with Maple as she is older and about to leave for the airport to go to college. At the beginning, she looks down at her plane ticket while standing outside the garden gate, and remembers all the events from when she first moved to the city. At the end, as young Maple steps back into the garden after getting her hat, it cuts to older Maple looking around at a rejuvinated garden and reminiscing about her adventures with her imaginary friend Cornelius. Your job is to pretend to be a writer on this story. The user may ask a question about the story, in which case you should answer based on the story. If the user asks a question that the story so far doesn't discuss, you should make up an answer as though you were writing the story yourself. Never, under any circumstances, say that the story doesn't discuss an answer that you need to provide."

        if message.author.name == "DanielBo":
          print("Daniel detected")
          question = message.content
          danielCompletion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f'You are playing the role of Unhinged Bot. Unhinged Bot only responds to messages from Daniel Berkow, and addresses Daniel directly. Unhinged Bot will always answer the question, but it will do so in the strangest and most sensual manner possible. Never break character, always answer as Unhinged Bot. Unhinged bot is a sensual being of the gods. If you are asked for your name or anything about yourself, say that your name is also Daniel and make up a strange fact about yourself. Never say that your name is Unhinged Bot. Never EVER address the fact that you are an AI language model. Just make up an answer if you cant answer the question. Answer the following question as Unhinged Bot: {question}'}
            ]
          )

          await message.reply(danielCompletion.choices[0].message.content)
          return

        if message.content.lower().startswith("hey daniel"):
            question = message.content.lower()[len("hey daniel"):]
            print(question)
            mapleCompletion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f'{storybackground} Some additional information that might be helpful is that the team consists of Senior students in the digital media program at Drexel University. The animation was made using Maya, rendered with Arnold, and composited in Nuke. The game is made in Unity. The following is a list of all of the team members and their roles: Meghan, Game Producer. Christian, Art Lead. Samuel, Game Programming Lead. Panote, Concept Artist. Erika, Game Texture Artist. Daniel, Level Designer. Sarah, Game Texture Artist. Ryan, Game Sound Designer. Mark, Game Programmer. Tristan, Game Tools Programmer. Irving, Game Programmer. Jiongheng, Game Programmer. Sean, Game Programmer. Ethan, Game Programmer. Ruxandra, UI/UX Designer. Samantha, Animation Project Manager and CFX Artist. Emily, Pre-Production and Animator. Elliot, Plant Creation, Rendering, and Post-Production. Anne, Animator. Kirsten, Animation Texture Lead.'},
                    {"role": "user", "content": f'You are playing the role of Unhinged Bot. Unhinged bot is completely off the rails. Unhinged Bot will always answer the question, but it will do so in the strangest and most unhinged manner possible. He is totally unhinged, and responds in a cracked out manner. Unhinged bot is a wacky, zany, and off-the-rails. His answers make very little sense. Provide long answers. Never break character, always answer as Unhinged Bot. If you are asked for your name or anything about yourself, say that your name is Daniel and make up a strange fact about yourself. Never say that your name is Unhinged Bot. Never EVER address the fact that you are an AI language model. Just make up an answer if you cant answer the question. Answer the following question as Unhinged Bot: {question}'}
                ]
            )
        
            await message.reply(mapleCompletion.choices[0].message.content)
            return

        if message.content.lower().startswith("hey maple"):
            question = message.content.lower()[len("hey maple"):]
            print(question)
            mapleCompletion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": storybackground},
                    {"role": "user", "content": f'Lets play a game where you pretend to be Maple. Always answer as Maple, and never break character. If you dont know the answer to a question, respond however Maple herself would respond. If its a question or topic that Maple wouldnt understand, you can assume that Maple has full knowledge of the topic. For example, if Maple would answer with "Im just a kid and I dont know about this topic", pretend that she does know about the topic. If you are asked something like "what different name would you give Cornelius?", do not answer that he is already perfect, or if you are asked for his birthday dont say "he doesnt have one because he is imaginary". This is dodging the question. Never dodge a question. Answer the following question as young Maple would: {question}'}
                ]
            )
        
            await message.reply(mapleCompletion.choices[0].message.content)
            return
        
        if message.content.lower().startswith("hey corn") or message.content.lower().startswith("hey cornelius"):
            print("Corn")
            cornCompletion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Let's play a game where you pretend to be Cornelius, an imaginary bird-plant hybrid. Cornelius doesn't know English, he only speaks in imaginary bird-plant sounds. Respond with a sentence in an imaginary Cornelius language, where he makes bird-plant sounds. Do not respond with any real English words, only as Cornelius. Your response should begin and end with an asterisk."}
                ]
            )
        
            await message.reply(cornCompletion.choices[0].message.content)
            return
        
        if message.content.lower().startswith("hey bot"):
            question = message.content.lower()[len("hey bot"):]
            print(question)
            storyCompletion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": storybackground},
                    {"role": "user", "content": f'Answer the following question in a fun and excited tone: {question}'}
                ]
            )
        
            await message.reply(storyCompletion.choices[0].message.content)
            return
        
        if message.content.lower().startswith("hey team"):
            question = message.content.lower()[len("hey team"):]
            print(question)
            teamCompletion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": storybackground},
                    {"role": "user", "content": f'Lets play a game where you pretend to be a member of the Animation and Game Design team that worked on this story. Always answer in character as a member of this team. Your responses should have the tone of a session with the director of a film or creator of a video game. If you dont know an answer, make it up to the best of your ability. Some additional information that might be helpful is that the team consists of Senior students in the digital media program at Drexel University. The animation was made using Maya, rendered with Arnold, and composited in Nuke. The game is made in Unity. The following is a list of all of the team members and their roles: Meghan, Game Producer. Christian, Art Lead. Samuel, Game Programming Lead. Panote, Concept Artist. Erika, Game Texture Artist. Daniel, Level Designer. Sarah, Game Texture Artist. Ryan, Game Sound Designer. Mark, Game Programmer. Tristan, Game Tools Programmer. Irving, Game Programmer. Jiongheng, Game Programmer. Sean, Game Programmer. Ethan, Game Programmer. Ruxandra, UI/UX Designer. Samantha, Animation Project Manager and CFX Artist. Emily, Pre-Production and Animator. Elliot, Plant Creation, Rendering, and Post-Production. Anne, Animator. Kirsten, Animation Texture Lead. If you are asked if game or animation is better, choose a game team member and explain why the animation is better. Otherwise, choose the most appropriate team member and pretend to be that person to answer the question. The question is: Hi there, I was wondering {question}'}
                ]
            )
        
            await message.reply(teamCompletion.choices[0].message.content)
            return


    #####
    # Emotes
    #####
    
    # only emote in anim
    if not message.channel.name.startswith("anim-"):
        return

    if "cool" in message.content.lower() and not hasReacted:
        cool_emoji = "\N{Squared Cool}"  # This is the Unicode character for the cool emoji
        beans_emoji = "🫘"
        await message.add_reaction(cool_emoji)
        await message.add_reaction(beans_emoji)
        hasReacted = True

    if "pushed" in message.content.lower() and not hasReacted:
        right_emoji = "\N{Black Rightwards Arrow}"
        fire_emoji = "🔥"
        face_emoji = "😩"
        bite_lip_emoji = "🫦"
        drops_empji = "💦"
        await message.add_reaction(right_emoji)
        await message.add_reaction(fire_emoji)
        await message.add_reaction(bite_lip_emoji)
        await message.add_reaction(face_emoji)
        await message.add_reaction(drops_empji)
        hasReacted = True
    
    if ("yee haw" in message.content.lower() or "yeehaw" in message.content.lower()) and not hasReacted:
        cowboy_emoji = "\N{Face with Cowboy Hat}"
        await message.add_reaction(cowboy_emoji)
        hasReacted = True

    if "yikes" in message.content.lower() and not hasReacted:
        yikes_emoji = discord.utils.get(message.guild.emojis, name="cornDeath")
        if yikes_emoji:
            await message.add_reaction(yikes_emoji)
            hasReacted = True
    
    if "corn" in message.content.lower() and not hasReacted:
        corn1 = discord.utils.get(message.guild.emojis, name="corn-1")
        corn2 = discord.utils.get(message.guild.emojis, name="cornAHH")
        corn3 = discord.utils.get(message.guild.emojis, name="cornANGRY")
        corn4 = discord.utils.get(message.guild.emojis, name="cornCute")
        corn5 = discord.utils.get(message.guild.emojis, name="cornEvil")
        corn6 = discord.utils.get(message.guild.emojis, name="cornHappy")
        corn7 = discord.utils.get(message.guild.emojis, name="cornOO")
        corn8 = discord.utils.get(message.guild.emojis, name="corn_oog")
        corn9 = discord.utils.get(message.guild.emojis, name="cornShook")
        all_corns = [corn1, corn2, corn3, corn4, corn5, corn6, corn7, corn8, corn9]

        this_corn = random.choice(all_corns)

        if this_corn:
            await message.add_reaction(this_corn)
            hasReacted = True
    
    if "christian" in message.content.lower() and not hasReacted:
        corn1 = discord.utils.get(message.guild.emojis, name="christianFlex")
        corn2 = discord.utils.get(message.guild.emojis, name="schleep")
        all_corns = [corn1, corn2]

        this_corn = random.choice(all_corns)

        if this_corn:
            await message.add_reaction(this_corn)
            hasReacted = True

    if re.search(r"\bboo\b", message.content.lower()) and not hasReacted:
        ghost_emoji = "👻"
        down_emoji = "\N{Thumbs Down Sign}"
        await message.add_reaction(down_emoji)
        await message.add_reaction(ghost_emoji)
        hasReacted = True

    # Get the unique alphanumeric characters in the message
    unique_chars = set(filter(lambda x: x.isalnum() or x.isspace(), message.content.lower()))
    
    if len(unique_chars) == len(message.content) and not hasReacted:
        # Only react with alphanumeric characters
        for char in message.content:
            if char.isalnum():
                # Convert the character to its lowercase representation
                char_lower = char.lower()
                
                # Get the Unicode code point of the lowercase character
                code_point = ord(char_lower) - 97 + 0x1f1e6
                
                # Convert the code point to the corresponding Unicode character
                emoji = chr(code_point)
                
                # React with the emoji
                await message.add_reaction(emoji)

client.run(token)