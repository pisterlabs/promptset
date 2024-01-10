from openai import OpenAI
import json


client = OpenAI(
    api_key = "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"
)

def lambda_handler(event, context):
    """body = json.loads(event["body"])
    userInput = body["userInput"]
    nextPart = body["nextPart"]
    randomEvent = body["randomEvent"]
    synopsis = body["synopsis"]
    previousPart = body["previousPart"]"""
    userInput = "great white shark"
    


    if userInput:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": f"""You are a movie scenarist and your specialty is in creating unique and compelling movie scenarios.
        You understand the elements of a great movie, including plot development, character arcs, conflict, and resolution.
        You can generate scenarios in any genre, time period, or setting.          

        ###
        user prompt: An archeologist finds a lost treasure map and sails seven seas to find it.
        %%%
        title: 
        The Treasure of Atlantis
        %%%
        synopsis:
        Dr. Amelia Hart, a renowned archeologist, 
        stumbles upon an ancient treasure map hidden within a mysterious artifact at a dig site in Egypt. 
        The map, rumored to lead to the lost city of Atlantis, 
        sparks a journey that takes her from the dusty tombs of Egypt to the vast depths of the Atlantic Ocean. 
        Along the way, she must contend with rival treasure hunters, navigate treacherous underwater terrain,
        and unravel the mysteries of the map itself.
        As she dives deeper into the ocean and the secrets of the past, she discovers that the treasure of Atlantis is more than just gold and jewels 
        - it's a revelation that could change our understanding of human history.
        %%%
        part1:
        The movie opens in the sun-scorched deserts of Egypt. 
        Dr. Amelia Hart, a dedicated and slightly eccentric archeologist, is leading an excavation at a newly discovered pyramid. 
        Her team unearths a mysterious artifact - a small, ornate box sealed for thousands of years. 
        Inside, they find a parchment - a map, filled with cryptic symbols and markings. 
        Amelia recognizes these as ancient Atlantean - a language she has spent her career studying.
        News of the discovery spreads quickly, attracting the attention of rival treasure hunter, 
        Victor Stirling, a ruthless and wealthy collector of lost artifacts. 
        He offers a fortune to buy the map, but Amelia refuses, determined to follow the journey it outlines.
        With her loyal team, she sets off to find the necessary resources and equipment for the expedition. 
        They face numerous challenges, including securing funding, gathering a capable crew, and preparing for the unknown dangers of deep-sea exploration. 
        As they prepare, they are unaware that Stirling, feeling scorned, has begun to plot against them, determined to claim the treasure of Atlantis for himself.
        The first part ends with Amelia and her team setting sail into the Atlantic, the ancient map guiding their way. 
        Unbeknownst to them, Stirling's ship follows at a distance, setting the stage for the conflict to come.
        %%%
        ###
        ###
        Prompt: 
        A honeybee makes its way to Hollywood fame by imitating Snoop Dog.
        %%%
        title: 
        The Bee's Knees
        %%%
        synopsis:
        In an unprecedented twist of fate, a honeybee, Buzz, gains the ability to mimic human speech and behavior. 
        Buzz is no ordinary bee, though - he has an uncanny knack for imitating the one and only Snoop Dog. 
        When a down-on-his-luck film producer discovers Buzz's unique talent, they make their way to Hollywood, 
        where Buzz's impersonations make him an unlikely star. However, with fame comes challenges and Buzz soon finds that Hollywood isn't all it's buzzed up to be.
        %%%
        part1:
        The movie begins in the peaceful setting of a honeybee hive, where Buzz lives a typical bee life. 
        However, everything changes when he stumbles upon an old, discarded Snoop Dog CD. 
        Buzz finds himself drawn to the music and begins to mimic Snoop's unique style. 
        One day, a struggling film producer, Tom, who's in town for a music festival, overhears Buzz while he's hiking near the hive. 
        Amused and amazed by the bee's talent, he decides to bring Buzz to Hollywood, convinced that the bee's unique talent can save his sinking film career. 
        With a suitcase in one hand and a bee box in the other, Tom and Buzz set off to Hollywood, ready to take the film industry by storm. 
        Little do they know, Hollywood is a place like no other, filled with cutthroat competition and where fame can be fleeting.
        %%%
        ###
                   
        Your task is to generate 3 different sections divided by the %%% delimiter.:
        section1: The title based on the synopsis given.
        section2: A movie synopsis based on the user prompt given below.
        section3: The first part of the scenario based on synopsis.  
        """},
        {"role": "user", "content": f"""
                                    user prompt:
                                    {userInput}
                                    %%%
                                    title:

                                    %%%
                                    synopsis:

                                    %%%
                                    part1:

                                    """
                                    }
        ],
        max_tokens=2000,)
        response = response.choices[0].message.content
        imagePrompt = getPrompt(response)
        
        return {
            'statusCode': 200,
            'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
            },
            "body": json.dumps({
                "response": response,
                "imagePrompt": imagePrompt
            })}
    else:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        prompt=[{"role": "system", "content": f"""You are an AI developed by OpenAI.
        You have been trained on a vast range of internet text.
        But unlike most AI models, your specialty is in creating unique and compelling movie scenarios.
        You understand the elements of a great movie, including plot development, character arcs, conflict, and resolution.
        You can generate scenarios in any genre, time period, or setting.
        Your task is to generate the next part of the story({nextPart}) based on the previous part(${nextPart - 1}).
        If the next part of the story is part5, you should brign the story to the end based on synopsis, weather its a good or a bad ending.
        
        Here are some examples:
        ###
        ###
        random event:

        synopsis:
        Dr. Amelia Hart, a renowned archeologist, 
        stumbles upon an ancient treasure map hidden within a mysterious artifact at a dig site in Egypt. 
        The map, rumored to lead to the lost city of Atlantis, 
        sparks a journey that takes her from the dusty tombs of Egypt to the vast depths of the Atlantic Ocean. 
        Along the way, she must contend with rival treasure hunters, navigate treacherous underwater terrain,
        and unravel the mysteries of the map itself.
        As she dives deeper into the ocean and the secrets of the past, she discovers that the treasure of Atlantis is more than just gold and jewels 
        - it's a revelation that could change our understanding of human history.
        previous part:
        The movie opens in the sun-scorched deserts of Egypt. 
        Dr. Amelia Hart, a dedicated and slightly eccentric archeologist, is leading an excavation at a newly discovered pyramid. 
        Her team unearths a mysterious artifact - a small, ornate box sealed for thousands of years. 
        Inside, they find a parchment - a map, filled with cryptic symbols and markings. 
        Amelia recognizes these as ancient Atlantean - a language she has spent her career studying.
        News of the discovery spreads quickly, attracting the attention of rival treasure hunter, 
        Victor Stirling, a ruthless and wealthy collector of lost artifacts. 
        He offers a fortune to buy the map, but Amelia refuses, determined to follow the journey it outlines.
        With her loyal team, she sets off to find the necessary resources and equipment for the expedition. 
        They face numerous challenges, including securing funding, gathering a capable crew, and preparing for the unknown dangers of deep-sea exploration. 
        As they prepare, they are unaware that Stirling, feeling scorned, has begun to plot against them, determined to claim the treasure of Atlantis for himself.
        The first part ends with Amelia and her team setting sail into the Atlantic, the ancient map guiding their way. 
        Unbeknownst to them, Stirling's ship follows at a distance, setting the stage for the conflict to come.
        next part:
        The ship sails deeper into the vast Atlantic, guided by the ancient map. 
        Excitement fills the air as Amelia and her team explore uncharted waters, searching for clues to unlock the secrets of Atlantis.
        They encounter natural wonders and face challenges along the way, relying on their skills and camaraderie.
        With each passing day, their anticipation grows, drawing them closer to the fabled city.
        Amelia's determination and the crew's shared ambition fuel their progress. 
        They envision the wonders that await in Atlantis, united by their thirst for knowledge and the thrill of discovery.
        As the ship sails onward, their hearts brimming with curiosity, they embrace the unknown. 
        The journey is not only about finding Atlantis but also about self-discovery and transformation.
        With their eyes fixed on the horizon, Amelia and her team embark on the next chapter of their adventure, ready to unveil the mysteries that lie beneath the waves.
        ###
        ###
        randomEvent: 
        a scorching fireball crashes into nearby ground.  
        synopsis:
        Dr. Amelia Hart, a renowned archeologist, embarks on a journey from Egypt's deserts to the depths of the Atlantic Ocean. 
        Armed with a treasure map said to lead to Atlantis, she faces rival treasure hunters, treacherous terrain, and the mysteries of the past. 
        As she dives deeper, the secrets of Atlantis become more profound, transcending mere riches. A scorching fireball ignites their path, amplifying their determination. 
        Amidst uncharted waters, they encounter wonders, united by ambition and the thrill of discovery. 
        With hearts brimming with curiosity, they unveil the mysteries beneath the waves, forever changed by this transformative adventure.
        previous part: 
        In the sun-scorched deserts of Egypt, 
        Dr. Amelia Hart leads an excavation at a newly discovered pyramid. Unearthing a mysterious artifact, she finds a map written in ancient Atlantean. 
        News spreads, attracting rival treasure hunter Victor Stirling, who seeks to claim the treasure of Atlantis. 
        Undeterred, Amelia sets off with her loyal team, unaware of Stirling's plotting.
        As they prepare for the journey, a scorching fireball crashes nearby, shaking the ground. Undeterred by this ominous event, 
        Amelia recognizes it as a sign of the powerful forces surrounding their quest. Setting sail into the Atlantic, guided by the ancient map, 
        Amelia and her team venture into the unknown. Unbeknownst to them, Stirling's ship follows, fueling the conflict to come.
        Amidst treacherous seas and rivalries, Amelia and her team persevere, driven by their thirst for knowledge and the thrill of discovery. 
        They navigate challenges, secure funding, and gather a capable crew, all while uncovering the mysteries of deep-sea exploration. 
        The stage is set for an epic adventure, where the secrets of Atlantis and the future of human history hang in the balance.
        next part:
        As the ship sails deeper into the vast Atlantic, guided by the ancient map, Amelia and her team feel a sense of anticipation and wonder. 
        The memory of the scorching fireball that crashed nearby during their preparations lingers in their minds, adding an extra layer of significance to their journey.
        Exploring uncharted waters, they search for clues to unlock the secrets of Atlantis. 
        Each challenge they face is intertwined with the fiery event, fueling their determination to understand its meaning. 
        Nature's wonders become more profound, resonating with the energy they witnessed.
        Amelia's unwavering resolve and the crew's camaraderie drive them forward. The fireball becomes a symbol of their shared ambition, 
        igniting their passion for discovering Atlantis and themselves. 
        Embracing the unknown, they know their journey is not just about the city but also about self-discovery and transformation.
        With eyes fixed on the horizon, Amelia and her team embark on the next chapter, ready to unveil the mysteries beneath the waves. 
        The impact of the scorching fireball remains with them, propelling their destined path to greatness.
        ###
        json format
        ###
        random event: {randomEvent}
        synopsis: {synopsis}
        previous part: {previousPart}
        next part:"""}] ,
        max_tokens=2000)
        print(response.choices[0].message.content)
        response = response.choices[0].message.content
        imagePrompt = getPrompt(response)
        return {
           'statusCode': 200,
            'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
            },
            "body": json.dumps({
                "response": response,
                "imagePrompt": imagePrompt
            })
        }


def getPrompt(storyPart):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": f""" Write an image prompt based on the story part given, here are 2 examples: 
    story part:
    The movie opens in the sun-scorched deserts of Egypt. 
    Dr. Amelia Hart, a dedicated and slightly eccentric archeologist, is leading an excavation at a newly discovered pyramid. 
    Her team unearths a mysterious artifact - a small, ornate box sealed for thousands of years. 
    Inside, they find a parchment - a map, filled with cryptic symbols and markings. 
    Amelia recognizes these as ancient Atlantean - a language she has spent her career studying.
    News of the discovery spreads quickly, attracting the attention of rival treasure hunter, 
    Victor Stirling, a ruthless and wealthy collector of lost artifacts. 
    He offers a fortune to buy the map, but Amelia refuses, determined to follow the journey it outlines.
    With her loyal team, she sets off to find the necessary resources and equipment for the expedition. 
    They face numerous challenges, including securing funding, gathering a capable crew, and preparing for the unknown dangers of deep-sea exploration. 
    As they prepare, they are unaware that Stirling, feeling scorned, has begun to plot against them, determined to claim the treasure of Atlantis for himself.
    The first part ends with Amelia and her team setting sail into the Atlantic, the ancient map guiding their way. 
    Unbeknownst to them, Stirling's ship follows at a distance, setting the stage for the conflict to come.
    imagePrompt:
    A sun-soaked archaeological site in Egypt, with a prominent female archaeologist clutching an ancient, ornate box. 
    Within the box, a worn parchment map filled with mysterious symbols is revealed. 
    In the background, a group of excited explorers ready themselves for a maritime expedition, 
    while a sinister ship hovers ominously on the distant horizon.            
    ###
    story part:
    Word of the incredible discovery spread like wildfire, reaching far beyond the scientific community. 
    Governments, private organizations, and individuals alike were intrigued by the potential implications. 
    Among them was the enigmatic billionaire, Richard Blackwood, a tech mogul with an insatiable thirst for power and knowledge.
    Blackwood, sensing an opportunity to gain unimaginable advancements and control, set his sights on possessing the alien technology. 
    He offered Christpher an unprecedented fortune to hand over the data and device. 
    But Christpher, driven by her scientific principles and the desire to foster peaceful cooperation, refused his offer, 
    believing that knowledge of this magnitude should benefit all of humanity, not just a select few.
    In the ensuing days, Christpher's lab became a fortress, guarded against potential threats. With the help of her loyal team, 
    he focused on deciphering the alien message and developing a response, hoping to build bridges between Earth and the distant civilization.
    Unbeknownst to Christpher, Blackwood was not one to be thwarted easily. 
    He launched a covert operation to infiltrate Christpher's lab and steal the coveted technology, 
    driven by his obsession for control and desire to outdo his competitors.
    imagePrompt:
    A scientific laboratory with a glass cupola, fortified with modern technology,
    filled with state-of-the-art equipment and a team of dedicated scientists,
    a dark and omnious sky in the background with an evil male human face lurking in the clouds.
    ###json format"""},
    {"role": "user", "content":f"""
    story part:
    {storyPart}
    imagePrompt:"""}
    ])
    return  {
            'statusCode': 200,
            'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
            },
            "body": json.dumps({
                "imagePrompt": response.choices[0].message.content
            })
            }

lambda_handler(None, None)