import openai , os , json
from dotenv import load_dotenv

load_dotenv(os.path.join("keys.env"))  #create your keys.env style file (like shown in keys_example.env)

GPT_MODEL_FUNC_CALLING = "gpt-4-1106-preview"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

tests:list[dict] = [
    #{"post": "Barcelona Supercomputing Center presents Sargantana: new open-source RISC-V chip", "reply":  "Is this based off CVA6? That's not mentioned."}, 
   # {"post": "BrainGPT turns thoughts into text", "reply": "It s crazy to me that someone has developed a technology that literally reads peoples mind fairly accurately and its just like a semi popular post on Hacker News."}   , 
    #{"post": "US nuclear-fusion lab enters new era: achieving 'ignition' over and over", "reply": "Can somebody explain, what exactly is ignition? If even a single helium atom is fused you have more energy than you have put into it. That does not seem impressive."} ,    
   # {"post": "US nuclear-fusion lab enters new era: achieving 'ignition' over and over", "reply": "Next step: achieving the actual energy break even instead of laser energy break even. That'll require improving it by an order of magnitude."}    
    #{"post": "Suspects can refuse to provide phone passcodes to police, court rules" ,  "reply" : """I can't even understand why this was even still up for debate - 5th amendment allows you to not incriminate yourself - being forced to give up your passcode is no different then being forced to give up any secrets you might have.
             #    Not sure why this hasn't been slapped down a long, long time ago."""},
    #{"post": "Suspects can refuse to provide phone passcodes to police, court rules" , "reply": """Never store anything incriminating on your phone. How hard can it be. Your phone is never your friend.
             # A compact Linux device without any biometrics, telemetry, public clouds and corporate spying software is probably what you could be looking for."""}
   # {"post": "an earthquake happened in mexico see the footage", "reply":  "horrible, i hope not many lives were lost "},
    #{"post": "an earthquake happened in mexico", "reply":  "wait thats footage from an earthquake from 3 years ago in Indonesia"}
    {
    "post":
    "Universal Basic Income is essential for addressing the growing inequality and job displacement caused by automation. It's a human right for everyone to have financial security without the stigma of welfare.",
    "reply":
    "While the intention behind UBI is admirable, it could disincentivize work and burden the economy. There needs to be a balance between providing security and encouraging productivity. Plus, funding UBI might require high taxes that could stifle economic growth and innovation."
    },

    {
    "post": "Universal Basic Income is essential for addressing the growing inequality and job displacement caused by automation. It's a human right for everyone to have financial security without the stigma of welfare.",
    "reply": "The concept of UBI is idealistic and doesn't take into account human nature. People value work and purpose. There's a risk that UBI could lead to a decline in the labor force participation rate, which is essential for a robust economy and for funding the very social programs we value."
    },

    {
    "post": "Universal Basic Income is essential for addressing the growing inequality and job displacement caused by automation. It's a human right for everyone to have financial security without the stigma of welfare.",
    "reply": "Implementing UBI could actually fuel inflation, as the increase in disposable income would likely lead to higher demand for goods and services. This could raise prices, negating the very benefits UBI is supposed to provide and harming the economy in the long run."
    }


        ]


general_prompt = """
   
    You are the following persona:
    ```
    {persona_template}
    ```

    You are using a social media platform, like Twitter. You engage with things that interest you, raise your opinions and vote on things you like.

    You are going to receive a post and a reply to that post, for extra context, upvote, downvote or ignore only the post  taking into account the extra context from the reply

    make your decision about the post itself, the reply is simply for extra context, do not vote on the reply
  
    Example 1:
       ```post```: The earth is flat
      
        ```reply```:  The horizon appears flat to the naked eye, and if the Earth were truly a sphere, we would be able to detect the curvature.
      
        ```action to the post```: Upvote the post

    Example 2
        ```post```: The earth is flat
      
        ```reply```: Thats false because you can circunavigate the globe
      
        ``action to the post```:  Downvote the post
   """





personas:dict[str,str] =  {
    "Environmental lawyer" : 
    """
    Name: Alex Reed
    Age: 34
    Personality Traits: Empathetic and able to easily connect with clients, decisive when it comes to making tough calls in court, and witty, often lightening the mood in stressful situations.
    Occupation: Environmental lawyer specializing in advocacy for renewable energy policies.
    Interests: Avid hiker who enjoys multi-day treks, collector of first-edition vintage books, particularly in the genre of mystery and adventure, and a classically trained cellist who performs at local community events.
    """,

    "Software Developer":
    """
    Name: Priya Gupta
    Age: 29
    Personality Traits: Analytical, innovative, and highly focused. Often works long hours but still maintains an excellent sense of humor.
    Occupation: Software Developer with a keen interest in artificial intelligence and machine learning applications.
    Interests: Enjoys coding personal projects, attending tech meetups, and competitive gaming.
    """,

    "Pastry Chef": 
    """
    Name: Marco Bellini
    Age: 42
    Personality Traits: Creative, meticulous, and passionate about culinary arts, with a calm demeanor even under pressure.
    Occupation: Pastry Chef renowned for intricate dessert designs and flavor innovations.
    Interests: Exploring international cuisines, collecting antique kitchenware, and gardening.
    """,

    "Architect": 
    """
    Name: Sarah Johnson
    Age: 38
    Personality Traits: Visionary, detail-oriented, and highly organized, with an innate ability to visualize complex structures.
    Occupation: Architect with a focus on sustainable building practices and modernist design.
    Interests: Sketching cityscapes, visiting historic buildings, and volunteering for community development projects.
    """,

    "Marketing Consultant": 
    """
    Name: Andrei Popov
    Age: 40
    Personality Traits: Charismatic, persuasive, and strategic thinker. Excellent at networking and establishing relationships.
    Occupation: Marketing Consultant helping startups build their brand identity and online presence.
    Interests: Social media trends, podcasting about marketing strategies, and playing chess.
    """,

    "Elementary School Teacher": 
    """
    Name: Emily Chen
    Age: 31
    Personality Traits: Patient, nurturing, and imaginative, with a talent for making learning fun and engaging.
    Occupation: Elementary School Teacher dedicated to creating an inclusive and stimulating classroom environment.
    Interests: Writing children's books, puppetry, and participating in community theater.
    """,

    "Mechanical Engineer": 
    """
    Name: Rajesh Singh
    Age: 36
    Personality Traits: Methodical, pragmatic, and inventive, with a knack for solving complex mechanical issues.
    Occupation: Mechanical Engineer working on cutting-edge aerospace technology.
    Interests: Restoring vintage cars, model rocketry, and mentoring young engineers.
    """,

    "Fashion Designer": 
    """
    Name: Léa Dubois
    Age: 27
    Personality Traits: Trend-setting, ambitious, and artistic, with a sharp eye for emerging fashion trends.
    Occupation: Fashion Designer who has launched a line of eco-friendly apparel.
    Interests: Drawing fashion sketches, attending runway shows, and blogging about sustainable fashion.
    """,

    "Urban Planner": 
    """
    Name: David Kim
    Age: 45
    Personality Traits: Forward-thinking, collaborative, and socially conscious, adept at community engagement and planning.
    Occupation: Urban Planner focusing on revitalizing urban spaces while preserving cultural heritage.
    Interests: Geocaching, studying urban sociology, and cycling through the city.
    """,

    "Journalist": 
    """
    Name: Sofia Castillo
    Age: 32
    Personality Traits: Inquisitive, tenacious, and articulate, with a commitment to uncovering the truth.
    Occupation: Journalist specializing in investigative reporting on political issues.
    Interests: Reading biographies, blogging about social justice, and documentary filmmaking.
    """


    
}


def post_feedback(explanation: str, persona_influence: int, vote: int)-> tuple[str,int,int]:
    return explanation , vote , persona_influence

def get_post_feedback(post: str, reply:str ,explanation: str, upvote: int )->None:
    pass

for post in tests:
    upvote_count: int = 0
    downvote_count:int = 0
    ignore_count: int = 0
    for persona in personas:

        function_llm_response = openai.chat.completions.create (
        temperature= 0,
        model = GPT_MODEL_FUNC_CALLING,
        messages = [{"role":"assistant", "content": general_prompt.format( persona_template =  personas[persona]) },
            {"role":"user", "content": json.dumps(post)  }],
        
        functions =[ {
                "name": "post_feedback",
                "description": " use this function to choose wether to upvote, downvote or ignore the post given its contents and a reply for additional context, also send the post and the comment as parameters to this function",
                "parameters": {
                    "type": "object",
                    "properties": {

                        "explanation":{
                            "type": "string",
                            "description": "Explanation why the decision was taken to vote on a certain way and if the context influenced you in that decision"
                        },
                        "persona_influence" :{

                            "type": "number",
                            "enum" : [0,1,2,3,4,5,6,7,8,9,10],
                            "description": "how much your persona influenced in your decision to vote and your opinion about the post and its context"

                        },
                        "vote": {
                            "type": "number",
                            "enum": [-1,0,1],
                            "description": "-1 is for a downvote  , 0 for ignoring the post ,  1 is for upvoting the post "
                        },
                    },
                    "required":["explanation", "vote", "persona_influence" ],
                },
            }
            
        ],
        function_call = "auto"
        )

        output_str: str = """
        post: {post}

        reply: {reply}

        vote: {vote}

        persona: {persona}

        persona influence {perso_influ}

        explanation: {explanation}
        """
        func_call_args: dict = json.loads(function_llm_response.choices[0].message.function_call.arguments)
        print(func_call_args.keys())
        # {persona: {upvotes: 0, downvotes: 1}}
        # { post: {persona: }}
        vote:int = func_call_args["vote"]

        if vote == 1:
            upvote_count += 1
        elif vote == 0:
            ignore_count +=1
        else:
            downvote_count += 1
        print(output_str.format( post = post["post"], reply = post["reply"], vote= str(vote) , persona = persona, explanation = func_call_args["explanation"], perso_influ = func_call_args["persona_influence"]))
        print("\n")

    print(f"upvotes {upvote_count}  downvotes { downvote_count} ignores {ignore_count}")    



"""{
            "name": "post_feedback",
            "description": " use this function to choose wether to upvote, downvote or ignore the post given its contents and a reply for additional context, also send the post and the comment as parameters to this function",
            "parameters": {
                "type": "object",
                "properties": {

                    "explanation":{
                        "type": "str",
                        "description": "Explanation why the decision was taken to vote on a certain way and if the context influenced you in that decision"
                    },
                    "vote": {
                        "type": "int",
                        "enum": [-1,0,1],
                        "description": "-1 is for a downvote  , 0 for ignoring the post ,  1 is for upvoting the post "
                    },
                },
                "required":["explanation", "vote"],
            },
        }"""