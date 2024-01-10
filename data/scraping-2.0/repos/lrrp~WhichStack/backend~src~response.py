import os

import openai

openai.api_key = os.getenv("OPENAI_KEY")

class Response():    
    async def request(self, ctx):   
        purpose = {"".join(f"{item}\n" for item in ctx["purpose"].values())}
        experience = {"".join(f"{item}\n" for item in ctx["experience"].values())}
        budget = {"".join(f"{item}\n" for item in ctx["budget"].values())}
        timeline = {"".join(f"{item}\n" for item in ctx["timeline"].values())}
        stacks = {"".join(f"{item}\n" for item in ctx["stacks"].values())}
        languages = {"".join(f"{item}\n" for item in ctx["languages"].values())}
        database = {"".join(f"{item}\n" for item in ctx["database"].values())}
        providers = {"".join(f"{item}\n" for item in ctx["providers"].values())}

        res = await openai.ChatCompletion.acreate(
          model="gpt-3.5-turbo",
          messages=[
                {
                    "role": "system",
                    "content": f'''

                        You'll be given a prompt.
                        The context is that a user has answered some questions.
                        You need to use the answers to those questions to pick them a tech stack for their project.
                        Make it human readable and friendly.
                        Make it look like you're telling the user and like it's you who's giving the suggestions
                        Don't title the paragraphs like "frontend:", use the titles within the paragraphs

                        Use their current knowledge to format your stack of choice
                        The stack must be a combination of the user's choices and must make sense to use in a real project.
                        There isn't a need to take into account every user input

                        Also tell them how the frameworks and languages you chose work well together. 

                        Tell them why you think the chosen tech stack is good based on what the user gave.

                        Structure the doc references like 'framework - url\n'.
                        And keep the docs at the bottom, telling them check them out.
                        Use normal bullet points and not numbered bullet points

                        The purpose of the project is: {purpose}
                        The user's experience in programming is: {experience}
                        The user's budget: {budget}
                        Their predicted timeframe for the project: {timeline}
                        The stacks they currently know: {stacks}
                        The programming languages they know: {languages}
                        Database hosts they are familar with: {database}
                        Hosting providers they are familiar with: {providers}
#
                    '''         
                }
            ]
        )

        if not res["choices"][0]["finish_reason"] == "stop":
            return None
        
        return res