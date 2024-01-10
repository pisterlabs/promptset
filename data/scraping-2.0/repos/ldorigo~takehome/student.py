import json
import time
from llm import OpenAifunction, OpenaiChatMessage, get_response_openai_nonstream


async def answer_question_as_student(
        frq: str,
        text: str, 
        answer_description: str 
): 
    
    prompt = f"""
You are tasked with writing an answer to the given question as a fourth-grader would write it. You receive a text and a question, and are tasked with producing an answer to the question using the text. The answer should be at least 150 words long. Make sure to emulate the writing style of a fourth-grader. Additionally, the quality of your answer should be as follows:

{answer_description}

"""


    messages_for_openai = [
        OpenaiChatMessage(
            role="system",  
            content=prompt,
        ),
        OpenaiChatMessage(
            role="user",
            content=f"""
Text: 

{text}

===================
Question:

{frq}
"""

        ),
    ]

    
    print("Sending to OpenAI")
    start_time = time.time()
    response = await get_response_openai_nonstream(
        messages_for_openai,
    )
    print(f"OpenAI response time: {time.time() - start_time}")
    return response

if __name__ == "__main__":
    text =  """A baseball uniform is a type of uniform worn by baseball players, and by some non-playing personnel, such as field managers and coaches. It is worn to indicate the person's role in the game and\u2014through the use of logos, colors, and numbers\u2014to identify the teams and their players, managers, and coaches.Traditionally, home uniforms display the team name on the front, while away uniforms display the team's home location. In modern times, however, exceptions to this pattern have become common, with teams using their team name on both uniforms. Most teams also have one or more alternate uniforms, usually consisting of the primary or secondary team color on the vest instead of the usual white or gray. In the past few decades throwback uniforms have become popular.The New York Knickerbockers were the first baseball team to use uniforms, taking the field on April 4, 1849, in pants made of blue wool, white flannel shirts (jerseys) and straw hats. Caps and other types of headgear have been a part of baseball uniforms from the beginning. Baseball teams often wore full-brimmed straw hats or no cap at all since there was no official rule regarding headgear. Under the 1882 uniform rules, players on the same team wore uniforms of different colors and patterns that indicated which position they played. This rule was soon abandoned as impractical.In the late 1880s, Detroit and Washington of the National League and Brooklyn of the American Association were the first to wear striped uniforms. By the end of the 19th century, teams began the practice of having two different uniforms, one for when they played at home in their own baseball stadium and a different one for when they played away (on the road) at the other team's ballpark. It became common to wear white pants with a white color vest at home and gray pants with a gray or solid (dark) colored vest when away. By 1900, both home and away uniforms were standard across the major leagues.In June 2021, MLB announced a long-term deal with cryptocurrency exchange FTX, which includes the FTX logo appearing on umpire uniforms during all games. FTX is MLB's first-ever umpire uniform patch partner. On November 11, 2022, FTX filed for Chapter 11 bankruptcy protection. MLB removed the FTX patches from umpires' uniforms before the 2023 season."""

    
    frq = "Analyze the evolution of baseball uniforms throughout history as described in the text. What significant changes have occurred and what factors might have contributed to these changes?"

    async def main():
        response = await answer_question_as_student(frq, text, "test")
        print(response)
    import asyncio
    asyncio.run(main())
    





