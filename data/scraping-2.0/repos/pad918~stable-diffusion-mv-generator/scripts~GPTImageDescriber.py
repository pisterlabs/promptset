import openai
import os
import time
from typing import List
from scripts.PromptRefiner import PromptRefiner


class GPTImageDescriber(PromptRefiner):
    def __self__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    def generate_setting(self, prompts, additional_context, n=10):
        if(n<0):
            raise Exception("GPT is refusing api calls, you may have an internet problem!")
        try:
            combined_lines:str = '\n\n'.join(prompts)
            completion = openai.ChatCompletion.create(
            max_tokens = 300,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""
                    Create a short senario from the given lyrics, make up relevant part of the story. 
                    Make the story based on who it is about, what it is about, the meaning of it and 
                    the mood of the story.  
                    Some additional context about the song:\n{additional_context}\n
                    \n The lyrics of the song:\n{combined_lines}
                    """}
                ]   
            )
            result:str = completion.choices[0].message.content
            print(f"Generated setting: \n{result}")
            return result
            
        except Exception as e:
            print("Gpt error, retrying")
            time.sleep(3)
            return self.generate_setting(prompts, additional_context, n-1)

    def refine_lyric(self, line, setting):
        completion = openai.ChatCompletion.create(
        max_tokens = 150,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"""
                You are an artist that creates images for lines of lyrics in songs.
                Images are created by describing the images contents in detail by using keywords. 
                The image descriptions will be put into an AI that generates images from descriptions.
                Here are two examples of good descriptions:
                
                1. houses in front, houses background, straight houses, digital art, smooth, sharp focus, gravity falls style, doraemon style, shinchan style, anime style  
                2. cute girl, crop-top, blond hair, black glasses, stretching, with background by greg rutkowski makoto shinkai kyoto animation key art feminine mid shot

                The images form a story together. The story of the song is as follows: \n{setting}

                Describe the image for the following line of lyrics:
                {line}
                """}
            ]
        )
        result:str = completion.choices[0].message.content
        print("---------------------------------------\n")
        print(f"Generated: {result}")
        return result
    

    def refine(self, prompts: List[str], options) -> List[str]:
        if(not options['checkbox_gpt_refinement']):
            return prompts
        
        additional_context = options['gpt_context']
        setting = self.generate_setting(prompts, additional_context)

        #Avoid slow summarization if no prompts are supplied!
        if(len(prompts)==0):
            return prompts
        
        refined = []
        for line in prompts:
            for i in range(10):
                try:
                    ref = self.refine_lyric(line, setting)
                    refined.append(ref)
                    print(f"{len(refined)}/{len(prompts)}\n")
                    break
                except Exception:
                    print("Failed to generate, retrying")
                    time.sleep(3)
        return refined