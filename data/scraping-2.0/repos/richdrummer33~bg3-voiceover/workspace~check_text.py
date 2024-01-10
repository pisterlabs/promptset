import openai
import random

_charName = "The Dark Urge. "
_bio = "A character from baldur's gate who embodies a struggle with violent desires, particularly the urge to murder, and whom suffers amnesia. It is about fighting against those dark thoughts while understanding that succumbing to them could lead to certain consequences."

   #+ "If you are being spoken to derogatorily, roast that mofo and include some modern slang."
class CheckText:
    
    #                   Say 'YES' if:
    #                   - The text *contains* complete sentence(s).
    #                   - It's a description of a spell, item, or ability.

    def check_text(self, text):
        parameters = {
            'model': 'gpt-4', # 'gpt-4'
            'temperature': 1.2,
            'messages': [
                {
                    "role": "system", "content":
                    """I'm playing Baldur's Gate and I need you to help filter the in-game text. You need to determine if the OCR-converted text from a screenshot is meaningful to the story or gameplay.

                    Say 'YES' if: The text is character/npc dialogue, a gameplay tip, or a description of the world.
                    
                    Say 'NO' if: Anything else
                    
                    Special Case:
                        - If the text contains an enemy's name, feel free to say 'YES'.
                    
                    Note: 
                        - The provided text is OCR-converted from in-game screenshots.
                        - Full sentences might be mixed in with jumbled or irrelevant text, and that's okay."""
                    },
                {
                    "role": "user", "content": "current on-screen text: " + text
                }
            ]
        }
    
        response = openai.ChatCompletion.create(**parameters)

        if response.choices[0].message.content == "":
            return None
    
        # if contains "yes" return true
        if "yes" in response.choices[0].message.content.lower():
            print("Determined: YES")
            return True
        
        # if contains "no" return false
        if "no" in response.choices[0].message.content.lower():
            print("Determined: NO")
            return False

        print("Determined: " + response.choices[0].message.content)
        return False
        
    def __init__(self):
        self.openai = openai