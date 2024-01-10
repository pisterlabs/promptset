import openai
import os

def extract_bullet_points(text):
    """
    Extract bullet points from a string and return a list of bullet points.
    
    Args:
        text (str): The input text containing bullet points.
        
    Returns:
        list: A list of bullet points.
    """
    bullet_points = []
    lines = text.split("\n")
    for line in lines:
        # Check if the line starts with a bullet point (e.g. "1. ", "2. ", etc.)
        if line.strip().startswith(("* ", "- ", "• ", "· ", "1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. ")):
            bullet_points.append(line.strip()[2:])
    return bullet_points


class KeyTakeaways:
    
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    def generate_key_takeaways(self, text_chunks_lib:dict) -> list:
        PROMPT = """
            You are a super intelligent human and helpful assistant. 
            I am giving you parts of a video transcription that I want to learn from.
            In bullet points, give me at most 3 key takeaways from this text.
        """
        
        final_takeaways = []
        for key in text_chunks_lib:
            for text_chunk in text_chunks_lib[key]:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=PROMPT + text_chunk,
                    temperature=0.4,
                    max_tokens=1024,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.6,
                )
                takeaways = extract_bullet_points(response.choices[0].text.strip())
                final_takeaways.extend(takeaways)
        
        
        return final_takeaways