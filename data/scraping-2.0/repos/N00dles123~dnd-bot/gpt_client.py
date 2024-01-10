import openai

CLOSING = {'.', '?', '!'}


class GPTClient:

    def __init__(self, openai_api_key: str):
        openai.api_key = openai_api_key

    def query_gpt(self, prompt: str, temperature: float = 0.6, max_tokens: int = 420):
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                # We first describe GPT's role and then provide an example
                {'role': 'system',
                 'content': ('You are a skilled bard who narrates background stories for D&D characters '
                             'based on their name, race, class, background, alignment, and other provided traits.'
                             'Craft a concise background story that is under 2000 characters.')},
                {'role': 'user', 'content': prompt}
            ]
        )

        return response.choices[0].message.content

    def create_character_story(self, name, race, char_class, background, appearance,
                               alignment, personality, ideals, bonds, flaws) -> str:
        character_traits = {
            'name': name,
            'race': race,
            'class': char_class,
            'background': background,
            'appearance': appearance,
            'alignment': alignment,
            'personality_traits': personality,
            'ideals': ideals,
            'bonds': bonds,
            'flaws': flaws
        }

        formatted_traits = '\n'.join([f'{key}: {value}' for key, value in character_traits.items() if value is not None])
        character_story = self.query_gpt(formatted_traits)

        # Truncate story to last closing punctuation
        while character_story[-1] not in CLOSING or len(character_story) > 2000:
            # Reverse search for the last occurrence of a closing punctuation
            for i in range(len(character_story) - 2, -1, -1):  # Start from the end and go backwards
                if character_story[i] in CLOSING:
                    # Truncate the string at the last closing punctuation
                    character_story = character_story[:i + 1]
                    break
            # If no closing punctuation is found, leave the string as-is

        return character_story
