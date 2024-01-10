import openai

from .chat_session import ChatSession


def create_image(prompt):
    return openai.Image.create(
        prompt=prompt, n=1, size="512x512", response_format="b64_json"
    )


GENERATE_IMAGE_DESCRIPTION_PROMPT = """
Given the following section of a short children's story involving the adventures of two cat siblings, 
write a very brief single line description summarizing the main plot element within. 
The most important thing is to say what the two cat siblings are doing, mention other major characters, 
and provide a settings for them. The characters should be simply described, rather than named. 
This will be used to generate a picture to accompany the text. Every description should begin "An oil painting of..."

The following characters may also be mentioned:

Miss Olive: A smart old owl who lives in the tree near the cats' house and gives good advice.
Daisy: A friendly and talkative squirrel who knows everything that's happening around.
Rusty: A kind, funny dog who lives with the cats and thinks they're his best friends.

Be sure to not use the names of Miss Olive, Daisy, or Rusty either, say "owl", "squirrel", and "dog" instead, 
and to only mention these characters if they appear in the story section.

Example: "An oil painting of two cats sitting in a tree while a dog sleeps at the base"

Story text beings now:
"""


def create_image_prompt(chat_history: ChatSession):
    story_ending = "\n\n".join(
        [x["content"] for x in chat_history.history if x["role"] != "system"][-3:]
    )

    prompt = f"{GENERATE_IMAGE_DESCRIPTION_PROMPT}\n\n{story_ending}"
    image_chat = ChatSession()
    image_chat.user_says(prompt)
    return image_chat.get_ai_response()
    # return openai.Completion.create(
    #     prompt=prompt, model="text-davinci-003", temperature=0.6,
    # )
