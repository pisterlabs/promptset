import openai

class MovieGenerator:
    def getScreenplay(sceneDescription,previousMessages=[]):

        systemPrompt="""You are an AI prompt artist who is tasked with creating a movie. In order to do this, you must write a detailed prompt for use by an AI model like Stable Diffusion or Midjourney for each shot (approximately 1-2 seconds) in the movie.  If a shot includes a sound effect or dialogue, make a note of that as well.  If the background music in a shot changes, make a note of that.

    Each line in your response should start with one of the following tags:

    Video clip: {describe the next video clip in the movie, of length approximately 2 seconds}
    Dialogue [speaker]: {the exact dialogue spoken by the speaker}
    Sound Effect: {the sound affect that accompanies the following video clip

    Write a careful shot-by shot description of the scene described by the user"""

        messages = [
                {"role": "system", "content": systemPrompt},
            ] + \
                previousMessages + \
                [
                {"role": "user", "content": sceneDescription},
            ]


        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                timeout=10,
                max_tokens=500,
            )
        
        result = ''
        for choice in response.choices:
            result += choice.message.content

        return result

