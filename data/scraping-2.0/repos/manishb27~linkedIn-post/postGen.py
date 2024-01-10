import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")


def postGen(user_provided_inpur):

    prompt = f'generate the Linkedin post of around 100 words with hashtags around the following User Provided Input: "{user_provided_inpur}" \n\nOutput:' 
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.9,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,)
    return response.choices[0].text


def image_prompt_gen(user_provided_inpur):

    prompt = f'generate a creative image description in not more than 10 words \n\nUser Provided Input: "{user_provided_inpur}" \n\nOutput:' 
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.9,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,)
    return response.choices[0].text



if __name__ == "__main__":
    # print(postGen("Launch of a new product of auto login once someone enters the museum."))
    print(image_prompt_gen("Launch of a new product of auto login once someone enters the museum."))