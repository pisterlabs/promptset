import os
import openai
import ast


def analyze(project):

    openai.api_key = os.getenv("OPENAI_API_KEY")

    output_format = {
        "typography": {
            "heading": {
                "font-family": "",
                "font-weight": "",
                "font-size": ""
            },
            "subheading": {
                "font-family": "",
                "font-weight": "",
                "font-size": ""
            },
            "body": {
                "font-family": "",
                "font-weight": "",
                "font-size": "",
            },
        },
        "color-palette": {
            "primary": "",
            "secondary": "",
            "accent": "",
            "highlight": "",
        },
        "padding": "",
        "margin": "",
        "gutter": "",
        "layout": "",
    }

    prompt = f"""Generate style guide dictionaries based on the brand's identity: {project}. Make decisions based on the look and feel mentioned in the requirements. Like if the brand is luxurious, the style guide should be minimalistic and luxurious.
    or if the brand is modern, the style guide should be modern and minimalistic. same with colors
    Consider aspects such as Color Palette, Padding, Gutter, Layout Style, Design Style, and Typography for each style guide.
    output must be in python dictionary format like {output_format} and it must be very detailed. output must not have any descriptive text before the dictionary. The style guide must be based on professional design principles and professionally used font names and color codes.
    the output must have the root key as 'design' make sure the output will contain the values (with units applied)
"""

    response = createResponse(prompt)

    design = ast.literal_eval(response)['design']

    return design


def createResponse(prompt):

    # response = openai.Completion.create(
    #     # model="text-davinci-003",
    #     model="gpt-3.5-turbo-instruct",
    #     prompt=prompt,
    #     temperature=1,  # Adjust temperature for desired creativity
    #     max_tokens=3800,    # Adjust max_tokens based on desired response length
    #     top_p=1.0,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0
    # )

    # return response.choices[0].text

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1,
        max_tokens=10000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    print(response)
    return response.choices[0].message.content
