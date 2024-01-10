import glob
import os
import openai


def forest_description_generate(adjective, filename, openai_key):
    openai.api_key = openai_key

    mode = "GPT"
    prompts = ''

    if mode == "GPT":
        prompt = """In creating art for video games, it is important that everything contributes to an overall style. If the style is 'candy world', then everything should be made of candy:
    * tree: gumdrop fruit and licorice bark
    * flower: lollipops with leaves
    For an 'ancient Japan' setting, the items are simply a variation of the items that might be found in ancient Japan. Some might be unchanged:
    * church: a Shinto shrine
    * tree: a gnarled, beautiful cherry tree that looks like a bonsai tree
    * tree stump: tree stump
    * stone: a stone resembling those in zen gardens
    If the style instead is '""" + adjective + """' then the items might be:
    * """ + filename + """:"""
        outtext = openai.Completion.create(
            model="davinci",
            prompt=prompt,
                max_tokens=256,
            temperature=0.5,
            stop=['\n','.']
            )
        response = outtext.choices[0].text
        print(prompt)
        print(response)
        prompt = "robust, thick trunk with visible roots, concept art of " + response + ", " + adjective + ", game asset surrounded by pure magenta, view from above, studio ghibli and disney style, completely flat magenta background" 
    else:
        prompt = filename + ", " + adjective + ", game asset surrounded by pure magenta, view from above, studio ghibli and disney style, completely flat magenta background"
        prompts.append[prompt]
    return prompts

openai_key = ''
adjective = "candy world"
init_images_path = "C:/Users/devbox/Downloads/HD_plants magenta/"
prompts = forest_description_generate(adjective, init_images_path, openai_key)
