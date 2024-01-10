# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from typing import Any, List
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
        return prompt
    else:
        prompt = filename + ", " + adjective + ", game asset surrounded by pure magenta, view from above, studio ghibli and disney style, completely flat magenta background"
        prompts.append[prompt]
    return prompts


class Predictor(BasePredictor):
    def predict(
        self,
        adjective: str = Input(description="Object", default="candy world"),
        filename: str = Input(description="rainbow forest", default="rainbow forest"),
    ) -> Any:
        """Run a single prediction on the model"""

        try:
            key = 'sk-'
            res = dict()
            print(adjective)
            print(filename)
            res['description'] = forest_description_generate(adjective,filename,key)
            return res
        except Exception as e:
            return f"Error: {e}"
