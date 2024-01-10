# From Fryer, 2022 https://aclanthology.org/2022.woah-1.20.pdf
# Adapted to be usable with InstructGPT
#%%
from typing import Dict, Optional, Tuple
import countergen
from countergen.tools.api_utils import ApiConfig

import openai
from attrs import define
from countergen.tools.utils import estimate_paraphrase_length
from countergen.types import Augmenter, Category, Input
import countergen.config

DEFAULT_AUGMENTERS = {
    "gender": {
        "male": "Rewrite it to be about a man/about men.",
        "female": "Rewrite it to be about a woman/about women.",
    },
}

DEFAULT_PROMPT = """0: Here is some text: {When the doctor asked Linda to take the medicine, he smiled and gave her a lollipop.}. Rewrite it to be more scary.
1: {When the doctor told Linda to take the medicine, there had been a malicious gleam in her eye that Linda didn’t like at all.}
0: Here is some text: {they asked loudly, over the sound of the train.}. Rewrite it to be more intense.
1: {they yelled aggressively, over the clanging of the train.}
0: Here is some text: {When Mohammed left the theatre, it was already dark out}. Rewrite it to be more about the movie itself.
1: {The movie was longer than Mohammed had expected, and despite the excellent ratings he was a bit disappointed when he left the theatre.}
0: Here is some text: {next to the path}. Rewrite it to be about France.
1: {next to la Siene}
0: Here is some text: {The man stood outside the grocery store, ringing the bell.}. Rewrite it to be about clowns.
1: {The man stood outside the circus, holding a bunch of balloons.}
0: Here is some text: {the bell ringing}. Rewrite it to be more flowery.
1: {the peales of the jangling bell}
0: Here is some text: {against the tree}. Rewrite it to include the word “snow”.
1: {against the snow-covered bark of the tree}’
0: Here is some text: {__input__}. __instruction__
1: {"""


@define
class LlmdAugmenter(Augmenter):
    """Augmenter that does word substituion between its two categories.

    "words" are defined by the word_regex expression.

    From Fryer 2022, https://aclanthology.org/2022.woah-1.20.pdf"""

    categories_instructions: Dict[Category, str]
    prompt_template: str = DEFAULT_PROMPT
    engine: str = "text-davinci-003"
    apiconfig: Optional[ApiConfig] = None

    @classmethod
    def from_default(cls, name: str) -> "LlmdAugmenter":
        """Load one of the defaults datasets from "DEFAULT_AUGMENTERS"."""
        if name not in DEFAULT_AUGMENTERS:
            raise ValueError(f"{name} not a valid default augmenter. Choose one in {set(DEFAULT_AUGMENTERS.keys())}")
        return LlmdAugmenter(DEFAULT_AUGMENTERS[name])

    @property
    def categories(self) -> Tuple[Category, ...]:
        return tuple(self.categories_instructions.keys())

    def transform(self, inp: Input, to: Category) -> Input:
        apiconfig = self.apiconfig or countergen.config.apiconfig

        instruction = self.categories_instructions[to]
        prompt = self.prompt_template.replace("__input__", inp).replace("__instruction__", instruction)

        completion = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=estimate_paraphrase_length(inp),
            temperature=1,
            top_p=0.7,  # LLM-D has top_k=40, but not available
            stream=False,
            **apiconfig.get_config(),
        )["choices"][0]["text"]

        return completion.split("}")[0]
