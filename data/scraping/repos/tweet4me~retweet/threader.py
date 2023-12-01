import os
import openai

from . import const 
from . import utils
from . import templates


def rethread(text):
    template = templates.get_other_template("twitter-threader-v0")
    prompt = template.substitute(text=text)
    prompt_length = len(prompt)
    output_size = 4000 - prompt_length - 1
    openai.organization = const.GPT3_ORG_ID
    openai.api_key = utils.get_gpt3_secret()
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=output_size,
        temperature=0.9,
        best_of=3
    )
    return response["choices"][0]["text"].strip()

if __name__ == "__main__":
    print(
        rethread(
            """How to evaluate manager performance? Value/impact/revenue? Yes, probably, but here is a thing, quite often value/revenue/impact is generated by the reports, not by manager. So how to be sure that manager actually valuable? I always say that for IC ultimate measurement is the impact but for the manager ultimate ruler is the amount of HC. Impact generate by the team could be done by ICs and manager might not only be un-helpful but even aregssively be preventing. But if with each given new HC impact of the org is grwoing this means that manager actually capable of converting given HC into new revenue/impact streams. At the same time if there is no HC given this quite often means that manager have not proved to the leadership that by giving more HC she/he can deliver more impact, as simple as that. So, if you would be to put the most generic KPI for evaluating manager performance, it would be: how much HC manager is getting.

Since this is twitter and people do love to overindex on small things for the sake of doing toxic commenting I do want to write usual disclaimer: yes there are many cases where it is out of managers hand to have more HC, there are cases where team should not only have the same size but even reduce in size and this will not be saying anything about manager. """
            )
    )