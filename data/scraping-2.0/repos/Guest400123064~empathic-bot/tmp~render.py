from typing import Callable

import json
import logging

import openai
from parlai.core.agents import Agent, create_agent, create_agent_from_shared


class RenderAgent(Agent):
    """Dummy agent that simply echos everything it observed. On initialization,
        it also print out all options it receives."""

    def __init__(self, opt, shared=None):
        """Init agent and echo options"""

        super().__init__(opt)
        logging.info('CONFIG:\n' + json.dumps(opt, indent=True))

        self.id = 'RenderBot'
        self.renderer = self._create_renderer(opt['Renderer'])
        if shared is None:
            logging.info('CREATED FROM PROTOTYPE')
            self.generator = create_agent(opt['Generator'])
        else:
            logging.info('CREATED FROM COPY')
            self.generator = create_agent_from_shared(shared['generator'])

    def share(self) -> dict:
        """Copy response function <self.resp_fn>"""

        logging.info('MODEL COPIED')

        shared = super().share()
        shared['generator'] = self.generator.share()
        return shared

    def act(self):
        """Simply copy the input messages"""

        obs = self.observation
        if obs is None:
            return {'text': 'Nothing to reply to yet'}

        # Call generator first
        # logging.info(self.generator)
        self.generator.observe(obs)
        raw_resp = self.generator.act().get('text')
        fin_resp, prompt = raw_resp, ''  # No openai render for now self.renderer(raw_resp)
        
        logging.info(f'ACT:\n' +
                     f'\t[ observed ] :: {obs.get("text", "[EMPTY]")}\n' +
                     f'\t[ raw_resp ] :: {raw_resp}\n' +
                     f'\t[ rendered ] :: {fin_resp}\n' +
                     f'\t[  prompt  ] :: {prompt}')
        return {'id': self.getID(),
                'text': fin_resp}

    def reset(self):
        """
        Reset the agent, clearing its observation.
        """

        self.observation = None
        self.generator.reset()

    def _create_renderer(self, opt) -> Callable[[str], str]:

        def gpt_completion(s: str) -> str:

            # Style transfer prompt table
            prompt_table = {
                "extroverted": """
                    - Here is some text: {What a nice day!}. Here is a rewrite of the text, which is more extroverted: {What a great day it is! Everything just feels so perfect!}
                    - Here is some text: {How about going out and enjoying the sunshine on the grass?}. Here is a rewrite of the text, which is more extroverted: {Why not go outside and enjoy the sunshine on the grass? It's a great way to relax and get some fresh air.}
                    - Here is some text: {Jenny, what's wrong with you?}. Here is a rewrite of the text, which is more extroverted: {What's wrong, Jenny? You seem upset.}
                    - Here is some text: {""" + s + "}. Here is a rewrite of the text, which is more extroverted: {",
                "agreeable": """
                    - Here is some text: {What a nice day!}. Here is a rewrite of the text, which is more agreeable: {What a wonderful day it is! I'm so glad that I get to spend it with you.}
                    - Here is some text: {How about going out and enjoying the sunshine on the grass?}. Here is a rewrite of the text, which is more agreeable: {Why not go outside and enjoy the sunshine on the grass?}
                    - Here is some text: {Jenny, what's wrong with you?}. Here is a rewrite of the text, which is more agreeable: {Jenny, what happened to you?}
                    - Here is some text: {""" + s + "}. Here is a rewrite of the text, which is more agreeable: {",
                "empathetic": """
                    - Here is some text: {What a nice day!}. Here is a rewrite of the text, which is more empathetic: {What a beautiful day it is!}
                    - Here is some text: {How about going out and enjoying the sunshine on the grass?}. Here is a rewrite of the text, which is more empathetic: {What about spending some time outside in the sunshine? It sounds like it would be really enjoyable to lie on the grass and soak up some vitamin D.}
                    - Here is some text: {Jenny, what's wrong with you?}. Here is a rewrite of the text, which is more empathetic: {Jenny, I can see that you're upset. Can you tell me what's wrong? I want to help you if I can.}
                    - Here is some text: {""" + s + "}. Here is a rewrite of the text, which is more empathetic: {"
            }
            prompt = prompt_table[opt['style']]

            # OpenAI API call
            openai.api_key = opt['token']
            completion = openai.Completion.create(prompt=prompt, **opt['generation_config'])
            return completion['choices'][0]['text'].strip('}'), prompt
 
        return gpt_completion
    
# %%
t = """Here is a sentence of high self-disclosure: {I was always scared as a catholic to go to church as a kid and would always talk my mom out of going which probably turned out to save me in the long run.}
Here is another sentence of high self-disclosure: {I love that, personally my father went outside to smoke and never smoked with us in the car but my friends parents did and I would feel so sick after being in there car no child should suffer through it.}
Here is another sentence of high self-disclosure: {My father was in the Navy and I have two sisters in the Army.}
Here is some text: {"""
print(t)

# %%
