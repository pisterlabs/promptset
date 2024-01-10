# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 12-06-2022
# =============================================================================
"""
This module contains a wrapper for GPT-3-based style transfer. On initialization,
  it will instantiate a base generator agent and a renderer function. The renderer
  function will use GPT-3 API call to perform style transfer.
"""

from typing import Callable, Dict

import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import re

import json
import logging

from parlai.core.agents import Agent, create_agent, create_agent_from_shared


class Gpt3RenderAgent(Agent):
    """A wrapper for GPT3-based style transfer. On initialization, it will instantiate 
        a base generator agent and a renderer function. The renderer function will 
        use GPT3 API call to perform style transfer."""

    gpt3_config_key      = "gpt3"
    generator_config_key = "generator"
    generator_shared_key = "generator"

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        
        logging.info("CONFIG:\n" + json.dumps(opt, indent=True))

        self.id = __class__.__name__
        self.renderer_config  = opt[__class__.gpt3_config_key]
        self.generator_config = opt[__class__.generator_config_key]
        
        self.renderer = self.create_renderer(self.renderer_config)
        
        # If we want to save the history of rendered responses
        #   or the base response from the generator. The idea 
        #   is to set whether the generator is aware of the style transfer
        self.is_render_aware = opt.get("is_render_aware", False)
        
        # Create generator from config file (first-time instantiation)
        if shared is None:
            logging.info("CREATED FROM PROTOTYPE")
            self.generator = create_agent(self.generator_config)
        
        # Create generator from shared state
        else:
            logging.info("CREATED FROM COPY")
            self.generator = create_agent_from_shared(shared[__class__.generator_shared_key])

    def share(self) -> Dict:
        """Copy response function <self.resp_fn>"""

        logging.info("MODEL COPIED")

        shared = super().share()
        shared[__class__.generator_shared_key] = self.generator.share()
        return shared
    
    def render_aware_act(self, observation) -> Dict:
        """Render the basic generated response and make the generator 
            aware of the style-transferred response by calling `generator.self_observe`
            to update the history of the generator."""
            
        message  = self.generator.batch_act([observation])[0]
        response = self.renderer(message)
        
        message.force_set("text", response["text"])
        self.generator.self_observe(message)
        
        logging.info(f"HISTORY: {self.generator.history.get_history_str()}")
        return response

    def act(self) -> Dict:
        """Forward the observation to the wrapped generator, then call the 
            GPT-3 renderer to perform style transfer."""

        observation = self.observation
        if observation is None:
            return {"id":            self.getID(), 
                    "gpt3_prompt":   "",
                    "text":          "",
                    "base_response": ""}

        # Call the wrapped generator
        self.generator.observe(observation)
        if (self.is_render_aware 
            and hasattr(self.generator, "batch_act")
            and hasattr(self.generator, "history")
            and hasattr(self.generator, "self_observe")):
            logging.info("RENDER AWARE ACT")
            return self.render_aware_act(self.generator.observation)
        
        logging.info("NORMAL ACT")
        return self.renderer(self.generator.act())

    def reset(self):
        """Reset the agent, clearing its observation."""

        self.observation = None
        self.generator.reset()

    def create_renderer(self, opt: Dict) -> Callable[[Dict], Dict]:
        """Create a GPT-3 renderer function based on the options passed in."""

        import openai

        def gpt_completion(s: str) -> str:
            """GPT-3 completion function. Takes a raw bot response and returns a new response.
                TODO: Should allow prompt to be passed in as well, or read from a config file."""

            prompt_table = {
                "empathetic": (
                    "Here is an empathetic sentence: {I thought it was terribly depressing what these children had to go through.}"
                    "\nHere is another empathetic sentence: {I feel so sad for everyone especially the old and sickly seems as they are in the worst position.}"
                    "\nHere is another empathetic sentence: {You always want better for your kid and I think giving them up to someone else would be the best option even though it hurts. So sad. My heart just breaks for these woman.}"
                    "\nHere is some text: {" + s + ".} Here is a rewrite of the text, which is more empathetic: {"
                ),
                "self-disclosure": (
                    "Here is a sentence of high self-disclosure: {I was always scared as a catholic to go to church as a kid and would always talk my mom out of going which probably turned out to save me in the long run.}"
                    "\nHere is another sentence of high self-disclosure: {I love that, personally my father went outside to smoke and never smoked with us in the car but my friends parents did and I would feel so sick after being in there car no child should suffer through it.}"
                    "\nHere is another sentence of high self-disclosure: {My father was in the Navy and I have two sisters in the Army.}"
                    "\nHere is some text: {" + s + ".} Here is a rewrite of the text, which is of higher self-disclosure: {"
                )
            }
            prompt = prompt_table.get(opt["style"])
            if prompt is None:
                raise ValueError(f"Style {opt['style']} not supported. Choose from {list(prompt_table.keys())}")

            # For debugging, return the prompt without calling GPT-3
            if self.renderer_config.get("is_dry_run", False):
                return f"[ echo render ] :: {s}", prompt

            openai.api_key = OPENAI_API_KEY
            completion = openai.Completion.create(prompt=prompt, **opt["generation_config"])
            return completion["choices"][0]["text"].strip(), prompt
        
        def renderer(message: Dict) -> Dict:
            """Take a raw bot response and return a rendered response using 
                the GPT-3 style-transfer defined above."""
                
            def clean_token(token: str) -> str:
                token = re.sub(r"_POTENTIALLY_UNSAFE__", "", token, flags=re.IGNORECASE)
                return token.strip()
            
            base_text = clean_token(message["text"])
            text, prompt = gpt_completion(base_text)
            response = {"id":            self.getID(),
                        "gpt3_prompt":   prompt,
                        "base_response": base_text,
                        "text":          text}
            return response

        return renderer
