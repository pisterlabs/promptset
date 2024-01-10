# Symbolic Stance Detection

import requests
from langchain.chains.base import Chain


import json
import sys, os
from typing import Dict, List


class StanceGetter(Chain):
    url : str = f"{os.getenv('STANCE_SERVER')}/stance/sentence"

    def get_stance(self, text, domain):
        response = requests.post(self.url,
                         json.dumps({"text": text, "domain": domain}),
                         headers={"Content-Type": "application/json"}).json()
        print(response, file=sys.stderr)
        return response

    def believes(self, strength):
        strength = abs(float(strength))
        if strength > 2.5:
            return "very strongly believes"
        elif strength > 1.5:
            return "strongly believes"
        elif strength > 0.5:
            return "believes"

    def believes_polarity(self, strength):
        strength = float(strength)
        if strength > 0:
            return "is true"
        elif strength < 0:
            return "is false"
        else:
            return "is undetermined"

    def sentiment(self, strength):
        strength = float(strength)
        if strength > 0.5:
            return "positive"
        elif strength < -0.5:
            return "negative"
        else:
            return "neutral"

    def stance_to_text(self, x):
        predicate = f"{x['belief_type']}({x['belief_trigger']}, {x['belief_content']})"
        belief = f"{self.believes(x['belief_strength'])} that the predicate {predicate} {self.believes_polarity(x['belief_strength'])}"
        sent = f"feels {self.sentiment(x['sentiment_strength'])} sentiment towards the idea that {predicate}."
        return [belief, sent]

    @property
    def input_keys(self) -> List[str]:
        return ['text', 'domain']

    @property
    def output_keys(self) -> List[str]:
        return ['stances', 'representations']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        stance = self.get_stance(inputs['text'], inputs['domain'])
        text = "The author of this statement:\n- " + "\n- ".join(sum([self.stance_to_text(y) for y in stance["stances"]], []))
        return {'stances': text, "representations": [x["stance_rep"] for x in stance["stances"]]}

StanceGetterChain = StanceGetter()
