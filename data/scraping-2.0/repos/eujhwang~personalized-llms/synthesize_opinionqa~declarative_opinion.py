from typing import Dict

from gptinference.base_prompt import Prompt
from gptinference.openai_wrapper import OpenAIWrapper
from gptinference.utils import write_json, read_jsonl_or_json
from tqdm import tqdm


class DeclarativeOpinionGPT(Prompt):
    """
    Creates "declarative opinion" using "question", "answer", "choices".

    Question: How often, if ever, did you use air guns, such as paintball, BB or pellet guns when you were growing up?
    Answer: Never.
    Declarative opinion: I never used air guns such as paintball, BB or pellet guns when I was growing up.
    """
    def __init__(self, engine: str, openai_wrapper: OpenAIWrapper):
        super().__init__()
        self.openai_wrapper = openai_wrapper
        self.engine = engine

    def make_query(self, question: str, answer: str) -> str:
        """ Prompt to convert question, answer to a declarative sentence.

        Question: How often, if ever, did you use air guns, such as paintball, BB or pellet guns when you were growing up?
        Answer: Never.

        Declarative form: I never used air guns such as paintball, BB or pellet guns when I was growing up.
        """
        if not question or not answer:
            return ""

        return f"""Convert the following question and answer into a declarative sentence.
    
Question: {question}
Answer: {answer}

Declarative form:
"""

    def __call__(self, question: str, answer: str) -> str:
        generation_query = self.make_query(question=question, answer=answer)
        generated_decl_sent = self.openai_wrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=30,
            stop_token="###",
            temperature=0.0
        )
        return generated_decl_sent.strip()  # gpt3 turbo adds newline in the beginning so strip it.

class DeclarativeOpinionSaver:
    """
    Adds "declarative_opinion" to implicit persona json.
    """

    def add_implicit_persona(self, decl: DeclarativeOpinionGPT, user_responses_jsons: Dict):
        """ For every implicit persona of every user, add declarative_opinion field.
        "implicit_persona": [
        {
           "qid": "GUNKILLF2_W26",
           "question": "Thinking about people who commit suicide using a gun, which comes closer to your view, even if neither is exactly right?",
           "choices": [
               "They would find a way to do it whether they had access to a gun or not",
               "They would be less likely to do it if they didn't have access to a gun",
               "Refused"
               ],
           "answer": "They would be less likely to do it if they didn't have access to a gun",
           "declarative_opinion": "xxxx",
           "subtopic_cg": [
               "crime/security"
               ]
        },

        ...

        ]
        """
        for user_response_json in tqdm(user_responses_jsons, desc="processing user response #"):
            for persona_qa in user_response_json["implicit_persona"]:
                persona_qa["declarative_opinion"] = decl(question=persona_qa["question"], answer=persona_qa["answer"])

        return user_responses_jsons


if __name__ == '__main__':
    in_path = "data/opinionqa/sampled_user_responses.json"
    out_path = "data/opinionqa/sampled_user_responses_decl.json"
    cache_path="data/cache/gpt_cache.jsonl"
    decl = DeclarativeOpinionGPT(engine="gpt-3.5-turbo", openai_wrapper=OpenAIWrapper(cache_path=cache_path))
    enhanced_json = DeclarativeOpinionSaver().add_implicit_persona(decl=decl, user_responses_jsons=read_jsonl_or_json(in_path))
    write_json(outpath=out_path, json_data=enhanced_json)