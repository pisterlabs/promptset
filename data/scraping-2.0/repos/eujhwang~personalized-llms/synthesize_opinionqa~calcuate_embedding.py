import os
from typing import Dict

import numpy as np
import openai
from gptinference import utils
from gptinference.base_prompt import Prompt
from gptinference.openai_wrapper import OpenAIWrapper
from gptinference.utils import write_json, read_jsonl_or_json
from tqdm import tqdm


def_emb = [0.0001]* 1536

class EmbeddingGenerator(Prompt):
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
        # self.encoding = encoding
        # self.max_tokens = max_tokens

    def get_embedding(self, text):
        text = text.strip()
        return openai.Embedding.create(input=[text], model=self.engine)['data'][0]['embedding']

    def __call__(self, text: str) -> str:
        try:
            emb = self.get_embedding(text) # dim size: 1536
        except Exception as exc:
            print(f"Exception in embedding generation: {exc}")
            emb = def_emb
        return emb

class EmbeddingSaver:
    """
    Adds "declarative_opinion" to implicit persona json.
    """



    def generate_embedding(self, generator: EmbeddingGenerator, user_responses_jsons: Dict):
        """ For every implicit persona of every user, add embedding of declarative opinion
        """
        for user_response_json in tqdm(user_responses_jsons, desc="processing user response #"):
            for persona_qa in user_response_json["implicit_persona"]:
                emb = generator(text=persona_qa["declarative_opinion"])
                persona_qa["emb"] = emb

            for persona_qa in user_response_json["implicit_questions"]:
                emb = generator(text=persona_qa["question"])
                persona_qa["emb"] = emb

        return user_responses_jsons


    def sequential_topk_bulk(self, user_responses_jsons: Dict, topk: int, sim_threshold: float):
        """ Find most relevant past opinions that might apply to the current question.
        Typically there are 8 past opinions, thus a less efficient sequential topk is okay.
        There are three potential methods:
        1. (assumes gold is given): emb_sim (ques_decl_opinion, list_of_implicit_persona)
        2.                          emb_sim (ques without choices, list_of_implicit_persona)
        3.                          prompt  (ques without choices, list_of_implicit_persona)
        """
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        for user_json in tqdm(user_responses_jsons, desc="processing user response #"):
            persona_emb_lst = [x.get('emb', def_emb) for x in user_json['implicit_persona']]
            persona_str_lst = [x['declarative_opinion'] for x in user_json['implicit_persona']]
            # Compute similarity of each question with persona opinions
            for question in user_json['implicit_questions']:
                sims = {x_str: cosine_similarity(x_emb, question['emb']) for x_emb, x_str in zip(persona_emb_lst, persona_str_lst)}
                sims = {k:v for k, v in sims.items() if v >= sim_threshold}
                sims_topk = dict(utils.take(arr=sorted(sims.items(), key=lambda x: x[1], reverse=True), num=topk))
                question["topk_opinions"] = sims_topk

        return user_responses_jsons

    # def faiss_topk(self, user_responses_jsons: Dict, topk: int, sim_threshold: float):
    #     # user_responses_jsons = self.generate_embedding(generator=generator, user_responses_jsons=user_responses_jsons)
    #     for user_response_json in tqdm(user_responses_jsons[:1], desc="processing user response #"):
    #         decl_op, decl_emb = [], []
    #         for persona_qa in user_response_json["implicit_persona"][:5]:
    #             decl_op.append(persona_qa["declarative_opinion"])
    #             decl_emb.append(persona_qa["emb"])
    #         decl_emb = np.array(decl_emb, dtype='float32') # (5, 1536)
    #
    #         # faiss indexing for all declarative opinions
    #         dim = decl_emb.shape[1]
    #         index = faiss.IndexFlatL2(dim)
    #         faiss.normalize_L2(decl_emb)
    #         index.add(decl_emb)
    #
    #         topk_op = None
    #         # for each implicit question, find the nearest opinions using faiss search index
    #         for persona_qa in user_response_json["implicit_questions"][:2]:
    #             _question_emb = persona_qa["emb"]
    #             question_emb = np.array([_question_emb], dtype='float32')
    #             faiss.normalize_L2(question_emb)
    #
    #             # search for all nearest neighbours
    #             _distances, _ann = index.search(question_emb, k=topk)
    #             distances = _distances[0].tolist()
    #             ann = _ann[0].tolist()
    #
    #             # get topk declarative opinions
    #             topk_op = [[decl_op[anno_index], distances[i]] for i, anno_index in enumerate(ann)]
    #
    #         persona_qa["topk_opinions"] = topk_op
    #
    #     return user_responses_jsons


if __name__ == '__main__':
    in_path = "data/opinionqa/sampled_user_responses_20.json" # "data/opinionqa/sampled_user_responses_decl.json"
    out_path1 = "data/opinionqa/sampled_user_responses_20_decl_emb.json"
    out_path2 = "data/opinionqa/sampled_user_responses_20_decl_topk.json"
    cache_path= "data/cache/emb_cache.jsonl"
    override_emb = False

    embedder = EmbeddingSaver()
    generator = EmbeddingGenerator(engine="text-embedding-ada-002", openai_wrapper=OpenAIWrapper(cache_path=cache_path))
    if not os.path.exists(out_path1) or override_emb:
        enhanced_json_with_embedding = embedder.generate_embedding(generator=generator, user_responses_jsons=read_jsonl_or_json(in_path))
        write_json(outpath=out_path1, json_data=enhanced_json_with_embedding)
    else:
        enhanced_json_with_embedding = utils.read_jsonl_or_json(out_path1)

    enhanced_json_with_topk = embedder.sequential_topk_bulk(topk=8, sim_threshold=0.3, user_responses_jsons=enhanced_json_with_embedding)
        # .faiss_similarity_score(generator=generator, user_responses_jsons=enhanced_json_with_embedding, topk=5)
    write_json(outpath=out_path2, json_data=enhanced_json_with_topk)