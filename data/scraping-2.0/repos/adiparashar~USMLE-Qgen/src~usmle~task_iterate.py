import re
from typing import Dict, List
from src.utils import Prompt
import json
from src.prompt_lib.prompt_lib.backends import openai_api

header = """Clinical Note: {clinical_note}

Topic: {topic}

Keypoint: {keypoint}
"""

example_template = """Context: {context}

Question: {question}

Correct answer: {correct_answer}

Distractor Options: {distractor_options}

Feedback for the above components:

Context feedback: {context_feedback}

Context score: {context_score}

Question feedback: {question_feedback}

Question score: {question_score}

Correct answer feedback: {correct_answer_feedback}

Correct answer score: {correct_answer_score}

Distractor options feedback: {distractor_option_feedback}

Distractor options score: {distractor_option_score}"""

prompt_template = """Context: {context}

Question: {question}

Attempted answer: {attempted_answer}

Reasoning: {reasoning}

Correct answer: {correct_answer}

Distractor Options: {distractor_options}

Feedback on the generated content with respect to various rubrics.

Context feedback: {context_feedback}

Context score: {context_score}

Question feedback: {question_feedback}

Question score: {question_score}

Correct answer feedback: {correct_answer_feedback}

Correct answer score: {correct_answer_score}

Distractor options feedback: {distractor_option_feedback}

Distractor options score: {distractor_option_score}

Reasoning feedback: {reasoning_feedback}

Reasoning score: {reasoning_score}"""
instr = """
Improve the context,question, correct answer and distractor options using each previous components' feedback and the reasoning feedback.
Generate a context, question, correct answer and distractor options that can achieve high scores on all the above feedback rubrics, given the clinical note, keypoint and topic. Do not generate the feedback for any of the component.:

"""
example_instr = """
Improved version of the above components using their respective feedbacks:
"""
class UsmleQgenTaskIterate(Prompt):
    def __init__(self, engine: str, prompt_examples: str) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.count = 0
        self.prompt = self.make_prompt(prompt_examples=prompt_examples)

    def make_prompt(self, prompt_examples: str) -> str:
        import pandas as pd
        with open(prompt_examples) as f:
            dictt = json.load(f)
        examples_df = pd.DataFrame.from_dict(dictt, orient="index").T
        prompt = []
        for _, row in examples_df.iterrows():
            prompt.append(
                self.make_one_iterate_example(
                    clinical_note=row["clinical_note"],topic=row["topic"], keypoint=row["keypoint"],content_to_fb=row["content_to_feedback"]
                )
            )
        return self.inter_example_sep.join(prompt) + self.inter_example_sep

    def make_one_iterate_example(self,clinical_note: str, keypoint:str, topic:str, content_to_fb: List[Dict]):
        """Given a list of examples that are incrementally improving, return a new example."""
        single_example = []
        print(content_to_fb[0].keys())
        if "reasoning_feedback" not in content_to_fb[0].keys():
            for example in content_to_fb:
                example_text = example_template.format(
                    context=example["context"],
                    question=example["question"],
                    correct_answer=example["correct_answer"],
                    distractor_options = example["distractor_options"],
                    context_feedback=self.setup_feedback(example["context_feedback"]),
                    context_score = example["context_score"],
                    question_feedback=self.setup_feedback(example["question_feedback"]),
                    question_score = example["question_score"],
                    correct_answer_feedback=self.setup_feedback(example["correct_answer_feedback"]),
                    correct_answer_score = example["correct_answer_score"],
                    distractor_option_feedback = self.setup_feedback(example["distractor_option_feedback"]),
                    distractor_option_score = example["distractor_option_score"]
                )
                single_example.append(example_text)
            return header.format(clinical_note=clinical_note,topic=topic,keypoint=keypoint) + example_instr.join(single_example)
        else:
            example = content_to_fb[0]
            example_text = prompt_template.format(
                context=example["context"],
                question=example["question"],
                attempted_answer = example["attempted_answer"],
                reasoning = example["reasoning"],
                correct_answer=example["correct_answer"],
                distractor_options = example["distractor_options"],
                context_feedback=self.setup_feedback(example["context_feedback"]),
                context_score = example["context_score"],
                question_feedback=self.setup_feedback(example["question_feedback"]),
                question_score = example["question_score"],
                correct_answer_feedback=self.setup_feedback(example["correct_answer_feedback"]),
                correct_answer_score = example["correct_answer_score"],
                distractor_option_feedback = self.setup_feedback(example["distractor_option_feedback"]),
                distractor_option_score = example["distractor_option_score"],
                reasoning_feedback = example['reasoning_feedback'],
                reasoning_score = example['reasoning_score']
            )
            # 
            return header.format(clinical_note=clinical_note,topic=topic,keypoint=keypoint) + example_text +  instr
    def setup_feedback(self,feedback):
        fb = ''
        if isinstance(feedback, str):
            return feedback
        #feedback = json.loads(feedback)
        #print(f"FB: {feedback}")
        for key in feedback.keys():
            fb += '\n' + key + ':' + feedback[key] 
        return fb
    def make_query(self, clinical_note: str, keypoint:str, topic:str,
        content_to_fb: List[Dict],) -> str:
        query_example = self.make_one_iterate_example(clinical_note, keypoint, topic, content_to_fb)
        return f"{self.prompt}{self.question_prefix}{query_example}{self.intra_example_sep}{self.answer_prefix}" + instr
        # return super().make_query(prompt, question)

    def __call__(
        self,
        clinical_note: str, keypoint:str, topic:str,
        content_to_fb: List[Dict]
    ) -> str:
    
        transfer_query = self.make_query(clinical_note, keypoint, topic, content_to_fb)
        self.count += 1
        print(" ========= ITERATE PROMPT =========")
        print(transfer_query)
        print(" ========= ITERATE PROMPT ENDS =========")
        #print(transfer_query)
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=transfer_query,
            engine=self.engine,
            max_tokens=2048,
            stop_token=self.inter_example_sep,
            temperature=0.7,
        )
        response = openai_api.OpenaiAPIWrapper.get_first_response(output)

        print("######")
        print()
        print("------")
        print(response)
        print("------")

        # response = 'Context: ' + re.search("Context: (.*)", response).group(1).strip()
        # print(response)
        context = re.search(r'Context:(.*?)(?=\n\n)', response,flags=re.IGNORECASE).group(1)
        question = re.search(r'Question:(.*?)(?=\n\n)', response,flags=re.IGNORECASE).group(1)
        correct_answer = re.search(r'correct answer:(.*?)(?=\n\n)', response.lower(),flags=re.IGNORECASE).group(1)
        distractor_options = re.search(r'distractor options:(.*)', response.lower(),flags=re.S).group(1)
        print(distractor_options)
        return context.strip(),question.strip(),correct_answer.strip(),distractor_options.strip()

    def make_input(
        self,
        title: str,
        acronyms_to_scores: Dict[str, str],
    ) -> str:
        input_txt = ""
        for acronym, scores in acronyms_to_scores.items():
            input_txt += self._make_input(
                title=title,
                acronym=acronym,
                scores=scores,
            )
        return input_txt


if __name__ == "__main__":
    obj = UsmleQgenTaskIterate(
        prompt_examples="data/prompt/usmle/iterate.jsonl", engine="whatever"
    )
    #print(obj.prompt)
    # print(obj.make_query(concepts=["a", "b"], sent_to_fb=[{"sentence": "a", "feedback": "a"}, {"sentence": "b", "feedback": "d"}]))
