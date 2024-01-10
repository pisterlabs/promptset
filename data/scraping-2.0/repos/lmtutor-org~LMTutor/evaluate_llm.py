from model.llm_langchain_tutor import LLMLangChainTutor
import re
import json
import pandas as pd
from tqdm import tqdm
import os
import openai


class LMTutorEvaluator:
    def __init__(self, evaluator='chatgpt', api_key=None) -> None:
        self.evaluator = evaluator
        if self.evaluator in ['chatgpt']:
            assert api_key is not None, "api_key is none!"
            openai.api_key = api_key
            self.model = "gpt-3.5-turbo"

        self.system_prompt = "You are a helpful assistant."

    def evaluate(self, **kwargs):
        if self.evaluator == 'chatgpt':
            return self.evaluate_by_chatgpt(**kwargs)

    def evaluate_by_chatgpt(self, question, answer1, answer2):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": "Consider the following two answers in response to the question:\n{question} \n\n Answer-1:\n{answer1} \n\n Answer-2:\n{answer2} \n\n Which answer is better-written and consistent with the fact? Please answer with one of the following options.\n\n Options:\n(A) Answer-1\n(B) Answer-2\n(C) Both answers are equally well-written and consistent with the fact.\n\nAnswer: I will choose Option".format(
                        question=question, answer1=answer1, answer2=answer2),
                },
            ],
            temperature=0,
            max_tokens=10,
        )

        return response


class AnswerGenerator:
    def __init__(self) -> None:
        pass

    def load_questions(self, question_path):
        self.questions = []
        ### txt files
        if question_path.endswith('.txt'):
            with open(question_path, "r") as f:
                for each in f:
                    self.questions.append(re.sub(r"^\d+\.\s*", "", each))
        elif question_path.endswith('.csv'):
            q_file = pd.read_csv(question_path)
            for _, each in q_file.iterrows():
                self.questions.append(each['questions'])

    def save_answers(self, save_path):
        with open(save_path, "w") as f:
            for each in self.answers:
                json.dump(each, f)
                f.write("\n")

    def similarity_search_statistics(self):
        raise NotImplementedError

    def generate_answers(self):
        raise NotImplementedError


class LMTutorAnswerGenerator(AnswerGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.lmtutor = LLMLangChainTutor(
            embedding="instruct_embedding",
            llm="hf_lmsys/vicuna-7b-v1.3",
            embed_device="cuda:6",
            llm_device="cuda:7",
        )
        self.lmtutor.load_vector_store("/home/yuheng/LMTutor/data/DSC-291-vector-lec")
        self.lmtutor.initialize_hf_llm()

    def generate_answers(self):  # based on the questions
        self.answers = []
        for q in tqdm(self.questions, desc="answering questions"):
            self.lmtutor.first_conversation = True
            self.lmtutor.memory.clear()

            answer = self.generate(q)
            self.answers.append({"question": q, "answer": answer})

    def generate(self, question):
        return self.lmtutor.conversational_qa(question)

    def similarity_search_statistics(self):
        result = []
        for q in tqdm(self.questions, desc="processing questions"):
            result.append([each[1] for each in self.lmtutor.similarity_search_thres(query=q)])

        result = pd.DataFrame(result)

        return result


class GPTAnswerGenerator(AnswerGenerator):
    def __init__(self, api_key, gpt_model: str) -> None:
        super().__init__()

        openai.api_key = api_key
        self.model = gpt_model
        self.system_prompt = "You are a teaching assistant for the Advanced Data Mining course. You are asked to answer questions from students."

    def generate_answers(self):
        self.answers = []
        for q in tqdm(self.questions, desc="answering questions"):
            answer = self.generate(q)
            self.answers.append({"question": q, "answer": answer})

    def generate(self, question):
        if self.model == "gpt-3.5-turbo":
            for i in range(3):
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {
                                "role": "user",
                                "content": question,
                            },
                        ],
                        temperature=0,
                        max_tokens=512,
                    )

                    response = response['choices'][0]['message']['content']
                    break
                except:
                    print("retry openai api")

        elif self.model in ["text-davinci-003", 'gpt-4']:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt="Answer the following question:\n\n" + question,
                max_tokens=512,
                temperature=0
            )

            response = response['choices'][0]['text']

        else:
            raise NameError("incorrect GPT model name.")

        return response


if __name__ == "__main__":
    lmtutor_eval = LMTutorAnswerGenerator()
    lmtutor_eval.load_questions("data/DSC-291-questions-2/question.txt")
    res = lmtutor_eval.similarity_search_statistics()

    print(res)
    # lmtutor_eval.generate_answers()
    # lmtutor_eval.save_answers(
    #     "data/DSC-291-questions-2/vicuna_1_3_instruct_embed_answers.txt"
    # )

    # model_name = "gpt-3.5-turbo" # text-davinci-003, gpt-3.5-turbo, gpt-4
    # gpt = GPTAnswerGenerator(api_key="", gpt_model=model_name)
    # gpt.load_questions("data/DSC-291-questions-3/questions.csv")
    # gpt.generate_answers()
    # gpt.save_answers(
    #     f"data/DSC-291-questions-3/{model_name}_answers.txt"
    # )
