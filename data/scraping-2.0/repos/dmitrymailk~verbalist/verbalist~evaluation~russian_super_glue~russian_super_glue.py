from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json
from pathlib import Path
from functools import partial
from datasets import load_dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from verbalist.generation.generation_utils import VerbalistConversation, generate
import openai


class RussianSuperGluePrompts:
    def lidirus_prompt(self, item):
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        prompt = f"""Текст: "{sentence1}"\nИспользуя текст, можно ли сказать, что утверждение "{sentence2}" корректно относительно ситуации из текста? Ответь только "да" или "нет"."""
        return prompt

    def rcb_prompt(self, item):
        sentence1 = item["premise"]
        sentence2 = item["hypothesis"]
        prompt = f"""premise: {sentence1}\nhypothesis: {sentence2}\nКак связаны между собой premise и hypothesis? Это contradiction, entailment или neutral? Ответь одним словом."""
        return prompt

    def parus_prompt(self, item):
        cause = "следствием " if item["question"] == "effect" else "причиной"
        prompt = f"Текст: {item['premise']}\nвыбор 1: {item['choice1']}\nвыбор 2: {item['choice2']}\nВыбери вариант который послужил {cause} для поля 'Текст'. В ответе напиши 'выбор 1' или 'выбор 2'."
        return prompt

    def muserc_prompt(self, item):
        prompt = f"""Текст: {item['paragraph']}\nвопрос: {item['question']}\nЯвляется ли "{item['answer']}" правильным ответом на этот вопрос? Основываясь на только тексте, ответь "правильно" или "неправильно" """
        return prompt

    def rucos_prompt(self, item):
        word_list = ", ".join([elem.strip() for elem in item["entities"]])
        text = item["passage"]
        prompt = f"""\nТекст: {text}\nЗапрос: {item['query']}\nСписок слов: {word_list}\nСогласно тексту, замени @placeholder запросе на наиболее подходящее слово из списка.\nВ качестве ответа верни только одно слово."""
        return prompt

    def terra_prompt(self, item):
        prompt = f"""Контекст: {item['premise']}\nВывод: {item['hypothesis']}\nЯвляется ли вывод правильным исходя из контекста? Думай шаг за шагом. В ответе напиши только "правильный" или "неправильный" """
        return prompt

    def russe_prompt(self, item):
        prompt = f"""Предложение 1: {item['sentence1']}\nПредложение 2: {item['sentence2']}\nЯвляется ли слово "{item['word']}" одинаковым по значению и смыслу в этих двух предложениях. В ответе напиши 'да' или 'нет'."""
        return prompt

    def rwsd_prompt(self, item):
        prompt = f"""Текст: {item['text']}\nОтвечай согласно тексту. Связаны ли "{item['span1_text']}" с "{item['span2_text']}"? В ответе напиши только "да" или "нет" """
        return prompt

    def danetqa_prompt(self, item):
        prompt = f"{item['question']}\nКонтекст: {item['passage']}\nИспользуя контекст, ответь на вопрос используя только да или нет."
        return prompt


class RussianSuperGlueEval:
    def lidirus_eval(self, item=None, result=None):
        answer = None

        result = result.lower()
        if "да" in result:
            answer = 1
        elif "не" in result:
            answer = 0
        else:
            answer = int(not bool(item["label"]))
        answer = "entailment" if answer == 1 else "not_entailment"
        return answer

    def rcb_eval(self, item=None, result=None):
        answer = item["label"]
        answer_map = {
            0: "entailment",
            1: "contradiction",
            2: "neutral",
            -1: "neutral",
        }
        answer = answer_map[answer]

        result = result.lower().replace(".", "")

        if result in ["neutral", "entailment", "contradiction"]:
            return result
        else:
            incorrect_answer = "neutral"
            return incorrect_answer

    def parus_eval(self, item=None, result=None):
        answer = None

        if "1" in result:
            answer = 0
        elif "2" in result:
            answer = 1
        else:
            answer = int(not bool(item["label"]))
        return answer

    def muserc_eval(self, item=None, result=None):
        answer = None
        result = result.lower()
        words_0 = [
            "неправил",
            "не правил",
            "нет,",
        ]
        words_1 = [
            "правил",
            "да,",
        ]

        if any([item in result for item in words_0]):
            answer = 0
        elif any([item in result for item in words_1]):
            answer = 1
        else:
            answer = int(not bool(item["label"]))

        return answer

    def terra_eval(self, item=None, result=None):
        answer = None
        result = result.lower()

        words_0 = [
            "неправил",
            "неверно",
            "нет,",
        ]
        words_1 = ["правил", "да,", "правд"]

        if any([item in result for item in words_0]):
            answer = "not_entailment"
        elif any([item in result for item in words_1]):
            answer = "entailment"
        else:
            answer = "not_entailment"
        return answer

    def russe_eval(self, item=None, result=None):
        answer = None
        result = result.lower()

        words_0 = [
            "неправил",
            "неверно",
            "нет,",
            "нет.",
        ]
        words_1 = [
            "правил",
            "да,",
            "да.",
        ]

        if any([item in result for item in words_0]):
            answer = "false"
        elif any([item in result for item in words_1]):
            answer = "true"
        else:
            answer = "false"
        return answer

    def rwsd_eval(self, item=None, result=None):
        answer = None
        result = result.lower()

        words_0 = [
            "неправил",
            "неверно",
            "нет,",
            "нет.",
        ]
        words_1 = [
            "правил",
            "да,",
            "да.",
        ]

        if any([item in result for item in words_0]):
            answer = "False"
        elif any([item in result for item in words_1]):
            answer = "True"
        else:
            answer = "False"
        return answer

    def rucos_eval(self, item=None, result=None):
        if self.split == "test":
            return result
        result = result.lower().replace(".", "")
        for true_answer in item["answers"]:
            true_answer = true_answer.lower().replace(".", "")
            # print(true_answer, result)
            if true_answer in result:
                return True
        return False

    def danetqa_eval(self, item=None, result=None):
        answer = None
        result = result.lower()

        words_0 = [
            "неправил",
            "неверно",
            "нет,",
            "нет.",
        ]
        words_1 = [
            "правил",
            "да,",
            "да.",
        ]

        if any([item in result for item in words_1]):
            answer = "true"
        elif any([item in result for item in words_0]):
            answer = "false"
        else:
            answer = "false"
        return answer


class RussianSuperGlueModels:
    @staticmethod
    def verbalist_generation(
        prompt=None,
        model=None,
        bot_token_id=9225,
        generation_config=None,
        tokenizer=None,
        conv_class=None,
        gen_llm_func=None,
    ):
        conversation = conv_class(bot_token_id=bot_token_id)
        conversation.add_user_message(prompt)
        prompt = conversation.get_prompt(tokenizer)
        # print("PROMPT", prompt)

        output = gen_llm_func(
            model,
            tokenizer,
            prompt,
            generation_config,
        )
        # print("RESULT", output)
        return output

    @staticmethod
    def saiga_mistral(
        prompt=None,
        model=None,
        bot_token_id=9225,
        generation_config=None,
        tokenizer=None,
        conv_class=None,
        gen_llm_func=None,
    ):
        conversation = conv_class(bot_token_id=bot_token_id)
        conversation.add_user_message(prompt)
        prompt = conversation.get_prompt(tokenizer)
        output = gen_llm_func(
            model,
            tokenizer,
            prompt,
            generation_config,
        )
        return output

    @staticmethod
    def chat_gpt(prompt=None):
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        return chat_completion["choices"][0]["message"]["content"].strip()

    def dummy(self, prompt=None):
        tasks_map = {
            "lidirus": "да",
            "rcb": "entailment",
            "parus": "1",
            "muserc": "правил",
            "terra": "правил",
            "russe": "правил",
            "rwsd": "да",
            "danetqa": "да",
            "rucos": "Тест",
        }
        return tasks_map[self.dataset_name]

    @staticmethod
    def load_saiga():
        MODEL_NAME = "IlyaGusev/saiga_mistral_7b"
        DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>"
        DEFAULT_RESPONSE_TEMPLATE = "<s>bot\n"
        DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."

        class Conversation:
            def __init__(
                self,
                message_template=DEFAULT_MESSAGE_TEMPLATE,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                response_template=DEFAULT_RESPONSE_TEMPLATE,
            ):
                self.message_template = message_template
                self.response_template = response_template
                self.messages = [{"role": "system", "content": system_prompt}]

            def add_user_message(self, message):
                self.messages.append({"role": "user", "content": message})

            def add_bot_message(self, message):
                self.messages.append({"role": "bot", "content": message})

            def get_prompt(self, tokenizer):
                final_text = ""
                for message in self.messages:
                    message_text = self.message_template.format(**message)
                    final_text += message_text
                final_text += DEFAULT_RESPONSE_TEMPLATE
                return final_text.strip()

        def generate(
            model,
            tokenizer,
            prompt,
            generation_config,
        ):
            with torch.no_grad():
                data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                data = data.to(model.device)
                output_ids = model.generate(
                    **data, generation_config=generation_config
                )[0]
                output_ids = output_ids[len(data["input_ids"][0]) :]
                output = tokenizer.decode(output_ids, skip_special_tokens=True)
                return output.strip()

        config = PeftConfig.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            # load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, MODEL_NAME, torch_dtype=torch.float16)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
        print(generation_config)

        inputs = [
            "Почему трава зеленая?",
            # "Сочини длинный рассказ, обязательно упоминая следующие объекты. Дано: Таня, мяч",
        ]
        for inp in inputs:
            conversation = Conversation()
            conversation.add_user_message(inp)
            prompt = conversation.get_prompt(tokenizer)

            output = generate(model, tokenizer, prompt, generation_config)
            print(inp)
            print(output)
            print()
            print("==============================")
            print()

        return {
            "model": model,
            "tokenizer": tokenizer,
            "conv_class": Conversation,
            "gen_llm_func": generate,
            "generation_config": generation_config,
        }

    @staticmethod
    def load_verbalist():
        weights_path = (
            "verbalist/model/models/verbalist_7b_v9/checkpoint-800/adapter_model"
        )
        tokenizer_path = "verbalist/model/models/verbalist_7b_v9/"

        config = PeftConfig.from_pretrained(weights_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            model,
            weights_path,
            torch_dtype=torch.float16,
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )

        inputs = ["Почему трава зеленая?"]

        conversation = VerbalistConversation(
            bot_token_id=12435,
        )
        conversation.add_user_message(inputs[0])
        prompt = conversation.get_prompt(tokenizer)
        print("PROMPT", prompt)
        generation_config_verbalist = GenerationConfig(
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            max_new_tokens=50,
            # no_repeat_ngram_size=15,
            repetition_penalty=1.1,
            temperature=0.5,
            top_k=20,
            top_p=0.95,
            do_sample=True,
        )
        output = generate(model, tokenizer, prompt, generation_config_verbalist)
        # print(inp)
        print(output)

        return {
            "model": model,
            "tokenizer": tokenizer,
            "conv_class": VerbalistConversation,
            "gen_llm_func": generate,
            "generation_config": generation_config_verbalist,
        }


class EvalRussianSuperGlue(
    RussianSuperGluePrompts,
    RussianSuperGlueEval,
    RussianSuperGlueModels,
):
    def __init__(
        self,
        dataset_name="danetqa",
        model_type=None,
        base_folder=None,
        generation_function=None,
        split=None,
        eval_name=None,
        debug_mode=False,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset = load_dataset("RussianNLP/russian_super_glue", dataset_name)
        self.split = split

        if dataset_name in ["lidirus"]:
            self.dataset = self.dataset["test"]
        else:
            if split != "test":
                self.dataset = self.dataset["validation"]
            else:
                self.dataset = self.dataset["test"]

        self.generation_function = generation_function
        self.model_type = model_type

        self.base_folder = Path(base_folder)
        self.eval_name = eval_name

        self.debug_mode = debug_mode

        if self.debug_mode:
            num = 10
            self.dataset = self.dataset.select(range(20, 20 + num))

        self.eval_filenames = {
            "lidirus": "LiDiRus",
            "rcb": "RCB",
            "parus": "PARus",
            "muserc": "MuSeRC_flat",
            "terra": "TERRa",
            "russe": "RUSSE",
            "rwsd": "RWSD",
            "danetqa": "DaNetQA",
            "rucos": "RuCoS",
        }

    def evaluate(self):
        task_name = self.eval_filenames[self.dataset_name]
        print(task_name)
        split_name = self.split
        if self.debug_mode:
            split_name = "debug"

        eval_folder = self.base_folder / f"{self.eval_name}" / split_name
        eval_folder.mkdir(exist_ok=True, parents=True)
        output_file = eval_folder / f"{task_name}.jsonl"

        if output_file.is_file() and not self.debug_mode and self.model_type != "dummy":
            with open(eval_folder / f"{task_name}.txt", "r") as f:
                print(f.read())
        else:
            predicts = []
            ground_true = []

            with open(eval_folder / f"{task_name}.log", "w") as f:
                idxs = []

                for item in tqdm(self.dataset):
                    prompt = self.get_prompt(item=item)
                    result = self.get_answer(prompt=prompt)

                    print(prompt, file=f)
                    print(f"predict answer = {result}", file=f)
                    if "label" in item:
                        print(f"real answer = {item['label']}", file=f)
                    else:
                        print(f"real answer = {item['answers']}", file=f)

                    answer = self.evaluate_answer(item=item, result=result)
                    gold_true = self.get_gold_true(item=item)

                    predicts.append(answer)
                    ground_true.append(gold_true)
                    idxs.append(item["idx"])
                acc = None
                if self.dataset_name != "rucos":
                    acc = str(accuracy_score(ground_true, predicts))
                else:
                    acc = str(0)
                    if self.split == "valid":
                        acc = str(accuracy_score(ground_true, predicts))
                        # print(ground_true, predicts)
                    idxs = [item["query"] for item in idxs]

                print(f"Accuracy: {acc}")

                with open(output_file, "w") as f:
                    for idx, predict in zip(idxs, predicts):
                        answer = {
                            "idx": idx,
                            "label": predict,
                        }
                        json.dump(answer, f, ensure_ascii=False)
                        f.write("\n")
                if self.dataset_name == "muserc":
                    self.save_muserc(
                        path_flat=output_file,
                        save_path=eval_folder / f"MuSeRC.jsonl",
                    )
                with open(eval_folder / f"{task_name}.txt", "w") as f:
                    f.write(acc)

    def get_answer(self, prompt):
        answer = None
        if self.model_type == "dummy":
            answer = self.dummy(prompt=prompt)
        else:
            answer = self.generation_function(prompt=prompt)
        answer = answer.strip()
        return answer

    def get_gold_true(self, item):
        handlers_map = {
            "lidirus": lambda item: {
                1: "entailment",
                0: "not_entailment",
            }[item["label"]],
            "rcb": lambda item: {
                0: "entailment",
                1: "contradiction",
                2: "neutral",
                -1: "neutral",
            }[item["label"]],
            "rucos": self.rucos_gold,
            "muserc": lambda item: item["label"],
            "terra": lambda item: {
                0: "entailment",
                1: "not_entailment",
                -1: "not_entailment",
            }[item["label"]],
            "russe": lambda item: {
                0: "false",
                1: "true",
                -1: "true",
            }[item["label"]],
            "rwsd": lambda item: {
                0: "False",
                1: "True",
                -1: "False",
            }[item["label"]],
            "danetqa": lambda item: {
                0: "false",
                1: "true",
                -1: "frue",
            }[item["label"]],
            "parus": lambda item: item["label"],
        }
        return handlers_map[self.dataset_name](item=item)

    def rucos_gold(self, item):
        if self.split == "test":
            return item["answers"]
        else:
            return True

    def get_prompt(self, item):
        handlers_map = {
            "lidirus": self.lidirus_prompt,
            "rcb": self.rcb_prompt,
            "rucos": self.rucos_prompt,
            "muserc": self.muserc_prompt,
            "terra": self.terra_prompt,
            "danetqa": self.danetqa_prompt,
            "parus": self.parus_prompt,
            "russe": self.russe_prompt,
            "rwsd": self.rwsd_prompt,
        }
        return handlers_map[self.dataset_name](item=item)

    def evaluate_answer(self, result, item):
        handlers_map = {
            "lidirus": self.lidirus_eval,
            "rcb": self.rcb_eval,
            "rucos": self.rucos_eval,
            "muserc": self.muserc_eval,
            "terra": self.terra_eval,
            "danetqa": self.danetqa_eval,
            "parus": self.parus_eval,
            "russe": self.russe_eval,
            "rwsd": self.rwsd_eval,
        }
        return handlers_map[self.dataset_name](item=item, result=result)

    def save_muserc(self, path_flat, save_path):
        with open(path_flat, "r") as f:
            lines = f.readlines()
            lines = [json.loads(item) for item in lines]
            real_prediction = {}
            for line in lines:
                paragraph, question, answer = (
                    line["idx"]["paragraph"],
                    line["idx"]["question"],
                    line["idx"]["answer"],
                )
                label = line["label"]
                if not paragraph in real_prediction:
                    real_prediction[paragraph] = {
                        "idx": paragraph,
                        "passage": {
                            "questions": {
                                question: {
                                    "idx": question,
                                    "answers": [
                                        {
                                            "idx": answer,
                                            "label": label,
                                        }
                                    ],
                                }
                            }
                        },
                    }
                else:
                    if (
                        not question
                        in real_prediction[paragraph]["passage"]["questions"]
                    ):
                        real_prediction[paragraph]["passage"]["questions"][question] = {
                            "idx": question,
                            "answers": [
                                {
                                    "idx": answer,
                                    "label": label,
                                }
                            ],
                        }
                    else:
                        real_prediction[paragraph]["passage"]["questions"][question][
                            "answers"
                        ].append(
                            {
                                "idx": answer,
                                "label": label,
                            }
                        )
        real_prediction = list(real_prediction.values())
        for item in real_prediction:
            item["passage"]["questions"] = list(item["passage"]["questions"].values())

        with open(save_path, "w") as f:
            for item in real_prediction:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")


if __name__ == "__main__":
    verbalist_config = RussianSuperGlueModels.load_verbalist()

    for name in [
        "lidirus",
        "rcb",
        "parus",
        "muserc",
        "terra",
        "russe",
        "rwsd",
        "danetqa",
        "rucos",
    ]:
        evaluation = EvalRussianSuperGlue(
            dataset_name=name,
            # split="valid",
            split="test",
            # model_type="dummy",
            base_folder="verbalist/evaluation/russian_super_glue/valid_evaluations/",
            # eval_name="dummy",
            # eval_name="saiga_mistral",
            eval_name="verbalist_7b_v9_800",
            # debug_mode=True,
            # generation_function=partial(
            #     RussianSuperGlueModels.chat_gpt,
            # ),
            # generation_function=partial(
            #     RussianSuperGlueModels.saiga_mistral,
            #     model=model,
            #     generation_config=generation_config
            # ),
            # generation_function=partial(
            #     RussianSuperGlueModels.saiga_mistral,
            #     model=model,
            #     generation_config=generation_config,
            # ),
            generation_function=partial(
                RussianSuperGlueModels.verbalist_generation,
                bot_token_id=12435,
                **verbalist_config,
            ),
        )

        evaluation.evaluate()
