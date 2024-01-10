import os
import re
import sys
import json
import random
import logging
import argparse

import openai
import tiktoken

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(funcName)s() - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


def read_txt(file, write_log=False):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8") as f:
        text = f.read()

    if write_log:
        characters = len(text)
        logger.info(f"Read {characters:,} characters")
    return text


def write_txt(file, text, write_log=False):
    if write_log:
        characters = len(text)
        logger.info(f"Writing {characters:,} characters to {file}")

    with open(file, "w", encoding="utf8") as f:
        f.write(text)

    if write_log:
        logger.info(f"Written")
    return


def read_json(file, write_log=False):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8") as f:
        data = json.load(f)

    if write_log:
        objects = len(data)
        logger.info(f"Read {objects:,} objects")
    return data


def write_json(file, data, indent=None, write_log=False):
    if write_log:
        objects = len(data)
        logger.info(f"Writing {objects:,} objects to {file}")

    with open(file, "w", encoding="utf8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    if write_log:
        logger.info(f"Written")
    return


class Config:
    def __init__(self, config_file):
        data = read_json(config_file)

        self.model = data["model"]
        self.static_dir = os.path.join(*data["static_dir"])
        self.state_dir = os.path.join(*data["state_dir"])
        self.output_dir = os.path.join(*data["output_dir"])

        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        return


class GPT:
    def __init__(self, model):
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(self.model)
        self.model_candidate_tokens = {
            "gpt-3.5-turbo": {
                "gpt-3.5-turbo": 4096,
                "gpt-3.5-turbo-16k": 16384,
            },
            "gpt-4": {
                "gpt-4": 8192,
                "gpt-4-32k": 32768,
            }
        }
        return

    def get_specific_tokens_model(self, text_in, out_tokens):
        in_token_list = self.tokenizer.encode(text_in)
        in_tokens = len(in_token_list)
        tokens = in_tokens + out_tokens

        for candidate, max_tokens in self.model_candidate_tokens.get(self.model, {}).items():
            if max_tokens >= tokens:
                break
        else:
            candidate = ""

        return in_tokens, candidate

    def run_gpt(self, text_in, out_tokens):
        in_tokens, specific_tokens_model = self.get_specific_tokens_model(text_in, out_tokens)
        if not specific_tokens_model:
            return ""

        # logger.info(text_in)
        logger.info("I'm alive! Please wait awhile >< ...")

        completion = openai.ChatCompletion.create(
            model=specific_tokens_model,
            n=1,
            messages=[
                {"role": "user", "content": text_in},
            ]
        )

        text_out = completion.choices[0].message.content
        return text_out


class State:
    def __init__(self, save_file=""):
        self.save_file = save_file

        self.log = ""
        self.pedia = {}
        self.animal_list = []
        self.shapeshift = ""
        self.companion = ""
        self.ended = False
        return

    def save(self):
        data = {
            "log": self.log,
            "pedia": self.pedia,
            "animal_list": self.animal_list,
            "shapeshift": self.shapeshift,
            "companion": self.companion,
            "ended": self.ended,
        }
        write_json(self.save_file, data, indent=2)
        return

    def load(self):
        data = read_json(self.save_file)
        self.log = data["log"]
        self.pedia = data["pedia"]
        self.animal_list = data["animal_list"]
        self.shapeshift = data["shapeshift"]
        self.companion = data["companion"]
        self.ended = data["ended"]
        return


class Game:
    def __init__(self, config):
        self.static_dir = config.static_dir
        self.state_dir = config.state_dir
        self.output_dir = config.output_dir
        self.summary_file = ""
        self.gpt = GPT(config.model)
        self.gpt4 = GPT("gpt-4")

        self.user_prompt_to_text = {}
        self.max_saves = 4
        self.state = State()
        self.all_animal_list = [
            "bear", "boar", "cat", "deer", "eagle", "wolf",
        ]
        self.action_to_text = {
            "exit": "紮營休息(離開遊戲)",
            "watch_pedia": "觀看動物圖鑑",
            "search_animal": "尋找新的動物",
            "visit_animal": "拜訪動物",
        }
        self.animal_data = {}

        # load prompt text
        user_prompt_dir = os.path.join(self.static_dir, "user_prompt")
        filename_list = os.listdir(user_prompt_dir)
        for filename in filename_list:
            user_prompt = filename[:-4]
            user_prompt_file = os.path.join(user_prompt_dir, filename)
            self.user_prompt_to_text[user_prompt] = read_txt(user_prompt_file)

        # load animal data
        for animal in self.all_animal_list:
            animal_file = os.path.join(self.static_dir, "animal", f"{animal}.json")
            self.animal_data[animal] = read_json(animal_file)

        return

    def run_start(self):
        # get start type
        user_prompt = self.user_prompt_to_text["start"]
        while True:
            text_in = input(user_prompt)
            if text_in == "1":
                start_type = "new"
                break
            elif text_in == "2":
                start_type = "load"
                break

        # get save file usage
        save_list_text = "\n存檔列表：\n"
        saveid_to_exist = {}
        for i in range(self.max_saves):
            save_id = str(i + 1)
            save_file = os.path.join(self.state_dir, f"save_{save_id}.json")
            if os.path.exists(save_file):
                saveid_to_exist[save_id] = True
                save_list_text += f"({save_id}) 舊有存檔\n"
            else:
                saveid_to_exist[save_id] = False
                save_list_text += f"({save_id}) 空白存檔\n"

        # get save file
        user_prompt = f"{save_list_text}\n使用存檔欄位： "
        while True:
            text_in = input(user_prompt)
            if start_type == "new":
                if text_in in saveid_to_exist:
                    use_save_id = text_in
                    break
            else:
                if saveid_to_exist.get(text_in, False):
                    use_save_id = text_in
                    break

        # initialize state
        self.summary_file = os.path.join(self.output_dir, f"summary_{use_save_id}.txt")
        use_save_file = os.path.join(self.state_dir, f"save_{use_save_id}.json")
        self.state = State(use_save_file)
        if start_type == "new":
            user_prompt = self.user_prompt_to_text["opening"]
            input(user_prompt + "\n開始旅程(按換行繼續)... ")
            self.state.log += user_prompt
            self.state.save()
        else:
            self.state.load()

        self.run_loop()
        return

    def get_action(self):
        # get available actions
        action_list = [
            "exit",
            "watch_pedia",
        ]

        if len(self.state.animal_list) < len(self.all_animal_list):
            action_list.append("search_animal")

        if self.state.animal_list:
            action_list.append("visit_animal")

        action_list_text = "\n行動列表：\n"
        actionid_to_action = {}
        for i, action in enumerate(action_list):
            action_id = str(i)
            actionid_to_action[action_id] = action
            action_text = self.action_to_text[action]
            action_list_text += f"({action_id}) {action_text}\n"

        # get action
        user_prompt = f"{action_list_text}\n努伊特的下一步： "
        while True:
            text_in = input(user_prompt)
            if text_in in actionid_to_action:
                use_action = actionid_to_action[text_in]
                break

        return use_action

    def run_loop(self):
        while True:
            if not self.state.ended and self.state.shapeshift and self.state.companion:
                self.do_end()

            action = self.get_action()

            if action == "exit":
                break

            elif action == "watch_pedia":
                self.do_watch_pedia()

            elif action == "search_animal":
                self.do_search_animal()

            elif action == "visit_animal":
                self.do_visit_animal()

        return

    def do_watch_pedia(self):
        action_text = self.action_to_text["watch_pedia"]
        user_prompt = f"\n{action_text}：\n"

        if self.state.pedia:
            for animal, data in self.state.pedia.items():
                info = data["info"]
                knowledge = data["knowledge"]
                friendliness = data["friendliness"]
                user_prompt += f"\n{info}\n\n理解度: {knowledge}\n友誼度: {friendliness}\n"
        else:
            user_prompt += "...空空如也...\n"

        self.state.save()
        input(user_prompt + "\n(按換行繼續)... ")
        return

    def do_search_animal(self):
        action_text = self.action_to_text["search_animal"]
        user_prompt = f"\n{action_text}：\n"

        searchable_animal_list = [
            animal
            for animal in self.all_animal_list
            if animal not in self.state.animal_list
        ]

        if searchable_animal_list:
            animal = random.choice(searchable_animal_list)
            self.state.animal_list.append(animal)

            data = self.animal_data[animal]
            name_zh = data["name_zh"]
            name_en = data["name_en"]
            title = data["title"]
            description = data["description"]

            search_log = f"\n努伊特在森林裡遇到了一隻{title}{name_zh}！\n"
            self.state.log += search_log
            user_prompt += search_log + "\n\n"

            info = f"{name_zh}({name_en})\n{description}"
            knowledge = 0
            friendliness = 0

            self.state.pedia[animal] = {
                "info": info,
                "knowledge": 0,
                "friendliness": 0,
            }
            user_prompt += f"動物圖鑑更新：\n{info}\n\n理解度: {knowledge}\n友誼度: {friendliness}\n"

        else:
            user_prompt += "...沒有找到任何新的動物...\n"

        self.state.save()
        input(user_prompt + "\n(按換行繼續)... ")
        return

    def do_visit_animal(self):
        action_text = self.action_to_text["visit_animal"]
        user_prompt = f"\n{action_text}：\n"

        # show animal list
        user_prompt += "\n努伊特在山林裡認識的動物：\n"
        animalid_to_animal = {}
        animalid_to_name = {}
        for i, animal in enumerate(self.state.animal_list):
            animal_id = str(i + 1)
            animalid_to_animal[animal_id] = animal

            data = self.animal_data[animal]
            name = data["title"] + data["name_zh"]
            animalid_to_name[animal_id] = name

            user_prompt += f"({animal_id}) {name}\n"

        # select animal
        user_prompt = f"{user_prompt}\n拜訪： "
        while True:
            text_in = input(user_prompt)
            if text_in in animalid_to_animal:
                use_animal = animalid_to_animal[text_in]
                use_animal_name = animalid_to_name[text_in]
                break

        visit_log = f"\n努伊特前去拜訪{use_animal_name}\n"
        self.state.log += visit_log

        # interact with animal
        user_prompt = f"\n努伊特前去拜訪{use_animal_name}：\n"
        user_prompt += "(1) 向對方學習\n(2) (...自行輸入5個字以上的任意行動...)\n\n他想要："
        while True:
            text_in = input(user_prompt)
            if text_in == "1":
                self.do_learn(use_animal)
                break
            elif len(text_in) >= 5:
                self.do_interact(use_animal, text_in)
                break

        self.state.save()
        return

    def do_learn(self, animal):
        # get full info
        static_pedia = self.animal_data[animal]
        name = static_pedia["name_zh"]
        full_info = "[簡介]\n" + static_pedia["description"]
        for title, content in static_pedia["details"].items():
            full_info += f"\n[{title}]\n{content}"

        # get completed question-answer pairs
        state_pedia = self.state.pedia[animal]
        if "qa_list" not in state_pedia:
            state_pedia["qa_list"] = []
        qa_list = state_pedia["qa_list"]

        # get new question
        if qa_list:
            question_list_text = "問題列表：\n"
            for i, (question, answer) in enumerate(qa_list):
                question_list_text += f"{i + 1}. {question}\n"

            questions = len(qa_list) + 1
            question_list_text += f"{questions}. "

            instruction_text = f"問{questions}個可以從以下文章得到答案的問題。只能使用單一句問句。"

            gpt_in = f"{instruction_text}\n\n文章：\n{full_info}\n\n{question_list_text}"
            use_question = self.gpt.run_gpt(gpt_in, 100)

        else:
            use_question = f"{name}的食物來源包括哪些？"

        # get answer
        log = f"{name}詢問努伊特：「{use_question}」"
        self.state.log += f"\n{log}"
        user_prompt = f"\n{log}\n\n努伊特回答： "
        use_answer = input(user_prompt)
        log = f"努伊特回答：「{use_answer}」"
        self.state.log += f"\n{log}"

        # evaluate answer
        gpt_in = f"參考資料：\n{full_info}\n\n問題：\n{use_question}\n\n答案：\n{use_answer}\n\n請問答案是否正確？請回答「是」或「否」一個字"
        gpt_out = self.gpt.run_gpt(gpt_in, 10)
        is_correct = "否" not in gpt_out
        old_knowledge = state_pedia["knowledge"]
        if is_correct:
            new_knowledge = old_knowledge + 5
            log = f"{name}對努伊特的回應相當滿意"
            user_prompt = f"{log}\n努伊特的理解度從{old_knowledge}上升到了{new_knowledge}\n\n(按換行繼續)... "
            qa_list.append((use_question, use_answer))

        else:
            new_knowledge = max(0, old_knowledge - 1)
            log = f"{name}並不同意努伊特的答案"
            user_prompt = f"{log}\n努伊特的理解度剩下{new_knowledge}\n\n(按換行繼續)... "

        self.state.log += f"\n{log}\n"
        state_pedia["knowledge"] = new_knowledge
        input(user_prompt)

        # learn shapeshift
        if not self.state.shapeshift and animal != self.state.companion and new_knowledge >= 10:
            self.state.shapeshift = animal

            name_zh = static_pedia["name_zh"]
            name_en = static_pedia["name_en"]
            evolve_zh = static_pedia["evolve_zh"]
            evolve_en = static_pedia["evolve_en"]
            title = static_pedia["title"]
            teacher = f"{title}{name_zh}"
            species = f"{name_zh}({name_en})"
            evolve = f"{evolve_zh}({evolve_en})"

            user_prompt = self.user_prompt_to_text["shapeshift"]
            user_prompt = user_prompt.replace("teacher", teacher)
            user_prompt = user_prompt.replace("species", species)
            user_prompt = user_prompt.replace("evolve", evolve)

            input(user_prompt + "\n繼續旅程(按換行繼續)... ")
            self.state.log += user_prompt

        self.state.save()
        return

    def do_interact(self, animal, use_interact):
        # animal info
        static_pedia = self.animal_data[animal]
        name_zh = static_pedia["name_zh"]
        name_en = static_pedia["name_en"]
        evolve_zh = static_pedia["evolve_zh"]
        evolve_en = static_pedia["evolve_en"]
        title = static_pedia["title"]
        teacher = f"{title}{name_zh}"
        species = f"{name_zh}({name_en})"
        evolve = f"{evolve_zh}({evolve_en})"
        companion = f"{title}{evolve_zh}"

        # interact
        use_interact = f"努伊特試圖和{teacher}互動：{use_interact}"
        gpt_in = f"撰寫一個短篇故事，最多使用3句話。\n\n短篇故事：{use_interact}"
        continuation = self.gpt.run_gpt(gpt_in, 200)

        # revise
        gpt_in = \
            f"將短篇故事改寫的較為合理且通順，最多使用3句話，不能使用「試圖」，必須包含「努伊特」和「{teacher}」。\n\n" \
            f"短篇故事：\n{use_interact}，{continuation}\n\n" \
            f"合理且通順的改編故事：\n"
        story = self.gpt.run_gpt(gpt_in, 200)

        # score
        gpt_in = f"故事：\n{story}\n\n這是個溫馨的故事嗎？請回答「是」或「否」一個字"
        feedback = self.gpt.run_gpt(gpt_in, 10)
        if "否" in feedback:
            score = random.randrange(-1, 5)
        else:
            score = 5

        state_pedia = self.state.pedia[animal]
        old_friendliness = state_pedia["friendliness"]
        new_friendliness = old_friendliness + score

        log = f"\n{story}\n"
        self.state.log += log
        user_prompt = f"{log}\n努伊特和{name_zh}的友誼度現在是{new_friendliness}\n\n(按換行繼續)... "
        state_pedia["friendliness"] = new_friendliness
        input(user_prompt)

        # learn companion
        if not self.state.companion and animal != self.state.shapeshift and new_friendliness >= 10:
            self.state.companion = animal

            user_prompt = self.user_prompt_to_text["companion"]
            user_prompt = user_prompt.replace("teacher", teacher)
            user_prompt = user_prompt.replace("species", species)
            user_prompt = user_prompt.replace("evolve", evolve)
            user_prompt = user_prompt.replace("companion", companion)

            input(user_prompt + "\n繼續旅程(按換行繼續)... ")
            self.state.log += user_prompt

        self.state.save()
        return

    def do_end(self):
        # get ending text
        shapeshift_pedia = self.animal_data[self.state.shapeshift]
        shapeshift_name_zh = shapeshift_pedia["name_zh"]
        shapeshift_title = shapeshift_pedia["title"]
        shapeshift_evolve_zh = shapeshift_pedia["evolve_zh"]
        teacher = f"{shapeshift_title}{shapeshift_name_zh}"

        companion_pedia = self.animal_data[self.state.companion]
        companion_evolve_zh = companion_pedia["evolve_zh"]
        companion_title = companion_pedia["title"]
        companion = f"{companion_title}{companion_evolve_zh}"

        user_prompt = self.user_prompt_to_text["ending"]
        user_prompt = user_prompt.replace("teacher", teacher)
        user_prompt = user_prompt.replace("shapeshift", shapeshift_evolve_zh)
        user_prompt = user_prompt.replace("companion", companion)
        self.state.log += user_prompt
        input(user_prompt + "\n(按換行生成摘要)... ")

        # get story summary
        story = self.state.log
        story = re.sub(r"\n+", "\n", story).strip()
        instruction = "將這個長篇故事精簡為一個短篇故事。使用至少20句話。其中不能出現「10」"
        while True:
            gpt_in = f"{story}\n\n{instruction}"
            summary = self.gpt4.run_gpt(gpt_in, 1000)
            if summary:
                break
            story = story[:-100]

        summary = re.sub(r"\n+", "\n", summary).strip()

        # revise summary
        instruction = "根據事實，將故事修正為一個符合事實的短文。使用至少20句話。其中不能出現「10」"
        while True:
            gpt_in = f"事實：\n{story}\n\n故事：\n{summary}\n\n{instruction}"
            revision = self.gpt4.run_gpt(gpt_in, 1000)
            if revision:
                break
            story = story[:-100]

        write_txt(self.summary_file, revision)
        user_prompt = f"\n\n努伊特的旅程：\n{revision}\n\n(按換行繼續)... "
        input(user_prompt)

        self.state.ended = True
        self.state.save()
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="lab2_config.json")
    arg = parser.parse_args()

    openai.api_key = input("OpenAI API Key: ")

    config = Config(arg.config_file)
    game = Game(config)
    game.run_start()
    return


if __name__ == "__main__":
    main()
    sys.exit()
