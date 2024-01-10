import sys
import importlib
import argparse
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

sys.path.append("../ZeroShot-step-by-step-distillation")
run = importlib.import_module("run")
utils = importlib.import_module("src.utils")

load_dotenv()



def run_opro(dataset: str, n_prev_best: int, test_size: int, iterations: int):
    DATASET = dataset
    N_PRV_BEST = n_prev_best
    TEST_SIZE = test_size
    if DATASET == "anli1" or DATASET == "esnli":
        LABEL_PARSE = '(True|False|Inconclusive|Contradiction|Neutral|Entailment|not valid|valid|entails|contradicts|cannot be determined|uncertain|cannot determine)'
    elif DATASET == "cqa":
        LABEL_PARSE = '({choice_a}|{choice_b}|{choice_c}|{choice_d}|{choice_e}|a\)|b\)|c\)|d\)|e\)|1\)|2\)|3\)|4\)|5\)|1\.|2\.|3\.|4\.|5\.)'
    EXPL_PARSE = '.(.*)'

    for x in range(iterations):
        utils.print_c(f"ITERATION {x + 1}/{iterations}", c="green")
        # 1.) evaluate all prompts
        run.run_experiment(DATASET, test_size=TEST_SIZE, model="gpt-3.5-turbo", seed=42)

        # 2.) load all prompts, and get their performance
        prompt_templates = utils.read_yaml(f"prompt-templates/{DATASET}.yaml")["templates"]
        prompt_metadata = utils.read_yaml(f"prompt-metadata/{DATASET}.yaml")

        # 3.) build previous best
        id_acc = [(id, meta["performance"]["accuracy"]) for id, meta in prompt_metadata.items()]
        id_acc.sort(key=lambda x: x[1])
        if len(id_acc) > N_PRV_BEST:
            id_acc = id_acc[len(id_acc) - N_PRV_BEST:]


        previous_best = ""
        for id, acc in id_acc:
            previous_best += "PRT:\n"
            previous_best += prompt_templates[id]['user_message'] + "\n"
            previous_best += "Score:\n"
            previous_best += str(round(acc*100)) + "\n\n"

        # (4.) load data to generate examples)
        pass

        # 5.) prompt meta prompt 8 times and build prompt templates
        with open(f"opro/meta-prompts/{DATASET}.txt", "r") as f:
            meta_prompt = f.read()

        meta_prompt = meta_prompt.replace("<[PREVIOUS_BEST]>", previous_best)

        print("#" * 75)
        print(meta_prompt)
        print("#" * 75)

        chat_model = ChatOpenAI(model = "gpt-3.5-turbo", temperature=1.1, max_tokens=200)

        i = 0
        while i < 8:
            print(f"Querying {i+1}/8...")
            response = chat_model.predict(meta_prompt, timeout=10)
            #print(response)
            if DATASET == "anli1" or DATASET == "esnli":
                if all([x in response for x in ["{premise}", "{hypothesis}", "<PRT>", "</PRT>"]]) and response.count("{") == 2: # ANLI1, ESNLI
                    i += 1
                    response = response.split("<PRT>")[1].split("</PRT>")[0]
                    response = response.strip("\n")
                    utils.add_prompt_to_yaml(f"prompt-templates/{DATASET}.yaml", response, LABEL_PARSE, EXPL_PARSE)
                    print(response)
                else:
                    i += 0.5
            elif DATASET == "cqa":
                if all([x in response for x in ["{question}", "{choice_a}", "{choice_b}", "{choice_c}", "{choice_d}", "{choice_e}", "<PRT>", "</PRT>"]]) and response.count("{") == 6: # CQA
                    i += 1
                    response = response.split("<PRT>")[1].split("</PRT>")[0]
                    response = response.strip("\n")
                    utils.add_prompt_to_yaml(f"prompt-templates/{DATASET}.yaml", response, LABEL_PARSE, EXPL_PARSE)
                    print(response)
                else:
                    i += 0.25
            else:
                raise ValueError("Invalid dataset")

        utils.print_c("FINISHED ITERATION", c="green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--n_prev_best", type=int)
    parser.add_argument("--test_size", type=int)
    parser.add_argument("--iterations", type=int)

    args = parser.parse_args()

    run_opro(args.dataset, args.n_prev_best, args.test_size, args.iterations)