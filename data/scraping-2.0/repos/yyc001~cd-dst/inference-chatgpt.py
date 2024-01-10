import json
from tqdm import tqdm
from prompter import Prompter
from openai import OpenAI
from dotenv import load_dotenv


def main(
        schema_path="data/multiwoz/data/MultiWOZ_2.2/schema.json",
        processed_data_path="data/MultiWOZ_2.2_preprocess/test.json",
        output_file="data/MultiWOZ_2.2_preprocess/test_out_gpt.json",
):
    load_dotenv()
    client = OpenAI()
    prompter = Prompter(schema_path)
    data = json.load(open(processed_data_path, "r"))
    # data = [item for item in data if item["value"] != "none"]
    response_list = []
    aga_num = 0

    jga_num = 0
    jga_tot = -1
    last_index_turn = ""
    last_full_state = True

    for idx, sample in enumerate(tqdm(data)):
        this_index_turn = f'{sample["index"]}|{sample["turn"]}'
        if last_index_turn == "":
            last_index_turn = this_index_turn
        if last_index_turn and last_index_turn != this_index_turn:
            last_index_turn = this_index_turn
            jga_tot += 1
            if last_full_state:
                jga_num += 1
            last_full_state = True

        if idx > 0 and idx % 100 == 0:
            print(f"AGA for 0~{idx}: {aga_num / (idx + 1)}")
            print(f"JGA for 0~{idx}: {jga_num / jga_tot}")
        prompt = prompter.get_input_text(sample["dialogue"], sample["domain"], sample["slot"])
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompter.instruction},
                {"role": "user", "content": prompt},
            ]
        )
        output = completion.choices[0].message.content

        # print(output)
        response = prompter.get_response(output)

        if sample['value'].lower() == response.lower():
            aga_num += 1
        else:
            last_full_state = False
            # print(sample["dialogue"])
            print(prompt)
            print(f"{aga_num / (idx + 1)}|||{sample['slot']}|||{sample['value']}|||{response}")

        response_list.append({
            "index": sample["index"],
            "turn": sample["turn"],
            "domain": sample["domain"],
            "slot": sample["slot"],
            # "active": sample["active"],
            "value": response,
            "ground_truth": sample["value"]
        })
    print("accuracy:", 1 - aga_num / len(data))
    json.dump(response_list, open(output_file, "w"))


if __name__ == "__main__":
    main()
