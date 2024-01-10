import json
import pathlib
import re
import time
import argparse

from base import openai, SLEEP_SECONDS

system_msg = """Given a set of boxes and their initial contents, you are to make changes based on the specified actions. Understand and apply the following actions as described:

    Move: Transfer an item from one box to another. For example, "Move car from Box 1 to Box 2" means you should remove the "car" from Box 1 and place it in Box 2.
    Remove: Take out an item from a box, and it's no longer available. For instance, "Remove ball from Box 3" means the "ball" should be taken out of Box 3 and it won't be placed in any other box.
    Put: Add a new item to a box. For example, "Put apple into Box 4" means you should place an "apple" in Box 4.
    Empty: Clear out all the contents of a box. "Empty Box 5" means Box 5 should have no items left in it.
    Replace: Substitute one item in a box with another item. For instance, "Replace cat with dog in Box 6" means you should remove the "cat" from Box 6 and place a "dog" in its stead.
    Swap: Exchange the positions of two items from different boxes. For example, "Swap pen in Box 1 with pencil in Box 2" means the "pen" in Box 1 should be moved to Box 2, and the "pencil" in Box 2 should be moved to Box 1.

Using the descriptions and actions provided above, determine the final contents of each box after applying all specified changes.
Here is an example:

Description: Box 0 contains the shirt and the seaweed and the thunder, Box 1 contains the charger and the pants and the cat and the bag and the storm, Box 2 contains nothing, Box 3 contains the piano and the ring and the bird and the guitar, Box 4 contains the horse and the controller and the table, Box 5 contains the toothbrush, Box 6 contains the boot and the towel. Replace the bird and the guitar with the glove and the puzzle in Box 3. Put the glasses into Box 2. Remove the towel and the boot from Box 6. Replace the seaweed and the thunder with the elephant and the freezer in Box 0. Swap the charger in Box 1 with the glasses in Box 2. Remove the charger from Box 2. Move the shirt and the elephant and the freezer from Box 0 to Box 5. Remove the puzzle from Box 3. Move the storm and the cat from Box 1 to Box 4. Move the toothbrush from Box 5 to Box 2.
Statement: Box 0 contains nothing, Box 1 contains the glasses and the pants and the bag, Box 2 contains the toothbrush, Box 3 contains the piano and the ring and the glove, Box 4 contains the horse and the controller and the table and the storm and the cat, Box 5 contains the shirt and the elephant and the freezer, Box 6 contains nothing.
"""
# user1_msg = 'Description: Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map.'
# assistant1_msg = 'Statement: Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map.'
#
# user2_msg = 'Description: Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map. Remove the car from Box 0. Remove the paper and the string from Box 3. Put the plane into Box 0. Move the map from Box 6 to Box 2. Remove the bill from Box 4. Put the coat into Box 3.'
# assistant2_msg = 'Statement: Box 0 contains the plane, Box 1 contains the cross, Box 2 contains the bag and the machine and the map, Box 3 contains the coat, Box 4 contains nothing, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle.'


def parse_output(output):
    text = output.replace("Statement: ", "")
    box_descs = re.findall(r'Box \d+ contains (?:[^B]*(?:(?=, Box)|$))', text)

    box_dict = {}
    for desc in box_descs:
        desc = desc.replace(".", "")
        match = re.match(r'Box (\d+) contains (.*)', desc)

        if match:
            box_num = 'Box ' + match.group(1)
            items = [item.strip() for item in re.split(', and | and |, ', match.group(2))]
            box_dict[box_num] = items

    return box_dict


def process_dataset(
        dataset_path,
        output_base_path,
        engine,
        temperature,
):
    with open(dataset_path, 'r') as aggregated_boxes_file:
        aggregated_boxes = aggregated_boxes_file.readlines()

    for json_str in aggregated_boxes:
        data = json.loads(json_str)
        sentence = data['sentence']
        sentence_hash = data['sentence_hash']
        output_path = pathlib.Path(f"{output_base_path}/{sentence_hash}.json")

        if not output_path.is_file():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                response = openai.ChatCompletion.create(
                    model=engine,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": f'Description: {sentence}'},
                    ],
                    temperature=temperature,
                )
                output = response['choices'][0]['message']['content']

                parsed_output = parse_output(output)
                json_parsed_output = json.dumps(parsed_output, indent=4)

                with open(output_path, 'w') as f:
                    f.write(json_parsed_output)
                print(sentence_hash, "finished")
            except openai.error.OpenAIError as e:
                print(e)
                print("sleeping")
                time.sleep(SLEEP_SECONDS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the dataset using OpenAI.')
    parser.add_argument('--dataset_path', type=str, default='datasets/complex_aggregated_data.jsonl',
                        help='Path to the input aggregated data.')
    parser.add_argument('--output_base_path', type=str,
                        default='results/complex-boxes-dataset/plaintext/gpt-3.5-turbo/',
                        help='Base path for prediction outputs.')
    parser.add_argument('--engine', type=str, default="gpt-3.5-turbo", help='OpenAI engine to use.')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature setting for OpenAI model.')

    args = parser.parse_args()

    process_dataset(
        args.dataset_path,
        args.output_base_path,
        args.engine,
        args.temperature,
    )
