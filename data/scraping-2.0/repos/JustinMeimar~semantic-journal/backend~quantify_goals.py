from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


CLIENT = OpenAI()

# goals_dict is a 3 length dictionary with goals
def init_chat(goals_dict):
    goals_list = list(goals_dict.values())
    SYSTEM_PROMPT = f"""
    You are a goal helper trying to evaluate quantitatively how well the user has performed on three goals. You need to generate 3 numbers ranging from 0 to 10, where 10 is completely meeting the goal an 0 is doing the opposite of the goal, respectively for the following 3 goals, and increments of 0.5: \n 1. {goals_list[0]} \n 2. {goals_list[1]} \n 3. {goals_list[2]}. Return the three numbers, with just the numbers, one on each new line, like this: 

    '0.5\n5.0\n9.5'

    You must output three numbers. If the input is unrelated give three zeros.
    """
    return [{'role': 'system', 'content': SYSTEM_PROMPT}]

def get_response(messages):
    response = CLIENT.chat.completions.create(
        model='gpt-4',
        messages=messages,
    )
    return response


# journal_entry is what the user entered to evaluate their goals. convos is convo history
def get_nums(convos, journal_entry):
    convos.append({'role': 'user', 'content': journal_entry})

    response = get_response(convos)
    num_list = list(map(float, response.choices[-1].message.content.split('\n')))

    convos.append({"role": "assistant", "content": response.choices[0].message.content})
    print(num_list)
    return num_list


def main():
    CONVOS = init_chat(["run faster", "eat better", "exercise daily"])
    nums = get_nums(CONVOS)
    print(nums)

if __name__ == '__main__':
    main()

    







