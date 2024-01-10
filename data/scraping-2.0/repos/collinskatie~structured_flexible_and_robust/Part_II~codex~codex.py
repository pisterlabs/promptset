import openai
from time import sleep

def get_completion(prompt, temperature, stop):
    sleep(10)
    response = openai.Completion.create(
        engine='davinci-codex',
        prompt=prompt,
        temperature=temperature,
        max_tokens=800,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop,
        n=1,
    )
    return response.choices[0]['text']

def test_get_completion():
    print(get_completion(code, 0.))

code = r'''def stack_planner_000():
    """
    There are items on a table. This function stacks them
    in a specific way.

    Initially:
    phone is on the table. plate is on the table. tablet is on the table.

    Goal:
    plate is on the table. tablet is on plate. phone is on tablet.

    Actions:
    """
    stack(tablet, plate)
    stack(phone, tablet)
    return



def stack_planner_001():
    """
    There are items on a table. This function stacks them
    in a specific way.

    Initially:
    tissue_box is on the table. phone is on tissue_box. tablet is on phone.

    Goal:
    phone is on the table. tissue_box is on the table. tablet is on tissue_box.

    Actions:
    """
    unstack(tablet)
    unstack(phone)
    stack(tablet, tissue_box)
    return



def stack_planner_002():
    """
    There are items on a table. This function stacks them
    in a specific way.

    Initially:
    frisbee is on the table. keyboard is on frisbee. tablet is on the table.

    Goal:
    tablet is on the table. frisbee is on tablet. keyboard is on frisbee.

    Actions:
    """
    unstack(keyboard)
    stack(frisbee, tablet)
    stack(keyboard, frisbee)
    return



def stack_planner_003():
    """
    There are items on a table. This function stacks them
    in a specific way.

    Initially:
    speaker is on the table. phone is on speaker. alarm_clock is on phone.

    Goal:
    phone is on the table. alarm_clock is on phone. speaker is on the table.

    Actions:
    """
'''

if __name__ == '__main__':
    print(test_get_completion())
