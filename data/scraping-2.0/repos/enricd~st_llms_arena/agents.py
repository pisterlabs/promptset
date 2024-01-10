from time import sleep, time
from random import randint
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.callbacks import get_openai_callback



def board_to_char(board_config, board_state, chars_type="emojis"):

    if chars_type == "emojis":
        CELL = "⬜"
        SNAKE1_HEAD = "🟢"
        SNAKE1 = "🟩"
        SNAKE2_HEAD = "🔵"
        SNAKE2 = "🟦"
        FOOD = "🍎"

    elif chars_type == "_GBR":
        CELL = " _"
        SNAKE1_HEAD = " G"
        SNAKE1 = " g"
        SNAKE2_HEAD = " B"
        SNAKE2 = " b"
        FOOD = " R"

    chars_board_list = []

    for y in range(board_config["GRID_SIZE"]):
        line = []
        for x in range(board_config["GRID_SIZE"]):

            if (x, y) in board_state["snake1"]["body"]:
                if (x, y) == board_state["snake1"]["body"][0]:
                    line.append(SNAKE1_HEAD)
                else:
                    line.append(SNAKE1)

            elif (x, y) in board_state["snake2"]["body"]:
                if (x, y) == board_state["snake2"]["body"][0]:
                    line.append(SNAKE2_HEAD)
                else:
                    line.append(SNAKE2)

            elif (x, y) in board_state["food"]:
                line.append(FOOD)

            else:
                line.append(CELL)

        chars_board_list.append(line)

    chars_board = "\n".join([f"{i:02}" + "".join(line) for i, line in enumerate(chars_board_list)])

    return chars_board


def get_agent_action(agent, llm, prompt, board_config, board_state, is_test=False):

    if not is_test:
        template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(prompt["sys_msg"]),
                HumanMessagePromptTemplate.from_template(prompt["human_msg"]),
            ]
        )

        emojis_board = board_to_char(board_config, board_state)
        chars_board = board_to_char(board_config, board_state, chars_type="_GBR")
        board_state_str = str(board_state)

        messages = template.format_messages(emojis_board=emojis_board, chars_board=chars_board, board_state_str=board_state_str)
        if agent == 2:
            print("messages:", messages)

        # OpenAI API call
        t0 = time()
        with get_openai_callback() as cb:
            agent_response = llm(messages).content
        llm_time = time() - t0

        if "⬆️" in agent_response:
            dir = "U"
        elif "⬇️" in agent_response:
            dir = "D"
        elif "⬅️" in agent_response:
            dir = "L"
        elif "➡️" in agent_response:
            dir = "R"
        else: 
            dir = None
        
        return (dir, agent_response, llm_time, cb.completion_tokens, cb.total_cost)

    else:
        
        sleep(0.5)
        if agent == 1:
            dir = "D" if randint(0, 1) == 0 else "R"
        
        elif agent == 2:
            dir = "U" if randint(0, 1) == 0 else "L"

        return (dir, "Test", 0.5, 0, 0)
        


if __name__ == "__main__":

    # --- Test case ---

    board_config = {
        "GRID_SIZE": 15,
        "SQUARE_SIZE": 35,
        "LINE_THICKNESS": 2,

        "BACKGROUND_COLOR": (30, 20, 20),
        "LINES_COLOR": (75, 40, 40),
        "SNAKE1_COLOR": (20, 200, 20),
        "SNAKE2_COLOR": (209, 31, 177),
        "FOOD_COLOR": (20, 200, 250)
    }

    board_state = {
        "turn": 0,
        "snake1": {
            "body": [(5, 2), (4, 2), (3, 2), (2, 2)],
            "dir": "R",
            "is_alive": True,
        },
        "snake2": {
            "body": [(9, 12), (10, 12), (11, 12), (12, 12)],
            "dir": "L",
            "is_alive": True,
        },
        "food": [(7, 7), (8, 1)],
    }

    print(board_to_char(board_config, board_state))
    print("")
    print(board_to_char(board_config, board_state, chars_type="_GBR"))