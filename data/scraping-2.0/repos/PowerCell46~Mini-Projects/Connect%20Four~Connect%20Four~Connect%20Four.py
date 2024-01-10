import time
from colorama import Fore
import pyfiglet
import openai

API_KEY = open("apiKey", "r").read()
openai.api_key = API_KEY

first_player_points = 0
second_player_points = 0
the_players_want_to_play = True

figures_dictionary = {
        "blank_space": "□",
        "first_player": "⚪",
        "second_player": "⚫"
    }

print(Fore.LIGHTGREEN_EX + (pyfiglet.figlet_format("Welcome", font="larry3d")))

time.sleep(1)


def get_bot_move(playing_field):
    prompt = f'We are playing Connect four. I have placed my disc. (I am playing with this symbol: {figures_dictionary["first_player"]} and you are playing with this symbol: {figures_dictionary["second_player"]}). This symbol: {figures_dictionary["blank_space"]} is an empty space. The current game state is: {playing_field}. Choose a column (greater than 0 and smaller than 8) and return a response in the format: "I choose column (your selected column).'
    conversation_history = [
        {"role": "user",
         "content": prompt
         }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )

    bot_move = response['choices'][0]['message']['content']
    return bot_move


def connect_four_game(opponent_type):
    playing_field = [[figures_dictionary["blank_space"] for el in range(7)] for row in range(6)]

    current_player_index = 0

    def win_check():
        for row in playing_field:  # Rows Checking
            current_element = row[0]
            current_counter = 1
            for index in range(1, len(row)):
                if row[index] == current_element:
                    current_counter += 1
                    if current_counter == 4 and current_element != figures_dictionary["blank_space"]:
                        return current_element

                else:
                    current_element = row[index]
                    current_counter = 1

        for index in range(0, len(playing_field[0])):  # Columns Checking
            current_element = playing_field[0][index]
            current_counter = 1
            for xedni in range(1, len(playing_field)):
                if playing_field[xedni][index] == current_element:
                    current_counter += 1
                    if current_counter == 4 and current_element != figures_dictionary["blank_space"]:
                        return current_element
                else:
                    current_element = playing_field[xedni][index]
                    current_counter = 1

        middle_index_row = 0
        middle_index_column = 0

        while middle_index_row < len(playing_field) and middle_index_column < len(playing_field[0]):  # left upper to right lower diagonal
            current_diagonal_indexes = []
            middle_index_row += 1
            middle_index_column += 1

            current_row = middle_index_row - 1
            current_column = middle_index_column
            while current_column > 0 and current_row < len(playing_field):
                current_column -= 1
                current_row += 1
                if current_row == len(playing_field):
                    break
                current_diagonal_indexes.insert(0, [current_row, current_column])

            current_row = middle_index_row
            current_column = middle_index_column - 1
            while current_row > 0 and current_column < len(playing_field[0]):
                current_row -= 1
                current_column += 1
                if current_column == len(playing_field[0]):
                    break
                current_diagonal_indexes.append([current_row, current_column])

            current_diagonal_elements = []
            for lst in current_diagonal_indexes:
                current_diagonal_elements.append(playing_field[lst[0]][lst[1]])

            if f'{figures_dictionary["first_player"]}{figures_dictionary["first_player"]}{figures_dictionary["first_player"]}{figures_dictionary["first_player"]}' in "".join(current_diagonal_elements):
                return figures_dictionary["first_player"]
            elif f'{figures_dictionary["second_player"]}{figures_dictionary["second_player"]}{figures_dictionary["second_player"]}{figures_dictionary["second_player"]}' in "".join(current_diagonal_elements):
                return figures_dictionary["second_player"]

        middle_index_row = len(playing_field)
        middle_index_column = len(playing_field[0])

        while middle_index_row > 0 and middle_index_column > 0:  # right lower to left upper diagonal
            current_diagonal_indexes = []
            middle_index_row -= 1
            middle_index_column -= 1

            current_row = middle_index_row + 1
            current_column = middle_index_column
            while current_row > 0 and current_column < len(playing_field[0]):
                current_row -= 1
                current_column += 1
                if current_column == len(playing_field[0]):
                    break
                current_diagonal_indexes.insert(0, [current_row, current_column])

            current_row = middle_index_row
            current_column = middle_index_column + 1
            while current_row < len(playing_field) and current_column > 0:
                current_row += 1
                current_column -= 1
                if current_row == len(playing_field):
                    break
                current_diagonal_indexes.append([current_row, current_column])

            current_diagonal_elements = []
            for lst in current_diagonal_indexes:
                current_diagonal_elements.append(playing_field[lst[0]][lst[1]])

            if f'{figures_dictionary["first_player"]}{figures_dictionary["first_player"]}{figures_dictionary["first_player"]}{figures_dictionary["first_player"]}' in "".join(current_diagonal_elements):
                return figures_dictionary["first_player"]
            elif f'{figures_dictionary["second_player"]}{figures_dictionary["second_player"]}{figures_dictionary["second_player"]}{figures_dictionary["second_player"]}' in "".join(current_diagonal_elements):
                return figures_dictionary["second_player"]

        middle_index_row = len(playing_field)
        middle_index_column = -1

        while middle_index_row - 1 > 0 and middle_index_column + 1 < len(playing_field[0]):  # lower left to upper right
            current_diagonal_indexes = []
            middle_index_row -= 1
            middle_index_column += 1

            current_row = middle_index_row - 1
            current_column = middle_index_column
            while True:
                current_row += 1
                current_column += 1
                if current_row == len(playing_field) or current_column == len(playing_field[0]):
                    break
                current_diagonal_indexes.insert(0, [current_row, current_column])

            current_row = middle_index_row
            current_column = middle_index_column + 1
            while True:
                current_row -= 1
                current_column -= 1
                if current_column == -1 or current_row == -1:
                    break
                current_diagonal_indexes.append([current_row, current_column])

            current_diagonal_elements = []
            for lst in current_diagonal_indexes:
                current_diagonal_elements.append(playing_field[lst[0]][lst[1]])

            if f'{figures_dictionary["first_player"]}{figures_dictionary["first_player"]}{figures_dictionary["first_player"]}{figures_dictionary["first_player"]}' in "".join(current_diagonal_elements):
                return figures_dictionary["first_player"]
            elif f'{figures_dictionary["second_player"]}{figures_dictionary["second_player"]}{figures_dictionary["second_player"]}{figures_dictionary["second_player"]}' in "".join(current_diagonal_elements):
                return figures_dictionary["second_player"]

        middle_index_row = -1
        middle_index_column = len(playing_field[0])

        while middle_index_row + 1 < len(playing_field) and middle_index_column - 1 > 0:  # upper right to lower left
            current_diagonal_indexes = []
            middle_index_row += 1
            middle_index_column -= 1

            current_row = middle_index_row + 1
            current_column = middle_index_column
            while True:
                current_row -= 1
                current_column -= 1
                if current_row == -1 or current_column == -1:
                    break
                current_diagonal_indexes.insert(0, [current_row, current_column])

            current_row = middle_index_row
            current_column = middle_index_column - 1
            while True:
                current_row += 1
                current_column += 1
                if current_column == len(playing_field[0]) or current_row == len(playing_field):
                    break
                current_diagonal_indexes.append([current_row, current_column])

            current_diagonal_elements = []
            for lst in current_diagonal_indexes:
                current_diagonal_elements.append(playing_field[lst[0]][lst[1]])

            if f'{figures_dictionary["first_player"]}{figures_dictionary["first_player"]}{figures_dictionary["first_player"]}{figures_dictionary["first_player"]}' in "".join(current_diagonal_elements):
                return figures_dictionary["first_player"]
            elif f'{figures_dictionary["second_player"]}{figures_dictionary["second_player"]}{figures_dictionary["second_player"]}{figures_dictionary["second_player"]}' in "".join(current_diagonal_elements):
                return figures_dictionary["second_player"]

    while True:
        def check_for_empty_spaces():
            there_are_no_free_spaces = True
            for row in playing_field:
                if figures_dictionary["blank_space"] in row:
                    there_are_no_free_spaces = False
            if there_are_no_free_spaces:
                return "Draw"

        time.sleep(1)
        if check_for_empty_spaces() == "Draw":
            print(Fore.YELLOW + f'All positions of the board are occupied and no one won the match.')
            time.sleep(1)
            print(Fore.LIGHTGREEN_EX + (pyfiglet.figlet_format("Draw", font="larry3d")))
            time.sleep(1.5)
            return "Draw"

        try:
            if current_player_index % 2 == 0:  # First player's move
                selected_column = int(input(Fore.RED + "Player 1, enter the column that you wish to select: (1) "))

            else:  # Second player's move / Bot's move
                if opponent_type == "with a bot":
                    while True:
                        current_response = get_bot_move(playing_field)
                        print(Fore.LIGHTCYAN_EX + f'The Bot\'s response is: {current_response}')
                        selected_column = [el for el in current_response if el.isdigit()]
                        if selected_column:
                            selected_column = int(selected_column[0])
                            break

                        time.sleep(1)
                        print(f'The generated response is invalid and a new one will be generated...')

                else:
                    selected_column = int(input(Fore.BLUE + "Player 2, enter the column that you wish to select: (2) "))

            if selected_column < 1 or selected_column > 7:
                raise ValueError

            def print_playing_field():
                for row in playing_field:
                    for el in row:
                        if el == figures_dictionary["first_player"]:
                            print(Fore.RED + el, end=" ")
                        elif el == figures_dictionary["second_player"]:
                            print(Fore.BLUE + el, end=" ")
                        elif el == figures_dictionary["blank_space"]:
                            print(Fore.RESET + el, end=" ")
                    print()

            for index in range(5, -1, -1):
                if playing_field[index][selected_column - 1] == figures_dictionary["blank_space"]:
                    time.sleep(1)
                    playing_field[index][selected_column - 1] = figures_dictionary["first_player"] if current_player_index % 2 == 0 else figures_dictionary["second_player"]
                    print_playing_field()
                    time.sleep(1.5)
                    if current_player_index % 2 == 0:
                        print(f'Player 1 successfully put his symbol on position: row: {index + 1}, column: {selected_column}!')
                    else:
                        if opponent_type == "with a player":
                            print(f'Player 2 successfully put his symbol on position: row: {index + 1}, column: {selected_column}!')
                        else:
                            print(f'The Bot successfully put it\'s symbol on position: row: {index + 1}, column: {selected_column}!')
                    break

            else:
                time.sleep(1)
                print(Fore.YELLOW + f'The whole column is occupied!')
                time.sleep(1.5)
                print_playing_field()
                time.sleep(1.5)
                print(Fore.YELLOW + f'Choose a new column!')
                current_player_index -= 1

        except:
            time.sleep(1.5)
            print(Fore.YELLOW + f'You have entered an invalid position.')
            time.sleep(1.5)
            print(Fore.YELLOW + f'Choose a new column!')
            current_player_index -= 1

        victory_figure = win_check()

        if victory_figure == figures_dictionary["first_player"] or victory_figure == figures_dictionary["second_player"]:
            time.sleep(2)
            print((Fore.LIGHTGREEN_EX + pyfiglet.figlet_format("Game Over!", font="larry3d")))
            time.sleep(2)
            print(Fore.RED + (pyfiglet.figlet_format("First player won!", font="larry3d"))) if victory_figure == figures_dictionary["first_player"] else print(Fore.BLUE + (pyfiglet.figlet_format("Second player won!", font="larry3d")))
            time.sleep(3)
            return f'First player' if victory_figure == figures_dictionary["first_player"] else "Second player"

        current_player_index += 1


while the_players_want_to_play:
    time.sleep(1)
    opponent_type = input("Do you wish to play vs another Player or do you wish to play vs a Bot? (with a player/ with a bot) ").lower()

    if opponent_type != "with a player" and opponent_type != "with a bot":
        time.sleep(0.5)
        opponent_type = "with a bot"
        print(f'You entered an invalid opponent so the program chose you to play versus a Bot.')

    time.sleep(1)

    print(Fore.LIGHTGREEN_EX + (pyfiglet.figlet_format("Connect Four is starting!", font="larry3d")))

    time.sleep(1.5)

    current_game_result = connect_four_game(opponent_type)

    if current_game_result == "Draw":
        first_player_points += 1
        second_player_points += 1
    elif current_game_result == "First player":
        first_player_points += 1
    elif current_game_result == "Second player":
        second_player_points += 1

    print(Fore.LIGHTGREEN_EX + f'The current result is: First Player: {first_player_points}, Second Player: {second_player_points}')
    time.sleep(2.5)

    if input("Do you want to play another Game? Y/N ").lower() != "y":
        time.sleep(1.5)
        print(pyfiglet.figlet_format(f'Thanks for playing Connect Four!', font="larry3d"))
        time.sleep(2.5)
        print(pyfiglet.figlet_format(f'Bye!', font="larry3d"))
        the_players_want_to_play = False
