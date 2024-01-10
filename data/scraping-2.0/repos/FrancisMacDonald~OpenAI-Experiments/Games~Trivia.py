import os
import random
import re

import openai


class Trivia:
    def __init__(self, openai_api_key, leaderboard_file_name):
        self._points_for_correct_answer = 1
        self._double_points_mode = False
        self._openai_api_key = openai_api_key
        self._generated_questions = []
        self._active_question = None
        self._guessed_this_question = []
        self._correct_answer = None
        self._leaderboard = {}

        self._prompt_topics = "engineering, modern music, movies, comic books, video games, history, politics"

        if leaderboard_file_name.endswith(".txt"):
            self._leaderboard_file_name = leaderboard_file_name
        else:
            self._leaderboard_file_name = leaderboard_file_name + ".txt"

        # load leaderboard from file
        if self.load_leaderboard():
            print("Loaded leaderboard from file")
        else:
            print("Failed to load leaderboard from file")

    def clear_leaderboard(self):
        self._leaderboard = {}
        self.save_leaderboard()

    async def generate_new_question(self):
        _answers = ""
        for answered_question in self._generated_questions:
            _answers += f"{answered_question},"

        openai.api_key = self._openai_api_key

        _prompt = f"You are a trivia bot. " \
                  f"You are creative and imaginative and you like to generate interesting questions that people want to answer. " \
                  f"You know a lot about ${self._prompt_topics}. You will prefer to ask questions about these topics. " \
                  f"You ensure the questions have an answer that is true. " \
                  f"You do not ask about geography or cities. " \
                  f"You do not add quotes to the questions or answers. " \
                  f"You are asked to generate a question with 4 answers. " \
                  f"You will make sure only 1 answer is correct. " \
                  f"You will make 1 of the answers a funny joke that sounds like an answer. You will never tell us which answer is a joke. " \
                  f"You will not label the questions and answers with 'Q' or 'Question' or 'A' or 'Answer'. " \
                  f"You will not add single quotes, or double quotes to the questions or answers. " \
                  f"You will return a list seperated by | with the question in position zero, the correct answer at position 1, and the rest of the answers in the remaining positions. " \

        # if len(self._generated_questions) > 0:
        #     _prompt += f"You cannot ask the following questions: '{_answers}' "

        _prompt += "The list: "

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=_prompt,
            temperature=0.9,
            max_tokens=100,
            # frequency_penalty=0,
            # presence_penalty=0.6,
        )

        choices = [re.sub(r'\(.*\)', '', choice.replace('"', '').strip()) for choice in response.choices[0].text.split("|")]
        print("choices: " + str(choices))
        scrambled_answers = choices[1:5]
        random.shuffle(scrambled_answers)
        scrambled_answers.insert(0, choices[0])

        self._correct_answer = choices[1]
        self._active_question = scrambled_answers
        self._generated_questions.append(choices[0])
    
    def start_double_points_mode(self):
        self._double_points_mode = True

    def stop_double_points_mode(self):
        self._double_points_mode = False
    
    def load_leaderboard(self):
        return self.load_leaderboard_from_file(self._leaderboard_file_name)

    def _get_new_text_question(self):
        self.generate_new_question()

        return f"{self._active_question[0]}\n" \
               f"A: {self._active_question[1]}\n" \
               f"B: {self._active_question[2]}\n" \
               f"C: {self._active_question[3]}\n" \
               f"D: {self._active_question[4]}\n"

    def change_prompt_topics(self, new_topics):
        self._prompt_topics = new_topics

    def _get_answer_from_input(self, answer):
        answer = answer.upper()

        if answer == "A" or answer == "1":
            return self._active_question[1]
        elif answer == "B" or answer == "2":
            return self._active_question[2]
        elif answer == "C" or answer == "3":
            return self._active_question[3]
        elif answer == "D" or answer == "4":
            return self._active_question[4]
        else:
            return None

    def _guess_active_question(self, guess):
        answer_from_input = self._get_answer_from_input(guess)

        if not answer_from_input:
            return False

        return answer_from_input == self._correct_answer

    def guess_question_user(self, guess, userid):
        if userid not in self._leaderboard:
            self._leaderboard[userid] = 0

        if self._guess_active_question(guess):
            # question guessed correctly, clear the question
            self.reset_questions()

            if self._double_points_mode:
                self._leaderboard[userid] += self._points_for_correct_answer * 2
            else:
                self._leaderboard[userid] += self._points_for_correct_answer

            self.save_leaderboard()
            return True
        else:
            self._guessed_this_question.append(userid)
            return False

    def save_leaderboard_to_file(self, filename):
        try:
            with open(filename, "w") as f:
                if len(self._leaderboard) == 0:
                    f.write("")
                else:
                    for user in self._leaderboard:
                        f.write(f"{user}|||||{self._leaderboard[user]}\n")
        except Exception as e:
            print("Failed to save leaderboard")
            print(e)
            return False

    def load_leaderboard_from_file(self, filename):
        try:
            if not os.path.exists(filename):
                # create file
                with open(filename, "w") as f:
                    f.write("")
                return True

            with open(filename, "r") as f:
                for line in f:
                    user, score = line.split("|||||")
                    self._leaderboard[user] = int(score)
        except Exception as e:
            print("Failed to load leaderboard")
            print(e)
            return False

        return True

    def save_leaderboard(self):
        self.save_leaderboard_to_file(self._leaderboard_file_name)

    def clear_all_guesses(self):
        self._guessed_this_question = []

    def get_score(self, userid):
        if userid not in self._leaderboard:
            return 0

        return self._leaderboard[userid]

    def get_leaderboard(self):
        return sorted(self._leaderboard.items(), key=lambda x: x[1], reverse=True)

    def get_leaderboard_as_text(self):
        leaderboard = self.get_leaderboard()
        leaderboard_text = ""

        for i in range(len(leaderboard)):
            leaderboard_text += f"{i+1}. {leaderboard[i][0]}: {leaderboard[i][1]}\n"

        return leaderboard_text

    def reset_leaderboard(self):
        really_reset = input("Are you sure you want to reset the leaderboard? (y/n): ")
        if really_reset == "y":
            self.clear_leaderboard()
        else:
            print("Reset aborted")

    def reset_questions(self):
        self._active_question = None
        self._correct_answer = None
        self._guessed_this_question = []

    def reset_game(self):
        self._generated_questions = []
        self.reset_questions()

    def ask_question_in_console(self):
        print(self._get_new_text_question())

        success = False
        answer1 = input("Enter your answer: ")
        guesses = 0
        max_guesses = 2

        while not success and guesses < max_guesses:
            if self.guess_question_user(answer1, "console"):
                print("Correct!")
                success = True
            else:
                print("Incorrect! Try again!")
                guesses += 1
                answer1 = input("Enter your answer: ")

    @property
    def active_question(self):
        return self._active_question

    @property
    def guessed_this_question(self):
        return self._guessed_this_question

    @property
    def prompt_topics(self):
        return self._prompt_topics
