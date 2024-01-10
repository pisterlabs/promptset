import openai
import random
import time


class Game:
    def __init__(self, chain_of_thought=False, wait_time=1) -> None:
        with open("wordlist-eng.txt") as infile:
            word_list = infile.read().splitlines()
            # pick 25 random words
            word_list = random.sample(word_list, 25)

        # create some state variables
        self.winner = None
        self.words = word_list
        self.current_player = "red"
        self.not_current_player = "blue"
        self.wait_time = wait_time
        
        indices = list(range(25))
        random.shuffle(indices)

        # assign the words to each color.
        # Make lookups from color to words and words to color
        self.color2words = {}
        self.word2color = {}
        self.color2words["red"] = [self.words[i] for i in indices[:9]]
        self.color2words["blue"] = [self.words[i] for i in indices[9:17]]
        self.color2words["neutral"] = [self.words[i] for i in indices[17:24]]
        self.color2words["assassin"] = [self.words[indices[24]]]
        for color in self.color2words:
            for word in self.color2words[color]:
                self.word2color[word] = color
        
        # load the prompt template
        prompt_file = "user_prompt_template.txt"
        if chain_of_thought:
            prompt_file = "user_prompt_template_cot.txt"
        with open(prompt_file) as infile:
            self.use_prompt_template = "".join(infile.readlines())
        

    def print_remaining_words(self):
        longest_word_len = max([len(word) for word in self.words])
        longest_word_len += 2
        longest_color_len = max([len(color) for color in ["red", "blue", "neutral", "assassin"]])   
        longest_color_len += 2
        print("Red:".ljust(longest_color_len), end=" ")
        for word in self.color2words["red"]:
            print(word.ljust(longest_word_len), end=" ")
        print()
        print("Blue:".ljust(longest_color_len), end=" ")
        for word in self.color2words["blue"]:
            print(word.ljust(longest_word_len), end=" ")
        print()
        print("Neutral:".ljust(longest_color_len), end=" ")
        for word in self.color2words["neutral"]:
            print(word.ljust(longest_word_len), end=" ")
        print()
        print("Assassin:".ljust(longest_color_len), end=" ")
        for word in self.color2words["assassin"]:
            print(word.ljust(longest_word_len), end=" ")
        print()
    
    def get_guess_prompt(self, clue, num_words):
        # build the prompt from clue, num_words, the word_list and the template
        prompt = None
        return prompt


    def process_guesses(self, chatgpt_response_text, num_words, clue):
        # process the chatgpt_response_text to get the guesses and then update the game state accordingly
        guesses = chatgpt_response_text.split("####")[1].split()
        for guess in guesses:
            if guess not in self.words:
                print("Error. Word not in the list")
            else:
                color = self.word2color[guess]
                self.word2color.pop(guess)
                self.words.remove(guess)
                if color == "assassin":
                    print(f"{guess}: You got the assassin card")
                    self.color2words["assassin"].remove(guess)
                    self.winner = self.not_current_player
                    break
                elif color == "neutral":
                    print(f"{guess}: You got a neutral card")
                    self.color2words["neutral"].remove(guess)
                    break
                elif color == self.current_player:
                    print(f"{guess}: Correct!")
                    self.color2words[self.current_player].remove(guess)
                    if len(self.color2words[self.current_player]) == 0:
                        self.winner = self.current_player
                        break
                elif color == self.not_current_player:
                    self.color2words[self.not_current_player].remove(guess)
                    print(f"{guess}: you found the enemies card")
                    if len(self.color2words[self.not_current_player]) == 0:
                        self.winner = self.not_current_player
                    break
                time.sleep(self.wait_time)


    def turn(self):
        print("\n\nIt is the " + game.current_player + " team's turn!\n")
        self.print_remaining_words()
        # query user for clue
        clue = input("Enter your clue: ")
        while True:
            try:
                num_words = int(input("Enter the number of words: "))
                break
            except:
                print("Please enter a number only.")

        # format our prompt

        example_words = list([f"word_{i+1}" for i in range(num_words)])
        
        # equivalent expression using for loop
        # example_words = []
        # for i in range(num_words):
        #     example_words.append(f"word_{i+1}")

        example = " ".join(example_words)
        system_prompt = "You are an assistant for playing codenames."
        user_prompt = self.use_prompt_template.format(self.words, num_words, clue, example)
        # query chatgpt for guesses

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        response_text = response.choices[0].message.content
        print(response_text)
        # process chatgpt response
        self.process_guesses(response_text, num_words, clue)

        # switch current player
        if self.current_player == "red":
            self.current_player = "blue"
            self.not_current_player = "red"
        else:
            self.current_player = "red"
            self.not_current_player = "blue"

if __name__ == "__main__":
    random.seed(2)
    wait_time = 1.5
    api_key = None # change this line
    assert api_key is not None
    openai.api_key = api_key
    game = Game(chain_of_thought=False, wait_time=wait_time)
    while game.winner is None:
        time.sleep(wait_time)
        game.turn()
    print(f"\nThe {game.winner} team wins!")
