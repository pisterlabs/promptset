import fnmatch
import numpy as np

WORDS = []


def word2action(word):
    action = []
    for l in word:
        action.append(ord(l) - 97)
    return action


def wordlist(letters, letters_not, letters_inc_pos, words_lst):
    action_w = "?????"
    for l in letters:
        action_w = action_w[:l[1]] + l[0] + action_w[l[1] + 1:]
    filter_lines = fnmatch.filter(words_lst, action_w)
    # print("matching state_w filter_lines are {}", len(filter_lines))
    filter_list = []
    for word in filter_lines:
        should_add = True
        for l in letters_not:
            if l in word:
                should_add = False
                break
        if should_add:
            filter_list.append(word)
    # print("after removing letters_not {}", len(filter_list))
    filter_lines = []
    for word in filter_list:
        should_add = True
        for (l, pos) in letters_inc_pos:
            if word[pos] == l:
                should_add = False
                break
        # Probably not needed here - CHECK
        # for (l, pos) in letters_rep_not:
        #     if word[pos] == l:
        #         should_add = False
        #         break
        if should_add:
            filter_lines.append(word)
    # print("after removing letters_inc_pos {}", len(filter_lines))
    poss_actions = []
    poss_actions_dict = {}
    for (l, pos) in letters_inc_pos:
        for i, ch in enumerate(action_w):
            if (ch == '?' and i != pos):
                action_w1 = action_w
                # state_w1[i] = l
                action_w1 = action_w1[:i] + l + action_w1[i + 1:]
                if l not in poss_actions_dict:
                    poss_actions_dict[l] = []
                poss_actions_dict[l].append(action_w1)
                poss_actions.append(action_w1)
    # print("poss states are {}", poss_states)
    poss_words = []
    for l in poss_actions_dict:
        # print("l is "+l)
        poss_action_list = poss_actions_dict[l]
        poss_words_l = []
        for action_str in poss_action_list:
            # print("State_str is "+state_str)
            templist = fnmatch.filter(filter_lines, action_str)
            poss_words_l = poss_words_l + templist
        poss_words_l = list(dict.fromkeys(poss_words_l))
        # print("temp poss words are {}", len(poss_words_l))
        if (len(poss_words) > 0):
            poss_words = list(set(poss_words) & set(poss_words_l))
        else:
            poss_words = poss_words_l
    if (len(poss_words) == 0):
        poss_words = filter_lines
    # print("final poss words are {}", len(poss_words))
    # print(poss_words)

    # prob = random.uniform(0, 1)
    # index = 0
    # if prob < epsilon and len(poss_words) > 1:
    #     index  = random.randint(0, len(poss_words)-1)
    # print("index is "+str(index))
    # print("final length of possible words {}", len(poss_words))
    return poss_words


class State():
    def __init__(self):
        """
        Use state as follows:
        Index 0: Number of remaining turns
        Index 1-26: Whether a letter has been attempted
        Index 27-52: If a letter is absent from the word
        Index 53-N: Binary feature for each letter and each position
            and one of {Yes, Incorrect position for the letter}
            Arranged as A1Y A1I A2Y A2I .... A5I B1Y .... 
        """
        self.state_len = 1 + 26 + 26 + 2 * 5 * 26
        self.state = np.zeros(self.state_len)
        self.state[0] = 6

    def copy_state(self, state):
        """Copy over the input state

        Args:
            state (np array)
        """
        self.state = np.copy(state)

    def from_word(self, word):
        """Incorporate word into state. Only used for initial state.

        Args:
            word (str)
        """
        for i in word2action(word):
            self.state[i + 1] = 1

    def from_obs(self, state, word, obs):
        """Incorporate the information of last state and current observation

        Args:
            state: last state
            word: currently guessed word
            obs: observation received from OpenAI gym
        """
        self.copy_state(state.state)
        for a in word:
            self.state[ord(a) - 96] = 1

        self.state[0] -= 1

        result = None
        for w in obs['board']:
            if -1 in w:
                break
            result = w
        for pos in range(5):
            letter = ord(word[pos]) - 97  # A -> 0, B -> 1, ....
            letter_result = result[pos]
            if letter_result == 0:
                # The corresponding letter does not appear in the word
                self.state[27 + letter] = 1
            elif letter_result == 2:
                # The corresponding letter is at the correct position in the word
                self.state[53 + letter * 10 + pos * 2] = 1
            else:
                # The corresponding letter is at the incorrect position in the word
                self.state[53 + letter * 10 + pos * 2 + 1] = 1

    def possible_actions(self):
        """
        Read the list of possible words, and list all possible next words from this state
        """
        if self.state[0] == 0:
            return 0

        # Get the list of next possible words
        letters = []
        letters_not = []
        letters_inc_pos = []
        for i in range(26):
            alphabet = chr(i + 97)
            is_attempted = self.state[i + 1]
            if is_attempted == 1:
                # Check whether the word is absent
                absent = self.state[27 + i] == 1
                if absent:
                    letters_not.append(alphabet)
                    continue

                start_index = 53 + i * 10
                for pos in range(5):
                    # Check if present at pos
                    if self.state[start_index + pos * 2] == 1:
                        letters.append((alphabet, pos))
                        continue
                    if self.state[start_index + pos * 2 + 1] == 1:
                        letters_inc_pos.append((alphabet, pos))

        possible_next_words = wordlist(letters, letters_not, letters_inc_pos,
                                       WORDS)
        next_states = []
        for poss_word in possible_next_words:
            # Calculate the state
            s_prime = State()
            s_prime.copy_state(self.state)
            s_prime.state[0] -= 1
            for l in poss_word:
                if l not in letters:
                    s_prime.state[1 + ord(l) - 97] = 1
            next_states.append(s_prime)

        return possible_next_words, next_states


def load_words(filename):
    with open(filename + '.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        WORDS.append(line.strip().split()[0])
