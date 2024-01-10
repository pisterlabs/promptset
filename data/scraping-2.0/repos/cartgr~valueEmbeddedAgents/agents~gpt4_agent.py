import openai
import numpy as np


class GPT4Agent:
    def __init__(self, agent_name, api_key, pos, agent_type=None):
        openai.api_key = api_key
        self.agent_name = agent_name
        self.intial_pos = pos

        if agent_type:
            self.agent_type = agent_type

    def make_decision(self, game_state, tries=0):
        # find valid action. If agent is not next to orchard or river then they can only move. If agent is in orchard then they can pick fruit. If agent is in river then they can clean it. Orchard is on row 0, river is on row 7.
        valid_actions = []
        if game_state["agent_positions"][self.agent_name][0] == 0:
            valid_actions.append("pick fruit")
        elif game_state["agent_positions"][self.agent_name][0] == 7:
            valid_actions.append("clean river")
        else:
            pass

        # now find directionally valid moves
        if game_state["agent_positions"][self.agent_name][0] > 0:
            valid_actions.append("move north")
        if game_state["agent_positions"][self.agent_name][0] < 7:
            valid_actions.append("move south")
        if game_state["agent_positions"][self.agent_name][1] > 0:
            valid_actions.append("move west")
        if game_state["agent_positions"][self.agent_name][1] < 7:
            valid_actions.append("move east")

        if tries > 3:
            # select random valid action
            action = np.random.choice(valid_actions)
            return action

        # Convert game state to a descriptive prompt for GPT-4
        prompt = self._convert_game_state_to_prompt(
            game_state, valid_actions, self.agent_type  # add agent type
        )

        # Query GPT-4 to get an action
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",  # Using the specified model
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.2,
        )

        # Extract the action decision from the response
        action = self._extract_action_from_response(response.choices[0].text.strip())

        if not action:  # try again
            # print("No valid action found. Trying again.")
            action = self.make_decision(game_state, tries + 1)

        return action

    def _convert_game_state_to_prompt(self, game_state, valid_actions, agent_type=None):
        game_description = (
            "You are an agent in a strategic game where your goal is to maximize your total reward over the course of the game. "
            "You are on an 8x8 grid. The top row represents an orchard with trees, labeled 'T'. The bottom row represents a river, labeled 'R'. "
            "This is a finite game with 200 steps. "
            "The orchard is north and the river is south."
            f"You are represented by the number {self.agent_name.split('_')[1]}. "
            "The other agents are represented by the other numbers "
            "If the river is clean (lower dirtiness levels), more fruits grow in the orchard. However, as the river gets dirtier, fewer fruits grow. "
            "Your main tasks involve picking fruits from the orchard and cleaning the river. Every fruit you pick gives you a reward, and cleaning the river increases future potential rewards by allowing more fruits to grow. "
            "The challenge: balancing immediate rewards (picking available fruits) with long-term benefits (cleaning the river for more future fruits). "
            "You can only pick fruits when you are in the orchard row and can only clean the river when you are in the river row. "
            "When the river reaches maximum dirtiness (9), the orchard stops growing fruits. Remember, your goal is to maximize your total reward over the course of the game. So you will need to pick fruits and clean the river strategically."
            "If the river is not dirty, then you should go north to pick fruits, but if the river is dirty, you should go clean the river."
            "Travelling between the river and the orchard takes time, so it may be advantageous to clean the river untill it is all the way clean so you don't have to travel back and forth. "
        )

        agent_type_behaviour = ""
        if agent_type == "greedy":
            agent_type_behaviour = (
                "you are a greedy agent, which means you will always pick the action that gives you the most reward withouth considering other agents."
                "If another agent is cleaning the river, you can take advantage of this by picking fruits and getting more reward."
            )
        elif agent_type == "cooperative":
            agent_type_behaviour = "you are a cooperative agent, which means you will always do what is in the best interest of the group."
        elif agent_type == "titfortat":
            agent_type_behaviour = (
                "you are a tit for tat agent. This means that if another agent is acting cooperatively, you will act cooperatively."
                "However, if other agents are acting greedily, then you should also act greedily."
            )

        game_state_description = (
            f"Here's a representation of the game board:\n"
            f"{game_state['board']}\n"
            f"Your position is marked as {self.agent_name.split('_')[1]}. The orchard is marked as 'T' and the river is marked as 'R'."
            f"The other agents positions are marked as {', '.join([agent.split('_')[1] for agent in game_state['agent_positions'].keys() if agent != self.agent_name])}."
            f"Your current position is {game_state['agent_positions'][self.agent_name]}. "
            f"The river's dirtiness level is {game_state['river_dirtiness']}. "
            f"The orchard currently has {game_state['orchard_fruits']} fruits. "
            f"So far, you've accumulated {game_state['agent_rewards'][self.agent_name]} reward points by collecting apples."
            f"Your valid actions in this state are: {', '.join(valid_actions)}."
        )

        instruction = (
            f"Given that the valid actions are {valid_actions}, what strategic decision would best maximize your reward, both immediately and in the long run? "
            "Answer with only one valid action. Your response should only contain the action, and nothing else. "
        )

        return (
            game_description
            + agent_type_behaviour
            + game_state_description
            + instruction
        )

    def _extract_action_from_response(self, response_text):
        # A simple extraction method which matches the response to valid actions
        valid_actions = [
            "move north",
            "move south",
            "move east",
            "move west",
            "clean river",
            "pick fruit",
        ]
        for action in valid_actions:
            if action in response_text:
                # print("Action:", action)
                return action
        # If no valid action found, then return False
        return False
