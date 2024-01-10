import json
from openai import AsyncOpenAI

openai_client = AsyncOpenAI()
default_model = 'gpt-4-1106-preview'


async def get_answer_full(system_prompt, user_prompt, use_proxy=False, model=default_model):
    try:
        completion = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        print("get_answer_full: ", completion.choices[0].message.content)
        return completion.choices[0].message.content

    except Exception as e:
        return f"Error: {e}"


def post_process_talk(say: str) -> str:
    # try:
    say = say.strip('`').strip()

    # Removing the preceding 'json' if it exists
    if say.startswith("json"):
        say = say[4:].strip()
    # Load the JSON from the string
    data = json.loads(say)
    
    # Check for the required fields
    required_fields = ["Reflection", "Strategy", "Chat", "Validation", "Reasoning"]
    if not all(field in data for field in required_fields):
        raise ValueError("The input JSON does not contain all required fields.")
    
    # Return the "Chat" field
    return data["Chat"]
    
    # except json.JSONDecodeError:
    #     raise ValueError("Invalid JSON format.")


def post_process_vote(vote: str) -> str:
    try:
        # Load the JSON from the string
        data = json.loads(vote)
        
        # Check for the required fields
        if "Analysis_of_each_player" not in data or "Voting" not in data:
            raise ValueError("The input JSON does not contain all required fields.")
        
        # Validate each analysis entry
        for entry in data["Analysis_of_each_player"]:
            if not all(field in entry for field in ["Player_id", "Reflection", "Observation", "If_human", "Reasoning", "Validation"]):
                raise ValueError("Some analysis entries are missing required fields.")
            
            if not isinstance(entry["Player_id"], int):
                raise ValueError("Player_id should be an integer.")
            
            if not isinstance(entry["If_human"], bool):
                raise ValueError("If_human should be a boolean.")
        
        # Validate the "Voting" field
        if not all(isinstance(pid, int) for pid in data["Voting"]):
            raise ValueError("Voting should be a list of integers.")
        
        voting_string = json.dumps(data["Voting"])
        return voting_string
    
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format.")


def load_message_from_text_file(txt_file_path):
    """
    Load message from a text file.

    Parameters:
    - txt_file_path (str): The path to the text file.

    Returns:
    - string: A string of contents in the file.
    """
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        message = file.read()
    
    return message


# for special char escape
def custom_format(text, **kwargs):
    temp_text = text.replace('{', '<<CUR_OPEN>>').replace('}', '<<CUR_CLOSE>>')
    formatted_text = temp_text.format(**kwargs)
    return formatted_text.replace('<<CUR_OPEN>>', '{').replace('<<CUR_CLOSE>>', '}')


class CicadaAgentBase:
    def __init__(self, total_players=5, total_rounds=3, char_limit=500, vote_score=25, voted_score=100):
        self.total_players = total_players
        self.total_rounds = total_rounds
        self.char_limit = char_limit
        self.vote_score = vote_score
        self.voted_score = voted_score

    def talk(self, round_id, player_id, chat_history) -> str:
        raise NotImplementedError

    def vote(self, player_id, chat_history) -> str:
        raise NotImplementedError


class CicadaAgent(CicadaAgentBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prompt_talk = ""
        self.prompt_vote = ""
        self.on_init()

    def on_init(self):
        self.prompt_talk = custom_format(
            load_message_from_text_file("lib/prompt_talk_template_v1.txt"),
            total_players=self.total_players,
            total_rounds=self.total_rounds,
            char_limit=self.char_limit,
            vote_score=self.vote_score,
            voted_score=self.voted_score,
            other_players = self.total_players - 1
        )
        self.prompt_vote = custom_format(
            load_message_from_text_file("lib/prompt_vote_template_v1.txt"),
            total_players=self.total_players,
            total_rounds=self.total_rounds,
            char_limit=self.char_limit,
            vote_score=self.vote_score,
            voted_score=self.voted_score,
            other_players = self.total_players - 1
        )

    async def talk(self, round_id, player_id, chat_history):
        # print(chat_history)
        # return f"Player {player_id} Round {round_id}"

        suffix = f"You are Agent #{player_id} in Round #{round_id + 1}.\n"
        suffix += f"The chat history is as follows:\n{chat_history}"

        say = await get_answer_full(self.prompt_talk , suffix, use_proxy=True)

        return post_process_talk(say)

    async def vote(self, player_id, chat_history):
        # return f"[{player_id}]"
        suffix = f"You are Agent #{player_id}.\n"
        suffix += f"The chat history is as follows:\n{chat_history}\n"

        say = await get_answer_full(self.prompt_vote , suffix, use_proxy=True)

        return post_process_vote(say)
