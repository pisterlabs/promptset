import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

MODEL = os.getenv("ANTHROPIC_MODEL")

class ClaudePlayer(Anthropic):

    moves = []

    def make_move(self, move: str):
        self.moves.append(f"{HUMAN_PROMPT} {move}")
        completion = self.completions.create(
            model="claude-2.1",
            max_tokens_to_sample=200,
            prompt='\n\n'.join(self.moves) + AI_PROMPT,
        )
        print(f"ClaudePlayer: {completion.completion}")
        self.moves.append(f"{AI_PROMPT} {completion.completion}")
        return completion.completion