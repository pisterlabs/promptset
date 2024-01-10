import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

import os
import dotenv
dotenv.load_dotenv()
OPEN_AI_KEY = os.getenv('OPENAI_API_KEY')


class QualityAssuranceAgent():
    def __init__(self):
        self.client = Anthropic(api_key=OPEN_AI_KEY)

    def check_quality(
        self,
        chat_log: str,
        conclusion: str
    ) -> str:
        completion = self.client.completions.create(
            model="claude-instant-1",
            max_tokens_to_sample=300,
            prompt=f"{HUMAN_PROMPT} You are a quality assurance analyst. Given the data from a chat log and the conclusion, return 'APPROVED' if you see no errors. \nCHATLOGS:{chat_log}\nCONCLUSIONS{conclusion}{AI_PROMPT}",
        )

        return completion.completion.strip()
