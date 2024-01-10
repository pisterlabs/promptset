from typing import Any, List, Tuple

import openai

from aialogue.utils import get_logger, join_naturally, pluralise, quote

logger = get_logger(__name__)


class DialogueGenerator:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.system_prompt = "You generate simple dialogues for young children who are starting to learn English as a second language, making sure to include specific keywords."
        logger.info(
            "Initialising DialogueGenerator with model_name: `%s` and system_prompt `%s`.",
            self.model_name,
            self.system_prompt,
        )

    def _generate_user_prompt(self, names: Tuple[str, str], keywords: List[str]) -> str:
        names_joined = join_naturally(list(names))
        keywords_joined = join_naturally([quote(k) for k in keywords])
        keyword_message = pluralise("keyword", len(keywords))
        user_prompt = " ".join(
            [
                f"Generate a dialogue between two people called {names_joined}.",
                f"Include the {keyword_message} {keywords_joined}.",
            ]
        )
        logger.info("Generated user prompt: `%s`.", user_prompt)
        return user_prompt

    def generate_dialogue(self, names: Tuple[str, str], keywords: List[str]) -> str:
        completion: Any = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self._generate_user_prompt(names, keywords),
                },
            ],
        )
        dialogue = completion.choices[0].message.content
        logger.info("Generated dialogue: `%s`.", dialogue)
        return dialogue
