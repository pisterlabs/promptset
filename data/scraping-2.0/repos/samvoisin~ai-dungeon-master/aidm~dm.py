import json
from pathlib import Path
from typing import Dict, Optional, Union

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential


class AIDungeonMaster:
    def __init__(
        self, system_prompt_path: Path, model_kwargs: Optional[Dict] = None
    ) -> None:
        # set up prompt
        with open(system_prompt_path, "r") as fh:
            system_prompt_text = fh.read()

        system_prompt_text += "\n\n{history}"
        system_prompt = SystemMessagePromptTemplate.from_template(
            template=system_prompt_text, input_variables=["history"]
        )
        user_prompt = HumanMessagePromptTemplate.from_template("```{message}```")
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [system_prompt, user_prompt]
        )

        # model parameters
        self.model_kwargs = model_kwargs or dict(
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
            frequency_penalty=0.2,
        )

        # set up model and chain
        llm = ChatOpenAI(
            model="gpt-4", **self.model_kwargs
        )


        memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=8192 - 500)
        self.chain = LLMChain(
            llm=llm,
            prompt=self.chat_prompt,
            verbose=True,
            memory=memory,
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def query_model(self, message: str) -> str:
        return self.chain.predict(message=message)

    def save_conversation_history(self, save_file: Union[Path, str]= "conversation_history.json") -> None:

        if self.chain.memory is None:
            raise ValueError("Cannot save conversation history without a memory buffer")

        chat_history = self.chain.memory.dict()

        # currently only safe to save the conversation history in the save_files directory
        save_dir = Path("./save_files")
        save_path = save_dir / save_file
        with open(save_path, "w") as fh:
            json.dump(chat_history, fh)

    @classmethod
    def construct_from_history(
        cls, system_prompt_path: Path, history_path: Path
    ) -> "AIDungeonMaster":
        with open(history_path, "r") as fh:
            conversation_history = json.load(fh)

        aidm = cls(system_prompt_path=system_prompt_path)

        memory = ConversationTokenBufferMemory.construct(conversation_history)
        aidm.chain.memory = memory

        return aidm
