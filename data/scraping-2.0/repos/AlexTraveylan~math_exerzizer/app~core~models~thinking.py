"""Thinking model"""
from io import StringIO
import logging
from app.core.app_types import OpenAiMessage
from app.core.constants import SEPARATOR
from app.core.models.completion import completion

logger = logging.getLogger(__name__)


class Thinking:
    """Force openai to think and produce a beter exercice"""

    def __init__(self, initial_prompt: list[OpenAiMessage], generated_exercice: str):
        self.initial_prompt = initial_prompt
        self.generated_exercice = generated_exercice

    def thinking_prompt(self) -> OpenAiMessage:
        """Return a thinking prompt"""

        thinking_prompt = StringIO()
        thinking_prompt.write(f"TON ROLE\n{SEPARATOR}")
        thinking_prompt.write("Tu es un élève de 4eme au collège\n")
        thinking_prompt.write("Tu as un regard critique\n")
        thinking_prompt.write(
            "Ton but est de donner ton point de vue critique sur un exercice de mathématiques\n"
        )
        thinking_prompt.write("Dans l'objectif que ton professeur l'améliore\n")
        thinking_prompt.write(f"EXERCICE\n{SEPARATOR}")
        thinking_prompt.write(f"{self.generated_exercice}\n")
        thinking_prompt.write(f"REFLECHI\n{SEPARATOR}")
        thinking_prompt.write("Est-ce que tu comprends l'énoncé et les questions ?\n")
        thinking_prompt.write("Es-tu d'accord avec les réponses ?\n")
        thinking_prompt.write(
            "D'autres réponses sont-elles possibles pouvant créer une confusion ?\n"
        )
        thinking_prompt.write(
            "Est-ce qu'une confusion est-elle possible sur les unités ?\n"
        )
        thinking_prompt.write("Les questions sont elles suffisamment explicites ?\n")
        thinking_prompt.write("Chaque question demande-t-elle une unique réponse ?\n")
        thinking_prompt.write(
            "Ces réponses sont-elles uniquement des entiers ou des décimaux ?\n"
        )

        return OpenAiMessage(role="system", content=thinking_prompt.getvalue())

    def get_reflexion(self, thinking_prompt: OpenAiMessage) -> OpenAiMessage:
        """Return a reflexion prompt"""

        response = completion([thinking_prompt])

        if response is None:
            raise ValueError("OpenAI API is not responding")

        return OpenAiMessage(role="assistant", content=response)

    def get_reformat(self, reflexion: OpenAiMessage) -> str:
        """Return a reformat prompt"""

        prompt = [*self.initial_prompt]
        exercice_p = OpenAiMessage(role="system", content=self.generated_exercice)
        prompt.append(exercice_p)

        new_system_prompt = StringIO()
        new_system_prompt.write(f"CONSIGNE SUPPLEMENTAIRE\n{SEPARATOR}")
        new_system_prompt.write("Voici les remarques d'un eleve de 4eme :\n")
        new_system_prompt.write("prends en compte ses remarques\n")
        new_system_prompt.write("et reformule ton exercice si necessaire\n")
        new_system_prompt.write(f"REMARQUES DE L'ELEVE\n{SEPARATOR}")
        new_system_prompt.write(f"{reflexion.get('content')}\n")

        prompt.append(
            OpenAiMessage(role="system", content=new_system_prompt.getvalue())
        )

        response = completion(prompt)

        return response
