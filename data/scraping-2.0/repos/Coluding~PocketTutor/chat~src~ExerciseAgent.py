from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import langchain

from CustomExceptions import PrivateAttributeException


class ExerciseAgent:
    def __init__(self, model: langchain.llms) -> None:

        self.template = """Du bist ein Tutor für Mathe Schüler der 12. Klasse am Gymnasium in Deutschland. Sie werden dir eine Frage stellen,
                bitte achte darauf, nicht die Lösung direkt zu sagen sondern sie Schritt für Schritt zu erarbeiten. 
                Es geht um folgende Aufgabe: ###{aufgabe}###. Sie hat folgende Lösung ###{loesung}###. {verlauf}.
                Antworte auf die Frage kurz, klar und verständlich: {frage}"""

        self.__message_template = None
        self.__prompt_template = PromptTemplate(
            input_variables=["aufgabe", "loesung", "frage", "verlauf"],
            template=self.template)

        self.__model: langchain.llm = model
        self.__chain: langchain.chains = LLMChain(llm=model, prompt=self.__prompt_template)
        self._verlauf = ""

    @property
    def verlauf(self):
        raise PrivateAttributeException("You are not allowed to access!")

    @verlauf.setter
    def verlauf(self, verlauf_input: str):
        raise PrivateAttributeException("Do not set verlauf attribute!")

    def run(self, aufgabe: str, loesung: str, frage: str) -> str:
        result: str = self.__chain.run({"aufgabe": aufgabe, "loesung": loesung,
                                        "frage": frage, "verlauf": self._verlauf})
        self._set_verlauf(frage, result)
        return result

    def _set_verlauf(self, frage: str, antwort: str) -> None:
        self._verlauf = ("Du hattest folgende Konversation bis jetzt" + "Frage des Schülers: " + frage +
                        ". Deine Antwort: " + antwort + ".")





