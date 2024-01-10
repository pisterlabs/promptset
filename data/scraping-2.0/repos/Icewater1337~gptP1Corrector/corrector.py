from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


class Corrector():
    def __init__(self, task_dict):
        #llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
        llm = ChatOpenAI(temperature=0.1, model_name="gpt-4")
        prompt = PromptTemplate(
            input_variables=["task", "solution_full"],
            template="An der universitaet haben wir eine Voreslung Programmieren für Beginner. In dieser Vorlesung müssen Übungen gelöst werden."
                     "Gegeben ist folgende Übung: {task}. \n\n"
                     "Bitte bewerte folgende Lösungen anhand der gegebenen übung und schau, dass du zu jeder Aufgabe ein kurzes feedback gibst und "
                     "insbesondere auch sagst ob der Code stimmt. "
                     "Bitte gib auch verbesserungsvorschläge für den Code, sprich wenn man ihn "
                     "z.B. irgendwie effizienter machen kann, oder der Code nicht den Java conventions entspricht. Bitte erkläre auch die Konzepte kurz die du kritisierst.: {solution_full}."
        )
        self.chain = LLMChain(llm=llm, prompt=prompt)
        self.task_text = task_dict["task"]
        self.task_files = task_dict["files"]

    def correct_series(self, java_tasks_dict):
        task_full = ""
        for file_name, file_content in java_tasks_dict.items():
            task_full += f"{file_name}:\n {file_content}\n\n"

        res = self.chain.run(task=self.task_text, solution_full=task_full)
        print(res)
