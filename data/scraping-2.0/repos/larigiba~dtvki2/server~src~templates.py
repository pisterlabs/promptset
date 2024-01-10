from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate

# condense question and chat history into single question
_condense_question_prompt_template = """Angesichts eines Chatverlaufs und der letzten Benutzerfrage \
die sich auf den Chatverlauf beziehen könnte, formulieren Sie eine eigenständige Frage \
die ohne den Chatverlauf verstanden werden kann. Beantworten Sie die Frage NICHT, \
formulieren Sie die Frage nur um, wenn es nötig ist, und geben Sie die Frage ansonsten unverändert zurück.

Chatverlauf:
{chat_history}
Letzte Benutzerfrage: {question}
Eingenständige Frage:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    _condense_question_prompt_template
)

# default document prompt
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

# answer prompt
_answer_prompt_template = """Sie sind ein Assistent für die Beantwortung von Fragen zu einem Softwareprogramm für Steuerberater. \
Beantworten Sie die Frage mit Hilfe der folgenden Kontextinformationen. \
Wenn Sie die Antwort nicht wissen, sagen Sie einfach, dass Sie es nicht wissen. \
Beschreiben Sie Schritt für Schritt, wie der Nutzer vorgehen soll, um das Problem zu lösen. \

Kontext: {context}

Frage: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(_answer_prompt_template)
