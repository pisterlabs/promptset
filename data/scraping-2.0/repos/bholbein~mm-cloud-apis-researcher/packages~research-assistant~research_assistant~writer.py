from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import ConfigurableField

WRITER_SYSTEM_PROMPT = "Du bist ein KI-Forschungsassistent für kritisches Denken. Dein einziger Zweck ist es, gut geschriebene, kritisch anerkannte, objektive und strukturierte Berichte zu vorgegebenen Texten zu verfassen."

# Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
RESEARCH_REPORT_TEMPLATE = """Information: 
--------
{research_summary}
--------

Mit den oben genannten Informationen beantworte die folgende Frage oder das Thema: "{question}" in einem ausführlichen Bericht --
Der Bericht sollte sich auf die Antwort der Frage konzentrieren, gut strukturiert, informativ,
tiefgehend sein, mit Fakten und Zahlen, falls verfügbar, und mindestens 1.200 Wörter umfassen.

Du solltest dich bemühen, den Bericht so lang wie möglich zu schreiben, unter Verwendung aller relevanten und notwendigen Informationen.
Du musst den Bericht in Markdown-Syntax verfassen.
Du MUSST deine eigene konkrete und valide Meinung auf Basis der gegebenen Informationen bilden. Weiche NICHT zu allgemeinen und bedeutungslosen Schlussfolgerungen ab.
Schreibe alle verwendeten Quellen-URLs am Ende des Berichts und achte darauf, keine doppelten Quellen hinzuzufügen, sondern nur einen Verweis für jede.
Du musst den Bericht im APA-Format verfassen.
Bitte gib dein Bestes, das ist sehr wichtig für meine Karriere."""  # noqa: E501


RESOURCE_REPORT_TEMPLATE = """Information: 
--------
{research_summary}
--------

Basierend auf den oben genannten Informationen, erstelle einen Empfehlungsbericht für Bibliografie für die folgende Frage oder das Thema: "{question}".
Der Bericht sollte eine detaillierte Analyse jeder empfohlenen Ressource bieten, wobei erläutert wird, wie jede Quelle zur Beantwortung der Forschungsfrage beitragen kann.
Konzentriere dich auf die Relevanz, Zuverlässigkeit und Bedeutung jeder Quelle.
Stelle sicher, dass der Bericht gut strukturiert, informativ, tiefgehend ist und der Markdown-Syntax folgt.
Schließe relevante Fakten, Zahlen und Daten ein, wann immer verfügbar.
Der Bericht sollte eine Mindestlänge von 1.200 Wörtern haben.

Bitte gib dein Bestes, das ist sehr wichtig für meine Karriere."""  # noqa: E501

OUTLINE_REPORT_TEMPLATE = """Information: 
--------
{research_summary}
--------

Mit den oben genannten Informationen erstelle ein Gerüst für einen Forschungsbericht in Markdown-Syntax für die folgende Frage oder das Thema: "{question}".
Das Gerüst sollte einen gut strukturierten Rahmen für den Forschungsbericht bieten, einschließlich der Hauptabschnitte, Unterabschnitte und der wichtigsten zu behandelnden Punkte.
Der Forschungsbericht sollte detailliert, informativ, tiefgehend sein und mindestens 1.200 Wörter umfassen.
Verwende die entsprechende Markdown-Syntax, um das Gerüst zu formatieren und die Lesbarkeit zu gewährleisten.

Bitte gib dein Bestes, das ist sehr wichtig für meine Karriere.."""  # noqa: E501

model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
).configurable_alternatives(
    ConfigurableField("report_type"),
    default_key="research_report",
    resource_report=ChatPromptTemplate.from_messages(
        [
            ("system", WRITER_SYSTEM_PROMPT),
            ("user", RESOURCE_REPORT_TEMPLATE),
        ]
    ),
    outline_report=ChatPromptTemplate.from_messages(
        [
            ("system", WRITER_SYSTEM_PROMPT),
            ("user", OUTLINE_REPORT_TEMPLATE),
        ]
    ),
)
chain = prompt | model | StrOutputParser()
