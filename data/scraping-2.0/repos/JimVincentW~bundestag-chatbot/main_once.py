import os
from langchain.chat_models import ChatOpenAI
import fitz


document_questions = {
    "Vorgang": [
        "Was ist der Hauptzweck dieses Vorgangs?",
        "Welche Parteien oder Organisationen sind an diesem Vorgang beteiligt?",
        "Was sind die Schlüsseldaten im Zusammenhang mit diesem Vorgang?",
        "Welche Schritte wurden bisher in diesem Vorgang unternommen?",
        "Welche rechtlichen oder politischen Auswirkungen hat dieser Vorgang?",
        "Welche Behauptungen, Formulierungen und rhetorischen Stilmittel sollten aus einer kritischen Perspektive beachtet werden?"
        ],
    "Gesetzesentwurf": [
            "Was ist das Hauptziel des Gesetzesentwurfs?",
            "Wer hat den Gesetzesentwurf eingereicht?",
            "Welche Änderungen werden durch den Gesetzesentwurf vorgeschlagen?",
            "Wie wirkt sich der Gesetzesentwurf auf bestehende Gesetze aus?",
            "Gibt es Kontroversen oder Meinungsverschiedenheiten in Bezug auf den Gesetzesentwurf?",
            "Was für einen Ton hat das Dokument?", 
            "Was für sprachliche und rhetorische Mittel werden verwendet?"
        ],
    "Antrag": [
        "Was ist der Hauptzweck des Antrags?",
        "Wer hat den Antrag gestellt?",
        "Welche Maßnahmen oder Entscheidungen werden im Antrag vorgeschlagen?",
        "Welche Argumente werden im Antrag vorgebracht?",
        "Was sind die potenziellen Auswirkungen des Antrags?",
        "Welche Behauptungen, Formulierungen und rhetorischen Stilmittel sollten aus einer kritischen Perspektive beachtet werden?"
    ],
     "Unterrichtung": [
            "Welche Informationen werden in der Unterrichtung mitgeteilt?",
            "Wer ist der Absender der Unterrichtung?",
            "An wen ist die Unterrichtung gerichtet?",
            "Was sind die Schlüsselpunkte der Unterrichtung?",
            "Welche Handlungen oder Entscheidungen werden in der Unterrichtung empfohlen?",
            "Was für einen Ton hat das Dokument?", 
            "Was für sprachliche und rhetorische Mittel werden verwendet?"
        ],
    "Stellungnahme": [
        "Wer hat die Stellungnahme abgegeben?",
        "Auf welches Thema oder welche Angelegenheit bezieht sich die Stellungnahme?",
        "Was sind die Hauptpunkte der Stellungnahme?",
        "Gibt es Meinungsverschiedenheiten oder Kontroversen in der Stellungnahme?",
        "Welche Handlungen oder Entscheidungen werden in der Stellungnahme empfohlen?",
        "Was für einen Ton hat das Dokument?", 
        "Was für sprachliche und rhetorische Mittel werden verwendet?"
    ],
    "Beschlussempfehlung": [
        "Was ist der Hauptzweck der Beschlussempfehlung?",
        "Wer hat die Beschlussempfehlung abgegeben?",
        "Welche Entscheidungen oder Maßnahmen werden in der Beschlussempfehlung vorgeschlagen?",
        "Was sind die Gründe für die in der Beschlussempfehlung vorgeschlagenen Maßnahmen?",
        "Welche Auswirkungen könnten die vorgeschlagenen Maßnahmen haben?"
    ],
    "Bericht": [
        "Was ist das Hauptthema des Berichts?",
        "Wer hat den Bericht verfasst?",
        "Welche Schlüsselinformationen enthält der Bericht?",
        "Welche Schlussfolgerungen werden im Bericht gezogen?",
        "Welche Empfehlungen werden im Bericht gegeben?"
    ],
    "Kleine": [
            "Was ist das Hauptthema des Berichts?",
            "Wer hat den Bericht verfasst?",
            "Welche Schlüsselinformationen enthält der Bericht?",
            "Welche Schlussfolgerungen werden im Bericht gezogen?",
            "Welche Empfehlungen werden im Bericht gegeben?",
            "Was für einen Ton hat das Dokument?", 
            "Was für sprachliche und rhetorische Mittel werden verwendet?"
        ]
}

roles_templates = {
    "role": ["referent", "journalist"],
    "team_role": "teamleiter",
    "action": "zusammenfassen",
    "input": ["Titel", "Struturzeilen", "Keywords", "Text"],
    "output": ["Keywords", "Topics", "Summary", "Important Sections", "OpenBookMemory"],
    "special_instruction": ["CreateOpenBookMemory", "Structurise"]

}

referent_leiter = {
    "role": "referent",
    "team_role": "teamleiter",
    "action": "zusammenfassen",
    "input": ["Titel", "Struturzeilen", "Keywords", "Text"],
    "output": ["Keywords", "Topics", "Summary", "Important Sections", "OpenBookMemory"],
    "special_instruction": ["CreateOpenBookMemory", "Structurise"]
}





class LegislativeDocumentProcessor:
    def __init__(self, document, api_key=None):

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.document = document

    def _build_prompt(self, content):
        instructions = {
            "role": ["referent", "journalist"],
            "team_role": "teamleiter",
            "action": "zusammenfassen",
            "input": ["Titel", "Struturzeilen", "Keywords", "Text"],
            "output": ["Keywords", "Topics", "Summary", "Important Sections", "OpenBookMemory"],
            "special_instruction": ["CreateOpenBookMemory", "Structurise"]
        }

        prompt = f"""
        Du bisst {instructions['role'][1]} des Bundestags.
        Du arbeitest in einem Teil und deine Rolle ist {instructions['team_role']}.
        
        Der name des aktuellen Dokuments ist {self.document["title"]}.
        
        Dafür hast du folgendes Dokument zur Verfügung:
        {instructions["input"][3]}: {content}.
         eines Teams.
        Deine Rolle ist {instructions["team_role"]}.
        
        Bitte fasse auch für das dynamische OpenBookMemory zusammen.
        """
        return prompt

        ("system", f"Du bist {instructions['role'][1]} des Bundestags."),
        ("human", f"""Du arbeitest in einem Teil und deine Rolle ist {instructions['team_role']}.
        Der name des aktuellen Dokuments ist {self.document["title"]}.
        Du bist außerdem dafür verantwortlich den ersten Eintrag für das "OpenBookMemory" zu schreiben. 
        Aufgrund dieser Arbeitsvorlage werden deine Teammitglieder die Arbeit fortsetzen. 
         """),
        ("ai", "Alles klar, was steht in dem Dokument?"),
        ("human", "Das Dokument: {document}")
    ]


    def process_document(self, content):
        model = ChatOpenAI(
            temperature=0,
            openai_api_key=self.api_key,
            model="gpt-3.5-turbo-0613"
        )
        prompt = self._build_prompt(content)
        response = model.generate(prompt)
        return response


sample_document = {
    "title": "Gesetz zur Änderung des Lobbyregister- und des Lobbyistenverhaltensgesetzes",
    "type": "Vorgang",
    "id": 300955,
    "intiative": ["gruene", "spd", "fdp"],
    "important_files": [
        "BT-Drucksache 20/7356",
        "1. Beratung BT-Plenarprotokoll 20/113, S. 13947C-13959B"
    ],
    "content": "..."
}
document = sample_document["content"] + " " + sample_document["title"] + " " + sample_document["important_files"][0] + " " + sample_document["important_files"][1] 



processor = LegislativeDocumentProcessor(sample_document)
summary = processor.process_document(sample_document["content"])
print(summary)


import os
from langchain.chat_models import ChatOpenAI
import fitz

class LegislativeDocumentProcessor:

    def __init__(self, document):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")
        self.document = document
        self.model = ChatOpenAI(
            temperature=0,
            openai_api_key=self.api_key,
            model="gpt-3.5-turbo-0613"
        )
        self.openbookmemory = []

    def ModelA(self, content):
        # ... Your Model A Code Here ...
        # After processing:
        return summary, self.openbookmemory

    def ModelB(self, content):
        # Deep analysis of the document and builds OpenBookMemory
        # ...
        return deep_summary, cross_references, notes, self.openbookmemory

    def ModelC(self, content):
        # Contextual analysis and relating content to existing knowledge
        # ...
        return contextual_notes, comparative_analysis, potential_implications, self.openbookmemory

    def ModelD(self, content):
        # Quality assurance and error correction
        # ...
        return refined_keywords, refined_summary, refined_notes, self.openbookmemory

    def ModelE(self, content):
        # Presentation and visualization
        # ...
        return graphical_data, infographics, interactive_dashboards, self.openbookmemory

    def process_document(self):
        content = self.document['content']

        # Model A processing
        summary_a, self.openbookmemory = self.ModelA(content)

        # Model B processing
        deep_summary, cross_references, notes, self.openbookmemory = self.ModelB(content)

        # Model C processing
        contextual_notes, comparative_analysis, potential_implications, self.openbookmemory = self.ModelC(content)

        # Model D processing
        refined_keywords, refined_summary, refined_notes, self.openbookmemory = self.ModelD(content)

        # Model E processing
        graphical_data, infographics, interactive_dashboards, self.openbookmemory = self.ModelE(content)

        return {
            "summary": summary_a,
            "deep_summary": deep_summary,
            "contextual_notes": contextual_notes,
            "refined_summary": refined_summary,
            "graphical_data": graphical_data,
            "openbookmemory": self.openbookmemory
        }



sample_document = {
    "type": "Vorgang",
    "id": 300955,
    "intiative": ["gruene", "spd", "fdp"],
    "important_files": [
        "BT-Drucksache 20/7356",
        "1. Beratung BT-Plenarprotokoll 20/113, S. 13947C-13959B"
    ],
    "content": "Just some random text for now."
}

processor = LegislativeDocumentProcessor(sample_document)
result = processor.process_document()

print(result)






# https://dip.bundestag.de/vorgang/bezahlbare-und-klimafreundliche-mobilit%C3%A4t-f%C3%B6rdern-klimaneutrale-kraftstoffe-im-verkehr/296949?term=296949&f.wahlperiode=20&rows=25&pos=1