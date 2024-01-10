"""
This Module contains the logic and prompts for the task of 
listing technical features of a product using the Google Search API as a default
"""

import json
import os
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from modules.get_page_html import parse_results
from modules.get_completion import get_completion
from models.data_model import data_model

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


def list_technical_features(part_number):
    """
    This function lists the technical features of a product.
    Specified by the part number of the product.   
    """

    search = GoogleSearchAPIWrapper(k=10)
    tool = Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=search.run,
    )

    parsed_result = parse_results(part_number)

    source = parsed_result["source"]
    result = parsed_result["content"]

    if result == "":
        source = "google"
        result = tool.run(f"{part_number}")

    prompt = f"""
    Deine Aufgabe ist es, technische Eigenschaften des Produktes '{part_number}' aufzulisten. D

    Die Ausgabe muss zwingend im validen JSON Format erfolgen.

    Bitte befolge die folgenden Schritte:
    1. Kategorisiere das Produkt als einen der folgenden Typen:
    - Kondensator
    - Optokoppler
    - Widerstand
    - Taste
    - Schraube
    - Stecker
    - Niete
    - Diode
    - Relay
    - Hülse
    Falls keine eindeutige Zuordnung möglich ist, verwende den Wert "Keine Angaben".

    2. Füge für die Produktgruppe relevante Attribute hinzu. Mindestens laut Datenmodel: {data_model} Beachte, dass alle Angaben inklusive der Attributswerte in Deutsch sein müssen.

    3. Beachte, dass die Montage nur die Werte "TH" oder "SMD" haben darf.

    4. Nutze den Kontext: *** {result} ***

    5. Gib als Quelle unbedingt {source} an

    6. Wenn du dir nicht sicher bist, gib "Keine Angaben" aus.

    Beispiel für die JSON-Ausgabe:
    {{
    "Produkt": "{part_number}",
    "Hersteller": "Yatego",
    "Quelle": "{source}",
    "Produktgruppe": "Kondensator",
    "Attribute": {{
        "Kapazität": "10 µF",
        "Montage": "TH",
        "Bemessungsspannung": "30 VAC/60 VDC",
        "Nennspannung": "50 V",
        "Ausführung": "Tantal",
        "Abmessungen": "5 mm x 11 mm"
    }}
    }}
    """
    completion = get_completion(prompt)

    try:
        completion_json = json.loads(completion)
    except json.JSONDecodeError:
        print(completion)
        return {"error": "Failed to parse JSON. Please check your inputs."}

    return {"message": completion_json}
