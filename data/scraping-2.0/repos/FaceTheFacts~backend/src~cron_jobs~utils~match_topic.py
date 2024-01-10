import os
import openai

topics_dict = {
    "1": "Medien, Kommunikation und Informationstechnik",
    "2": "Arbeit und Beschäftigung",
    "3": "Bildung und Erziehung",
    "4": "Europapolitik und Europäische Union",
    "5": "Landwirtschaft und Ernährung",
    "6": "Parlamentsangelegenheiten",
    "7": "Kultur",
    "8": "Recht",
    "9": "Umwelt",
    "10": "Verkehr",
    "11": "Außenwirtschaft",
    "12": "Sport, Freizeit und Tourismus",
    "13": "Verteidigung",
    "14": "Soziale Sicherung",
    "15": "Wissenschaft, Forschung und Technologie",
    "16": "Gesellschaftspolitik, soziale Gruppen",
    "17": "Entwicklungspolitik",
    "18": "Raumordnung, Bau- und Wohnungswesen",
    "19": "Wirtschaft",
    "20": "Energie",
    "21": "Außenpolitik und internationale Beziehungen",
    "22": "Öffentliche Finanzen, Steuern und Abgaben",
    "23": "Innere Sicherheit",
    "24": "Staat und Verwaltung",
    "25": "Ausländerpolitik, Zuwanderung",
    "26": "Neue Bundesländer",
    "27": "Politisches Leben, Parteien",
    "28": "Gesundheit",
    "29": "Geschäftsordnung",
    "30": "Immunität",
    "31": "Petitionen",
    "32": "Wahlprüfung",
    "33": "Humanitäre Hilfe",
    "34": "Familie",
    "35": "Frauen",
    "36": "Jugend",
    "37": "Senioren",
    "38": "Innere Angelegenheiten",
    "39": "Digitale Agenda",
    "40": "digitale Infrastruktur",
    "41": "Finanzen",
    "42": "Haushalt",
    "43": "Menschenrechte",
    "44": "Verbraucherschutz",
    "45": "Medien",
    "46": "Sport",
    "47": "Tourismus",
    "48": "Naturschutz",
    "49": "Forschung",
    "50": "Reaktorsicherheit",
    "51": "Technologiefolgenabschätzung",
    "52": "Regionales",
    "53": "Lobbyismus & Transparenz",
    "2900": "Klima",
}


def match_topic(statement, topic):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    labels = [f"{key}: {value}" for key, value in topics_dict.items()]
    prompt = f"Which topic best matches the statement: '{topic}: {statement}'? Options: {', '.join(labels)} \n\nReturn only the Option number (e.g. 1)"
    response = openai.Completion.create(
        model="text-davinci-003", prompt=prompt, max_tokens=5, temperature=0.0
    )
    matched_text = response.choices[0].text.strip()
    matched_id = int(matched_text.strip())

    if str(matched_id) in topics_dict:
        return matched_id
    else:
        raise ValueError(
            f"Matched ID {matched_id} is not in the predefined topics dictionary."
        )
