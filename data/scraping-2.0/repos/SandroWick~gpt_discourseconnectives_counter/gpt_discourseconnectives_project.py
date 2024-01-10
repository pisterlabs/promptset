import openai
import csv
import argparse
from collections import Counter
from typing import List
from data.discourse_connectors import discourse_connectors

# Ihr OpenAI GPT-3 API-Schlüssel
api_key = "[insert your API KEY here]"

def parse_arguments() -> argparse.Namespace:
    """CLI-Argumente parsen."""
    parser = argparse.ArgumentParser(description='Vergleicht die Häufigkeit von Diskursmarkern in Artikeln und GPT-3 Texten.')
    parser.add_argument('korpus', help='Pfad zum TSV-Korpus')
    parser.add_argument('--anzahl_artikel', type=int, default=10, help='Anzahl der zu vergleichenden Artikel')
    return parser.parse_args()

def count_connectors(text: str, connectors, List) -> Counter:
    """Zählt die Diskursmarker im Text."""
    words = text.lower().split()
    return Counter([word for word in words if word in connectors])

def get_gpt_text(prompt: str, token_limit: int) -> str:
    """Holt den generierten Text von GPT-3."""
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=token_limit
    )
    return response.choices[0].text.strip()

def trim_to_same_length(article_text: str, gpt_text: str) -> (str, str):
    """Kürzt die Texte auf die gleiche Länge."""
    token_limit = min(len(article_text.split()), len(gpt_text.split()))
    return ' '.join(article_text.split()[:token_limit]), ' '.join(gpt_text.split()[:token_limit])

def sum_total_connectors(counter: Counter) -> int:
    """Berechnet die Gesamtanzahl der Diskursmarker in einem Counter."""
    return sum(counter.values())

def main():
    """
    Pseudo-Code für main():

    1. CLI-Argumente parsen.
    2. Öffne den Korpus und lese die angegebene Anzahl an Artikeln.
    3. Für jeden Artikel:
        a. Nutze die ersten 10 Sätze als Prompt für GPT-3.
        b. Generiere den GPT-3 Text.
        c. Kürze Artikel und GPT-3 Text auf die gleiche Länge.
        d. Zähle die Diskursmarker in beiden Texten.
        e. Schreibe die Ergebnisse in die CSV-Datei.
    4. Berechne die Durchschnittswerte und gib sie im Terminal aus.
    """

    args = parse_arguments()

    # Ausgabe-CSV-Datei vorbereiten
    with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Head', 'Article_Connectors', 'Article_Text', 'GPT_Connectors', 'GPT_Text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        article_connector_totals = []
        gpt_connector_totals = []

        # Korpus öffnen und Artikel lesen
        with open(args.korpus, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for i, row in enumerate(reader):
                if i >= args.anzahl_artikel:
                    break

                # Prompt für GPT-3 erstellen
                article_text = row['content']
                head = row['head']
                prompt = '. '.join(article_text.split('. ')[:10])
                
                # GPT-3 Text generieren
                token_limit = len(article_text.split())
                gpt_text = get_gpt_text(prompt, token_limit)
                
                # Texte auf gleiche Länge kürzen
                article_text, gpt_text = trim_to_same_length(article_text, gpt_text)
                            
               # Diskursmarker zählen
                article_counts = count_connectors(article_text, discourse_connectors)
                gpt_counts = count_connectors(gpt_text, discourse_connectors)

                article_total = sum_total_connectors(article_counts)
                gpt_total = sum_total_connectors(gpt_counts)

                # Ergebnisse in der CSV-Datei speichern
                writer.writerow({'Head': head, 'Article_Connectors': article_total, 'Article_Text': article_text, 'GPT_Connectors': gpt_total, 'GPT_Text': gpt_text})

                article_connector_totals.append(article_total)
                gpt_connector_totals.append(gpt_total)

        # Durchschnittswerte berechnen
        avg_article_total = sum(article_connector_totals) / args.anzahl_artikel
        avg_gpt_total = sum(gpt_connector_totals) / args.anzahl_artikel


        # Durchschnittswerte im Terminal ausgeben
        print(f"Durchschnittliche Diskursmarker im Artikel: {avg_article_total}")
        print(f"Durchschnittliche Diskursmarker im GPT-Text: {avg_gpt_total}")
        

if __name__ == '__main__':
    main()

"""
Output:
Durchschnittliche Diskursmarker im Artikel: 26.1
Durchschnittliche Diskursmarker im GPT-Text: 24.6
"""