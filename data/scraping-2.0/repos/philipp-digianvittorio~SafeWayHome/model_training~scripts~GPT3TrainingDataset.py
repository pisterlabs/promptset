
import pandas as pd
import openai
from scripts.SQLAlchemyDB import db_select

articles = db_select("Articles")

api_key = "#"

def GPT_Completion(api_key, prompt, max_tokens=256):
    openai.api_key = api_key
    response = openai.Completion.create(engine="text-davinci-003",
                                        prompt=prompt,
                                        temperature=0.6,
                                        top_p=1,
                                        max_tokens=max_tokens,
                                        frequency_penalty=0,
                                        presence_penalty=0)
    print(response.choices)
    return response.choices[0].text



task = "Ordne für jeden Vorfall die Straftat einer der folgenden Klassen zu - Betrug, Diebstahl, Hausfriedensbruch, Einbruch, Raub, schwerer Raub, Erpressung, Verkehrsunfall, Verkehrsstraftat, Drogenhandel, Drogenbesitz, Waffenbesitz, Sachbeschädigung, Brandstiftung, fahrlässige Körperverletzung, Körperverletzung, gefährliche Körperverletzung, schwere Körperverletzung, Bedrohung, Widerstand, Exhibitionismus, sexuelle Belästigung, sexueller Übergriff, Vergewaltigung, Beleidigung, Tötungsdelikt, Sonstiges. Benenne den Haupttatort (Straße) und den Beginn der Tatzeit (Uhrzeit) und gib an, ob die Straftat in einem Gebäude begangen wurde: "

ex1 = '''Text: (th) Am Dienstag (22.11.2022) wurden zwei Crackdealer nach beobachteten Verkäufen festgenommen.
Vormittags folgten zivile Polizeibeamte zwei Drogenkonsumenten vom Bahnhofsgebiet zum Schweizer Platz, wo sie auf ihren Dealer trafen und ca. 0,25 Gramm Crack in Empfang nahmen. Die beiden Käufer wurden vor Ort kontrolliert, der 42-jährige wohnsitzlose Dealer im Bereich Eschenheimer Tor festgenommen. Er führte weitere ca. 0,5 Gramm Crack bei sich, welche er versuchte zu schlucken. Es folgte die Einlieferung in das Zentrale Polizeigewahrsam zwecks Prüfung einer richterlichen Vorführung.
Gegen 18:00 Uhr wurden Polizeibeamte auf einen weiteren Dealer im Bereich Am Hauptbahnhof aufmerksam. Sie identifizierten ihn als Verkäufer aus einem wenige Tage zuvor beobachteten Drogenhandel. Gegen den 42-Jährigen bestand zudem ein offener Haftbefehl.'''
result1 = '''[{'crime': ['Drogenhandel',], 'location': 'Schweizer Platz', 'time': 'Vormittag', 'indoors': False}, {'crime': ['Drogenhandel',], 'location': 'Am Hauptbahnhof', 'time': '18:00 Uhr', 'indoors': False}]'''

ex2 = '''Text: (dr) Eine Polizeistreife des 4. Reviers nahm am gestrigen Sonntag, den 20. November 2022, einen 19-Jährigen im Gutleutviertel fest, der sich bei einer Personenkontrolle besonders aggressiv zeigte. Bei ihm stellten sie auch Rauschgift sicher.
Eine Ruhestörung in der Gutleutstraße führte gegen 22:10 Uhr zu einer Personenkontrolle eines 19-Jährigen. Der junge Mann war offensichtlich nicht mit der polizeilichen Maßnahme einverstanden und machte dies deutlich, indem er Tritte und Schläge gegen die ihn kontrollierenden Beamten austeilte. Währenddessen versuchte er auch immer wieder ein Einhandmesser aus seiner Jackentasche zu ziehen, was jedoch unterbunden werden konnte. Den Beamten gelang es, den 19-Jährigen unter Widerstand festzunehmen. Als sie ihn durchsuchten, stießen sie auf Betäubungsmittel, darunter rund 90 Gramm Amphetamin und über 90 Ecstasy-Tabletten. Bei einer anschließenden Durchsuchung an der Anschrift seiner Eltern fanden die Beamten in seinem "Kinderzimmer" weitere Substanzen zur Herstellung von Drogen auf sowie verbotene Gegenstände. Sie stellten alle Beweismittel sicher.
Für den 19-Jährigen, welcher über keinen festen Wohnsitz verfügt, ging es in der Folge in die Haftzellen. Ihn erwartet nun ein Strafverfahren wegen des Verdachts des illegalen Drogenhandels und des Widerstands gegen Vollstreckungsbeamte. Er soll heute dem Haftrichter vorgeführt werden.'''
result2 = '''[{'crime': ['Sonstiges', 'Drogenhandel', 'Widerstand',], 'location': 'Gutleutstraße', 'time': '22:10 Uhr', 'indoors': False}]'''

ex3 = '''Text: (wie) Ein berauschter Autofahrer ohne Führerschein ist in der Nacht von Freitag auf Samstag bei Hattersheim vor der Polizei geflohen, konnte aber festgenommen werden.
Eine Streife der Autobahnpolizei wollte gegen 01:20 Uhr einen blauen Audi kontrollieren, da er mit eingeschalteter Nebelschlussleuchte auf der A 66 unterwegs war. Der Fahrer missachtete allerdings die Anhaltezeichen und wendete sein Fahrzeug, nachdem die Fahrzeuge bei Zeilsheim von der Autobahn abgefahren waren. Der Audi floh durch Zeilsheim und Sindlingen, überholte einen Linienbus mit hoher Geschwindigkeit und gefährdete in der Sindlinger Bahnstraße einen Fußgänger, der gerade einen Zebrastreifen nutzen wollte, aber rechtzeitig auf den Bürgersteig zurücktrat. Die Fahrt ging weiter bis nach Hattersheim, wo auch ein Fußgänger an einem Zebrastreifen gefährdet wurde. Der 18-Jährige aus Straßburg stand offensichtlich unter dem Einfluss von Betäubungsmitteln und war nicht im Besitz einer Fahrerlaubnis.
'''
result3 = '''[{'crime': ['Verkehrsstraftat',], 'location': 'Sindlinger Bahnstraße', 'time': '01:20 Uhr', 'indoors': False}]'''

ex4 = '''Text: (lo) In der heutigen Nacht wurde ein 59-jähriger Mann in der Altstadt von einem bislang unbekannten Täter angegriffen und lebensgefährlich verletzt. Die Polizei hat die Ermittlungen wegen eines versuchten Tötungsdeliktes aufgenommen.
Gegen 00:50 Uhr fanden Passanten den 59-Jährigen stark blutend im Bereich der Neuen Kräme. Der daraufhin alarmierte Rettungswagen verbrachte den Geschädigten in ein umliegendes Krankenhaus. Hier konnten mehrere Einstichstellen im Oberkörper des Geschädigten festgestellt werden. Nach Angaben des Geschädigten befand er sich bis ca. 00.00 Uhr in einer Lokalität am Römerberg. Von hier aus sei er in Richtung Neue Kräme fußläufig unterwegs gewesen.
Die Frankfurter Mordkommission ermittelt nun wegen eines versuchten Tötungsdelikts und sucht weitere Zeugen.'''
result4 = '''[{'crime': ['Tötungsdelikt',], 'location': 'Neue Kräme', 'time': '00:50 Uhr', 'indoors': False}]'''

ex5 = '''Text: POL-F: 221118 - 1336 Frankfurt-Schwanheim: Passanten halten Räuber fest Frankfurt (ots) (dr) In der Nacht von Mittwoch auf Donnerstag kam es in Schwanheim zu einem Straßenraub, bei dem ein 47-jähriger Mann einer 18-Jährigen gewaltsam das Mobiltelefon entwendete. Die 18-jährige Geschädigte und der 47-jährige Beschuldigte befanden sich zunächst in einem Bus der Linie 51 in Richtung Schwanheim. Als der Bus gegen 0:45 Uhr in der Geisenheimer Straße an der Haltestelle Kelsterbach Weg anhielt und die Geschädigte ausstieg, folgte ihr der Beschuldigte. Plötzlich schlug ihr der Mann mit der Faust ins Gesicht, sodass die Geschädigte zu Boden fiel und sich leicht verletzte. Nach dem Sturz entriss ihr der 47-Jährige ihr Mobiltelefon und flüchtete mit diesem in westliche Richtung. Gegen den 47-Jährigen wurde aufgrund des Straßenraubes ein Strafverfahren eingeleitet '''
result5 = '''[{'crime': ['Raub',], 'location': 'Geisenheimer Straße', 'time': '00:45 Uhr', 'indoors': False}]'''

prompt = task + "\n\n" + ex1 + "\n" + result1 + "\n" + "###" + "\n" + ex2 + "\n" + result2 + "\n" + "###" + "\n" + ex3 + "\n" + result3 + "\n" + "###" + "\n" + ex4 + "\n" + result4 + "\n" + "###" + "\n" + ex5 + "\n" + result5 + "\n" + "###" +  "\n"
len(prompt.split(" "))


def extract_crime_data(articles):
    crime_list = list()
    for idx in range(len(articles)):
        text = articles[idx]["headline"] + "\n" + articles[idx]["article"]
        try:
            y = eval(GPT_Completion(api_key, prompt + text))
        except:
            y = [{'crime': [], 'location': None, 'time': None, 'indoors': False}]
        for d in y:
            cl = {"hq_id": articles[idx]["hq_id"],
                  "article_id": articles[idx]["id"],
                  "date": articles[idx]["date"],
                  "crime": d["crime"],
                  "location": d["location"],
                  "time": d["time"],
                  "indoors": d["indoors"]}
            crime_list.append(cl)
    return crime_list


crime_list = extract_crime_data(articles)

hq_id = [x["hq_id"] for x in crime_list]
id_ = [x["article_id"] for x in crime_list]
date = [x["date"] for x in crime_list]
crime = [x["crime"] for x in crime_list]
location = [x["location"] for x in crime_list]
time = [x["time"] for x in crime_list]
indoors = [x["indoors"] for x in crime_list]


df = pd.DataFrame({"hq_id": hq_id,
                  "article_id": id_,
                  "date": date,
                  "crime": crime,
                  "location": location,
                  "time": time,
                  "indoors": indoors})


hq_id = [a["hq_id"] for a in articles3]
id = [a["id"] for a in articles3]
art = [a["headline"] + "\n" + a["article"] for a in articles3]
article_df = pd.DataFrame({"hq_id": hq_id,
                           "article_id": id,
                           "article": art})


dup_crime_ids = df[df["article_id"].duplicated()].index
dup_article_ids = article_df[article_df["article_id"].duplicated()].index


for i in range(len(dup_crime_ids)):
    if (len(dup_article_ids) == 0) or (dup_crime_ids[0] != dup_article_ids[0]):
        article_df = article_df.append(article_df.loc[dup_crime_ids[0]-1]).sort_index().reset_index(drop=True)
        dup_crime_ids = dup_crime_ids[1:]
        dup_article_ids = dup_article_ids + 1
    else:
        dup_crime_ids = dup_crime_ids[1:]
        dup_article_ids = dup_article_ids[1:]


df["article"] = article_df["article"]

df = df[(df["crime"].astype(str) != "[]") & (df["location"].str.len() > 3)]
df["location"] = df["location"].apply(lambda x: x.split("/")[0].split(",")[0])

df.to_excel("gpt_output_all.xlsx", index=False)
