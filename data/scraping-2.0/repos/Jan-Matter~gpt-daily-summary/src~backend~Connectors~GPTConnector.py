from dotenv import load_dotenv, find_dotenv
import openai
import asyncio
import os

load_dotenv(find_dotenv())

class GPTConnector:

    def __init__(self):
        openai.api_key = os.getenv("GPT_API_TOKEN")

    async def get_chat_respone(self, messages, model="gpt-3.5-turbo"):
        response = await openai.ChatCompletion.acreate(
        model=model,
        messages=messages
        )
        return response["choices"][0]["message"]["content"]


async def main():
    messages =[
                {"role": "system", "content": "You are a geopolitical journalist."},
                {"role": "user", "content": """Please summarize the following Text: 'Inflation: Bank of England verliert zunehmend die Kontrolle
Im abschließenden Bericht dieser Woche empfiehlt es sich, einen Blick nach Großbritannien zu werfen, wo die jüngste Inflationsentwicklung vielerorts zu langen Gesichtern geführt hat. Eine weitere Zinserhöhung durch die Bank of England im Mai scheint unter Beobachtern und Kommentatoren an den Finanzmärkten eine ausgemachte Sache zu sein. Interessant ist, auf welche Weise hochrangige Vertreter der Bank of England auf diese Entwicklung reagieren und die britische Öffentlichkeit gegen sich aufbringen.
 28.04.2023  Autor: Wirtschaftsfacts-Gastautor
lazyllama / Shutterstock.com
lazyllama / Shutterstock.com
Mittlerweile werden die Auswirkungen, die eine vollkommen rücksichtslose Geldpolitik der Zentralbanken nach sich zieht, allerorten spürbar. Nicht nur in westlichen Industriestaaten, sondern auch in weiten Teilen des Rests der Welt sind die allgemeinen Lebenshaltungskosten in den vergangenen beiden Jahren förmlich durch die Decke geschossen.

Wenn auch die Krise an den internationalen Energiemärkten für den Moment wieder ein wenig in den Hintergrund der teils hitzig geführten Debatten gerückt ist, so stehen nach wie vor die vielerorts abhebenden Lebensmittel- und Mietpreise im Zentrum der Betrachtungen.

Reallöhne sinken auf breiter Front
Gestern veröffentlichte Daten zeigen, dass nicht nur die Schweizerischen Eidgenossen unter kräftig sinkenden Reallöhnen leiden, sondern dass der Kaufkraftschwund ebenfalls unter den Nationen der Eurozone, allen voran in Deutschland, spürbar zugenommen hat.

Viele private Haushalte halten sich aus diesem Grund jetzt schon seit einiger Zeit mit ihrem Konsum zurück. Dass die deutsche Wirtschaft im ersten Jahresquartal auf ein Nullwachstum blickte und damit nur knapp an einer Rezession vorbeigeschrammt ist, macht die Dinge nicht besser. Vielmehr erhärten diese Daten die Stagflationsbefürchtungen.

Hieß es seitens der Verantwortlichen in Regierungen und Zentralbanken bis vor Kurzem noch, dass sich Inflation nur als ein temporäres Phänomen erweisen würde, so lassen deren jüngst getätigte Aussagen aufhorchen. 

Kampf ums nackte wirtschaftliche Überleben
Mancherorts wird dem Kartell aus Regierungen und Zentralbanken an den Finanzmärkten bereits der offene Vorwurf der Heuchelei gemacht. Während viele kleine und mittelständische Unternehmen um ihr nacktes Überleben kämpfen und Ottonormalbürger ihren Ausgabegürtel immer enger schnallen, scheint die Ignoranz unter jenen, die die aktuelle Lebenshaltungskrise verursacht haben und weiter befördern, kaum mehr irgendwelche Grenzen zu kennen.

Kritisiert wird unter anderem auch, dass viele Aussagen unter Amtsträgern und Zentralbanken rein überhaupt nichts mehr mit der sich darstellenden Gesellschafts- und Wirtschaftsrealität zu tun hätten, weshalb Differenzen und Abweichungen in der Wahrnehmung samt der kognitiven Dissonanz einem Höhepunkt entgegen strebten.

Ein gutes Beispiel hierfür lieferte in dieser Woche Huw Pill, einer der Top-Ökonomen der Bank of England. Im Rahmen eines Podcast-Interviews forderte Hug Pill die britischen Unternehmen nicht nur dazu auf, Preissteigerungen nicht mehr an die privaten Haushalte  abzuwälzen, sondern provozierte die Bürger des Landes ferner mittels folgender Aussage:

„Akzeptieren Sie, dass Sie in der Zukunft finanziell schlechter gestellt und ärmer sein werden.“


 

Ferner sollte jedermann im Land damit aufhören sich an die Illusion zu klammern, dass die individuelle Kaufkraft durch höhere Lohnforderungen oder eine Weitergabe von steigenden Energiepreisen durch Unternehmen an deren Kunden aufrechterhalten werden könne.

Großbritanniens Inflation weiterhin im prozentual zweistelligen Bereich
Angemerkt sei, dass die Inflation in Großbritannien auch im letzten Monat im prozentual zweistelligen Bereich verharrte. Waren die Konsensschätzungen unter Volkswirten und Analysten von einer auf durchschnittlich 9,8 Prozent sinkenden Inflation ausgegangen, so lag die annualisierte Teuerungsrate im Monat März tatsächlich bei 10,1 Prozent, wie aus der nachfolgenden Grafik von tradingeconomics.com hervorgeht.


Es war bereits der zweite Monat in Folge, in dem die Inflation in Großbritannien die zuvor kursierenden Erwartungen unter Volkswirten und Analysten übertraf, was wahrscheinlich zu einer weiteren Zinsanhebung der Bank of England im Monat Mai führen wird.

Nachdem sich die Teuerungsrate bereits im Februar auf einem hartnäckig hohen Niveau von 10,4 Prozent gehalten hatte, wurden sinkende Energiepreise im Monat März durch weiter steigende Lebensmittelpreise und teils deutliche Preisanhebungen im Kulturbereich, darunter Theatervorstellungen, Sportveranstaltungen und Konzerte, aufgewogen.

Zinsen werden weiter steigen
Auch die Kerninflationsrate, somit also exklusive Energie und Lebensmittel, verharrte im vergangenen Monat bei 6,2 Prozent. An den Finanzmärkten sprechen diese Daten glasklar für eine weitere unmittelbar bevorstehende Zinsanhebung durch die Bank of England im Monat Mai um 25 Basispunkte auf dann 4,5 Prozent.


Dass die Bank of England nach wie vor an einer baldigen Rückkehr zum eigens anvisierten Inflationsziel von zwei Prozent festhält, halten viele Beobachter und Kommentatoren an den Finanzmärkten inzwischen für einen einsetzenden Realitätsverlust.

Ferner ist die Bank of England, ähnlich wie die Federal Reserve Bank in den Vereinigten Staaten, mittlerweile in den Fokus der öffentlichen Inflationsdebatten gerückt. Selbst jene, welche die geldpolitischen Entscheidungen der Bank of England in der Vergangenheit noch verteidigt hatten, kommen augenscheinlich nicht mehr umhin, als den eigentlichen Grund fürdie abhebende Preise in jener über die letzten Jahre erfolgten Ausweitung des Geldangebots durch die Bank of England auszumachen.

Aussagen unter Vertretern der Bank of England werden inzwischen als Provokation wahrgenommen
Es war nicht das erste Mal, dass Aussagen unter hochrangigen Vertretern der Bank of England zu einem Aufschrei in der britischen Bevölkerung samt einer massiv aufkommenden Kritik geführt haben.

Bereits im vergangenen Jahr forderte Andrew Bailey, der Gouverneur der Bank of England, die Arbeitnehmer des Landes öffentlich zu einer Zurückhaltung in Sachen Lohnerhöhungen auf. Auf diese Weise, so Andrew Bailey, könnten die Arbeitnehmer ihren Teil dazu beitragen, die Inflation im Land zu bekämpfen.

Zum damaligen Zeitpunkt hatte die Teuerungsrate auf Jahresbasis ein Niveau von rund sieben Prozent erreicht. Da die Inflationsrate seitdem weiter bis in den zweistelligen Prozentbereich geklettert ist, deute unter Bezugnahme auf öffentlichen Debatten im Vereinigten Königreich einiges darauf hin, dass die Bank of England die Situation nicht mehr unter Kontrolle habe.


 

Aus der oben abgebildeten Grafik von tradingeconomics.com geht das Konsumklima unter den britischen Verbrauchern im Verlauf der vergangenen zwölf Monate unter Bezugnahme auf Daten der GFK Group hervor.

Wie sich zeigt, verharrte das allgemeine Konsumklima trotz eines Anstiegs auch im Monat April weit unter der Nulllinie. Ins Bild passt, dass die britische Wirtschaft schon seit Frühjahr letzten Jahres stagniert und in den kommenden Quartalen schrumpfen könnte.


 

Deutlich wird, dass die Bank of England, ähnlich wie die Federal Reserve Bank in den USA, zurzeit dabei ist, kleine und mittelständische Unternehmen wie auch Privathaushalte förmlich von der Landkarte zu fegen. Stagflation ist ein Begriff, welcher dem Gros der Bevölkerung in Großbritannien inzwischen geläufig sein wird.

Fed-Chef Jerome Powell spricht aus dem Nähkästchen

Eingegangen wird abschließend noch darauf, dass nach EZB-Chefin Christine Lagarde nun auch Fed-Chef Jerome Powell einem falschen Selenskyj-Anrufer aufgesessen ist und diesem bereitwillig Auskunft erteilte. Die in diesem Telefonat durch Jerome Powell getätigten und ins Internet gestellten Aussagen waren dazu angetan, die Finanzmärte wachzurütteln.

Denn danach wird die Federal Reserve Bank ihren Leitzins nicht nur im Mai, sondern wohl auch nochmals im Juni um jeweils 25 Basispunkte auf dann 5,25 bis 5,5 Prozent anheben. Laut Jerome Powell soll danach dann erst einmal eine Zinsanhebungspause in den USA eintreten.

Anders als an den Finanzmärkten erwartet, sollen die Zinsen in den USA dann jedoch über einen längeren Zeitraum auf dem dann vorherrschenden Niveau verharren. Es deutet zurzeit also nichts darauf hin, als ob die Fed den Forderungen unter Akteuren an den Aktienmärkten nach Zinssenkungen alsbald nachzugeben bereit sein wird.

Diese Zusammenfassung für CK*Wirtschaftsfacts von Roman Baudzus nimmt unter anderem Bezug auf einen Bericht von GB News auf YouTube. 

Was heißt das für mich konkret? (Roman Baudzus)
Wer die nachfolgende Grafik der Federal Reserve Bank of St. Louis berücksichtigt, wird sich wahrscheinlich darüber bewusst sein, dass Zinssenku'"""},
{"role": "assistant", "content": "The Bank of England appears to be losing control over inflation in the UK, as it remains in double figures and private households feel a squeeze on their purchasing power due to rising living costs. A recent podcast interview with a top economist at the bank has caused public outcry, as he suggests that citizens should accept their future financial hardship and businesses should stop passing on price increases to consumers. The bank is seen to be provoking public anger and may be facing criticisms of hypocrisy, as inflationary pressures continue despite assurances from officials. The Bank of England is expected to boost interest rates by 25 basis points in May to 4.5%, but whether it can achieve its aim of reducing inflation to 2% is in question. The situation has given rise to stagflation fears, causing small- and medium-sized businesses and households to struggle. Meanwhile, in the US, Federal Reserve Bank Chairman Jerome Powell has incited speculation about a possible further rate hike beyond May, indicating that there may not be an immediate end to rate hikes in the US."},
{"role": "user", "content": "What are the inflation rates in England according to the article?"}
            ]
    gpt_connector = GPTConnector()
    print(await gpt_connector.get_chat_respone(messages))


if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass