import pandas as pd
import openai
from PllUnderstandability.TextParser import IndexCalculator

openai.api_key = "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"
model_engine = "text-davinci-003"

'''
prompt = "switch to German language."
completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=4096-len(prompt),
    n=1,
    stop=None,
    temperature=0.5,
)

response = completion.choices[0].text
print(response)
'''

index_calculator = IndexCalculator()

def translate_text_to_easy_langage(text):
    prompt = "Folgenden Text in sehr leichte Sprache ohne Nebensätze umwandeln: %s" % text
    try:
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=4096-len(prompt),
            n=1,
            stop=None,
            temperature=0.5,
        )

        response = completion.choices[0].text
    except:
        response = "ERROR"
    return response

def add_to_csv(guideline, type, chapter, metric, text):
    if metric is not None and metric.number_words > 0:
        result = "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n" % (
            guideline,
            type,
            chapter,
            text,
            metric.number_chars,
            metric.number_words,
            metric.number_types,
            metric.number_sentences,
            metric.number_syllables,
            metric.number_polysyllable_words,
            metric.difficult_words,
            metric.number_words_longer_4,
            metric.number_words_longer_6,
            metric.number_words_longer_10,
            metric.number_words_longer_13,
            metric.ASL,
            metric.ASW,
            metric.flesch_reading_ease,
            metric.coleman_liau_index,
            metric.gunning_fog_index,
            metric.smog_index,
            metric.ari_index,
            metric.lix_index,
            metric.dale_chall_score
        )
    else:
        result = "%s|%s|%s|%s||||||||||||||||||||\n" % (
            guideline,
            type,
            chapter,
            text
        )
    return result


input_file = "//dkg-dc-01/Daten/Daten/Arbeitsverzeichnisse/OL/Vorträge/EbM-Kongress 2023/Verständlichkeitsanalyse PLL/2023-02-20_Verständlichkeit_ohne KI.xlsx"

df = pd.read_excel(input_file, sheet_name="Daten")

pll_rows = df.loc[df['Typ'] == 'PLL']

out_file = "Leitlinie|Kapitel|PLL-Text|PLL-ASL|PLL-ASW|PLL-LIX|PLL-FLESH_DE|DV3-Text|DV3-ASL|DV3-ASW|DV3-LIX|DV3-FLESH_DE|SUMM-Text|SUMM-ASL|SUMM-ASW|SUMM-LIX|SUMM-FLESH_DE\n"

N= len(pll_rows)
for index, row in pll_rows.iterrows():
    text = row['Text']
    translated_text = translate_text_to_easy_langage(text).replace("\n", " ").replace("\r", " ").replace("  ", " ").strip()
    if len(translated_text) > 0:
        metrics = index_calculator.Handle(translated_text)
        if metrics is not None:
            flesh_de = 180 * metrics.ASL - (58.5 * metrics.ASW)
            out_row = "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|||||\n" % (
                row['Leitlinie'], row['Kapitel'],
                text, row['ASL'], row['ASW'], row['LIX'], row['FLESH_DE'],
                translated_text, metrics.ASL, metrics.ASW, metrics.lix_index, flesh_de
                )
        else:
            out_row = "%s|%s|||||||||||||||\n" % (
                row['Leitlinie'], row['Kapitel'])

    out_file += out_row
    print ("%s / %s (%s)%%" % (index, N, 100*(index/N)))
    if index > 10:
        break

with open("pll_undestandability_with_KI.csv", "w", encoding="utf-8") as f:
    f.write(out_file)