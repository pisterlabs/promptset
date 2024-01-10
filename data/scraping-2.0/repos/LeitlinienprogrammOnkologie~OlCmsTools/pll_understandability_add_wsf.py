import pandas as pd
import openai
import nltk
import pyphen
from collections import Counter, defaultdict

nltk.data.load('tokenizers/punkt/german.pickle')
nltk.data.load('taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle')

dic = pyphen.Pyphen(lang="de_DE")

source_path = "//dkg-dc-01/Daten/Daten/Arbeitsverzeichnisse/OL/Vorträge/EbM-Kongress 2023/Verständlichkeitsanalyse PLL/2023-02-20_Verständlichkeit_ohne KI.xlsx"

df = pd.read_excel(source_path, sheet_name="Daten")

N = len(df)

def count_syllables_german(text):
    number_of_syllables = 0
    syllables_per_word = defaultdict(int)
    characters_per_word = defaultdict(int)

    for word in text.split(" "):
        # print(word)
        syllable_counter = 0
        # hyphenate word
        syllables = dic.inserted(word)
        # count first syllable of word
        syllable_counter += 1
        # and count the other syllables
        syllable_counter += syllables.count("-")
        number_of_syllables += syllable_counter
        syllables_per_word[syllable_counter] += 1
        characters_per_word[len(word)] += 1
        # print("  Chars: " + str(len(word)))
        # print("  Syllables: " + str(syllable_counter))

    return number_of_syllables, syllables_per_word, characters_per_word

def wiener_sachtext_formel(ms, sl, iw, es):
	"""https://de.wikipedia.org/wiki/Lesbarkeitsindex#Wiener_Sachtextformel
	Keyword arguments:
	MS -- Prozentanteil der Wörter mit drei oder mehr Silben,
	SL -- mittlere Satzlänge (Anzahl Wörter),
	IW -- Prozentanteil der Wörter mit mehr als sechs Buchstaben,
	ES -- Prozentanteil der einsilbigen Wörter.
	"""
	wsf = 0.1935 * ms + 0.1672 * sl + 0.1297 * iw - 0.0327 * es - 0.875
	return wsf

out_text = "# Sätze|# Wörter|# Silben|Silben/Wort|#Wörter (>3 Silben)|#Wörter (>6 Silben)|WSF\n"

for index, row in df.iterrows():
    text = row['Text']
    sents = nltk.sent_tokenize(text)
    number_of_sentences = len(sents)
    number_of_words = len(text.split(" "))
    number_of_syllables, syllables_per_word, characters_per_word = count_syllables_german(text)
    avg_sentence_length = number_of_words / number_of_sentences
    avg_number_of_syllables_per_word = number_of_syllables / number_of_words
    number_of_words_with_three_or_more_syllables = sum([v for k, v in syllables_per_word.items() if k >= 3])
    number_of_words_with_six_or_more_characters = sum([v for k, v in characters_per_word.items() if k >= 6])
    wsf = wiener_sachtext_formel(
        number_of_words_with_three_or_more_syllables / number_of_words * 100,
        avg_sentence_length,
        number_of_words_with_six_or_more_characters / number_of_words * 100,
        syllables_per_word[1] / number_of_words * 100)
    out_text += "%s|%s|%s|%s|%s|%s|%s\n" % (number_of_sentences, number_of_words, number_of_syllables, avg_number_of_syllables_per_word, number_of_words_with_three_or_more_syllables, number_of_words_with_six_or_more_characters, wsf)

out_path = "//dkg-dc-01/Daten/Daten/Arbeitsverzeichnisse/OL/Vorträge/EbM-Kongress 2023/Verständlichkeitsanalyse PLL/2023-03-14_Verständlichkeit_WSF.csv"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(out_text)