from wordfreq import word_frequency
import spacy
from openai import OpenAI
import openai

client = OpenAI(api_key='')

def check_freq(sentence):

  nlp = spacy.load("en_core_web_sm")
  words = [tok.lemma_ for tok in nlp(sentence) if tok.pos_ not in ["PUNCT", "SPACE"]]

  freq_dict = {}
  for word in words:
    freq = word_frequency(word, 'en')
    freq_dict[word] = freq

  vocab = dict(sorted(freq_dict.items(), key=lambda item: item[1]))
  return vocab

def words_exp(sentence):

  freq_dict = check_freq(sentence)

  explain = dict((k, v) for k, v in freq_dict.items() if v < 1e-4)

  words = list(explain.keys())

  return words

def simplify_words(sentence):
  words = words_exp(sentence)
  assis = 'Give me a new verison of sentences which paraphrase these words in super easy-to-understand words:'
  for word in words:
    assis += word
    assis += ', '
  assis += 'inside sentences and only give me the new version of explained sentences.'

  completion = client.chat.completions.create(
      model='gpt-4-1106-preview',
      messages=[
         {"role": "system",
         "content": assis},
        {"role": "user",
         "content": sentence}
      ],
      temperature=0.5,
      top_p=0.5,
  )

  return completion.choices[0].message.content

def passive_transform(sentence):

  completion = client.chat.completions.create(
      model='gpt-4-1106-preview',
      messages=[
        {"role": "system",
         "content": "Change all passive sentences into active sentences."},
        {"role": "user",
         "content": sentence}
      ],
      temperature=0.5,
      top_p=0.5,
  )

  return completion.choices[0].message.content

def simplify_structure(sentence):
  completion = client.chat.completions.create(
      model='gpt-4-1106-preview',
      messages=[
        {"role": "system",
         "content": "Change all sentences into simple sentences without missing any important details and ensure the sentences are readable and coherent. You can break original sentences down to achieve this. Also do not increase words' complexity."},
        {"role": "user",
         "content": sentence}
      ],
      temperature=0.5,
      top_p=0.5,
  )
  return completion.choices[0].message.content

def simplify_phrases(sentence):
  completion = client.chat.completions.create(
      model='gpt-4-1106-preview',
      messages=[
        {"role": "system",
         "content": "Replace compound adjectives and challenging phrases with super simple synonyms."},
        {"role": "user",
         "content": sentence}
      ],
      temperature=0.5,
      top_p=0.5,
  )
  return completion.choices[0].message.content

def output(sentence):
    tmp = passive_transform(sentence)
    tmp2 = simplify_structure(tmp)
    tmp3 = simplify_phrases(tmp2)
    tmp4 = simplify_words(tmp3)
    tmp5 = simplify_structure(tmp4)
    output = simplify_words(tmp5)
    return output
