from openai import OpenAI
from wordfreq import word_frequency
import spacy

client = OpenAI(api_key='')
T = 0.5
t = 0.5

def explain_sentences(statement):
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": f"Explain the following sentences. Only give me your explanation in a paragraph. Do not involve original sentences: {statement}",
            }
        ],
        temperature=0.5,
        top_p=0.5,
    )
    return completion.choices[0].message.content
  
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

def explain_words(sentence):
  words = words_exp(sentence)
  assis = 'Give me a new verison of sentences which correctly explain these words in simpler sentences or correctly paraphrase with super easy-to-understand words(consider their academic meaning if neccesary):'
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
      temperature=T,
      top_p=t,
  )

  return completion.choices[0].message.content

def examples(sentence):
    completion = client.chat.completions.create(
      model='gpt-4-1106-preview',
      messages=[
        {"role": "system",
         "content": "You are a helpful assistant."},
        {"role": "user",
         "content": f"Add one concise sentence of example if it has, to {sentence} for better understanding. Give me the sentences after adding this example."}
      ],
      temperature=0.5,
      top_p=0.5
    )
    return completion.choices[0].message.content

def simplify_structure(sentence):
  completion = client.chat.completions.create(
      model='gpt-4-1106-preview',
      messages=[
        {"role": "system",
         "content": "Break all sentences into simple sentences without missing any important details and ensure the sentences are readable and coherent. Also do not increase words' complexity, do not give me several points but a coherent paragraph."},
        {"role": "user",
         "content": sentence}
      ],
      temperature=0.5,
      top_p=0.5,
  )
  return completion.choices[0].message.content

def explain(sent):
  tmp = explain_sentences(sent)
  tmp2 = examples(tmp)
  tmp3 = explain_words(tmp2)
  tmp4 = simplify_structure(tmp3)
  result = explain_words(tmp4)
  return result
