from pytrends.request import TrendReq
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk
import openai
from transformers import pipeline
import os


openai.api_key = open(os.path.join('keys', 'open_ai.key')).read()
MAX_RESULTS = 10
nltk.download('wordnet')
nltk.download('punkt')
pytrend = TrendReq()


def get_google_trends_related_queries(query):
	print('google_trends_related_queries_sample')
	# Get related queries
	pytrend.build_payload(kw_list=[query])
	
	related_queries = pytrend.related_queries()[query]

	# Print the top 10 rising related queries
	ret = list(related_queries["rising"]["query"])
    
	# Print the top 10 top related queries
	ret += list(related_queries["rising"]["query"])
	return ret
        

def get_google_trends_related_topics(query):
	print('google_trends_related_topics_sample')
	# Get related queries
	pytrend.build_payload(kw_list=[query])

	related_topics = pytrend.related_topics()[query]
	# Print the top 10 rising related topics
	df = related_topics["rising"]

	ret = ['{}-{}'.format(x, y) for x, y in zip(df['topic_title'], df['topic_type'])]

	# Print the top 10 top related topics
	df = related_topics["top"]

	ret += ['{}-{}'.format(x, y) for x, y in zip(df['topic_title'], df['topic_type'])]
	return ret


def get_google_trends_suggestions(query):
	print('google_trends_suggestions_sample')
	keywords = pytrend.suggestions(keyword=query)

	ret = ['{}-{}'.format(q['title'], q['type']) for q in keywords]
	return ret
	

def get_synsets_word_related_phrases(word):
    # Get synsets for the given word
    synsets = wordnet.synsets(word)

    # Collect related words and phrases
    related = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            # Add synonyms
            if lemma.name() != word:
                related.add(lemma.name().replace("_", " "))

            # Add antonyms
            for antonym in lemma.antonyms():
                related.add(antonym.name().replace("_", " "))

    return list(related)


def get_synsets_phrase_related_phrases(query):
	print('synsets_phrase_related_phrases_sample')
	# Tokenize input phrase into words
	words = word_tokenize(query.lower())

	related = set()
	for word in words:
		related.update(get_synsets_word_related_phrases(word))

	return list(related)


def get_chatgpt_phrases(query):
	print('chatgpt_phrases_sample')

	def generate_ideas(prompt):
		response = openai.Completion.create(
		engine="text-davinci-003",
		prompt=prompt,
		max_tokens=100 * MAX_RESULTS,
		stop=None,
		temperature=0.5,
		)

		ideas = [choice.text.strip() for choice in response.choices]
		return ideas
	
	prompt = "Generate %d phrases related to '%s' useful for tech-startup" % (MAX_RESULTS, query)
	ideas = generate_ideas(prompt)
        
	assert len(ideas) == 1, ideas
	ideas = ideas[0].split('\n')
	ideas = [idea[3:] for idea in ideas if idea.strip()]

	return ideas


def get_chatgpt_ideas(query):
	print('chatgpt_ideas_sample')

	def generate_ideas(prompt):
		prompts = [prompt for _ in range(MAX_RESULTS)]
		response = openai.Completion.create(
			engine="text-davinci-003",
			prompt=prompts,
			max_tokens=500,
			stop=None,
			temperature=0.5,
		)

		ideas = [choice.text.strip() for choice in response.choices]
		return ideas

	prompt = "Generate idea or fact or insight related to '%s' that can be useful for tech-startup" % (query)
	ret = generate_ideas(prompt)
	return ret


def get_hugging_face_inference(query):
	print('hugging_face_inference_sample')

	generator = pipeline('text-generation', model = 'gpt2')
	prompt = "New idea/fact/insight related to the %s is " % query
	results = generator(prompt, max_length = 200, num_return_sequences=MAX_RESULTS)

	ideas = [r['generated_text'][len(prompt) + 1:].replace('\n\n', '\n') for r in results]
	return ideas


def main():
    print(get_google_trends_related_queries('coffee'))
    print(get_google_trends_related_topics('coffee'))
    print(get_google_trends_suggestions('apple headphones'))
    print('synsets_word_related_phrases_sample')
    print(get_synsets_word_related_phrases('whatever'))
    print(get_synsets_phrase_related_phrases('I want this cup of coffee for the morning'))
    print(get_chatgpt_phrases('cloud computing'))
    print(get_chatgpt_ideas('seamless processing'))
    print(get_hugging_face_inference('climate change'))


if __name__ == '__main__':
    main()